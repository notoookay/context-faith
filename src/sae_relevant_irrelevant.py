import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from tqdm import tqdm
import einops
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.metrics import roc_curve, auc
from datasets import load_dataset
import wandb
from transformers import AutoTokenizer

from transformer_lens import HookedTransformer, ActivationCache
from sae_lens import SAE

from utils import format_prompt, MODEL_ALIAS, LLAMA_3_1_8B_INSTRUCT_PROMPT_FORMAT

# Default parameters
DEFAULT_CONFIG = {
    "scoring_method": "absolute_difference",
    "min_activations": None,
    "entity_types": ["relevant", "irrelevant"]
}

def parse_args():
    """Parse command line arguments for the relevance detection analysis."""
    parser = argparse.ArgumentParser(description="Analyze relevance detection in language models using SAEs")
    
    # Required arguments
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name to analyze (e.g., 'google/gemma-2-2b-it')")
    parser.add_argument("--relevant_data_file", type=str, required=True,
                        help="Path to the JSONL data file with relevant contexts")
    parser.add_argument("--irrelevant_data_file", type=str, required=True,
                        help="Path to the JSONL data file with irrelevant contexts")
    parser.add_argument("--sae_repo_id", type=str, required=True,
                        help="Repository ID for the pre-trained SAEs")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="output/sae_relevant_irrelevant",
                        help="Directory to save the output results")
    parser.add_argument("--hook_name", type=str, default="resid_post",
                        help="Hook name used for activation extraction; options: 'resid_post', 'resid_pre', 'resid_mid'")
    parser.add_argument("--prompt_type", type=str, default="with_passage",
                        help="Type of prompt template to use")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layers to analyze")
    parser.add_argument("--scoring_method", type=str, default="absolute_difference",
                        choices=["absolute_difference", "relative_difference", "t_test"],
                        help="Method to score feature activations")
    parser.add_argument("--filter_threshold", type=float, default=0.02,
                        help="Threshold for filtering latents by activation frequency")
    parser.add_argument("--pos_to_analyze", type=str, default="-1",
                        help="Comma-separated list of positions to analyze in the generation (e.g., '-2,-1')")
    parser.add_argument("--save_relevant_activations", type=str, default=None,
                        help="Path to save the relevant activations")
    parser.add_argument("--save_irrelevant_activations", type=str, default=None,
                        help="Path to save the irrelevant activations")
    parser.add_argument("--load_relevant_activations", type=str, default=None,
                        help="Path to load the relevant activations")
    parser.add_argument("--load_irrelevant_activations", type=str, default=None,
                        help="Path to load the irrelevant activations")
    parser.add_argument("--top_k_features", type=int, default=20,
                        help="Number of top features to return")

    parser.add_argument("--min_activations", type=str, default=None,
                        help="Comma-separated minimum activation thresholds for relevant,irrelevant entities (e.g., '0.05,0.05')")
    parser.add_argument("--test_third_last_newline", action="store_true",
                        help="Test with third-last newline positions")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for activations")
    # Wandb related arguments
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="relevance-detection",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Weights & Biases run name")
    
    return parser.parse_args()

def load_model(model_name: str, device: str=None):
    """Load model using HookedTransformer."""
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
    print(f"Loading model {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype='bfloat16')
    model.eval()
    return model

def load_sae(repo_id: str, sae_id: str):
    """Load a pre-trained Sparse Autoencoder."""
    print(f"Loading SAE {sae_id} from {repo_id}")
    # Here we use cpu as the vram constraint, and inference speed is not an issue
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        repo_id, sae_id, device="cuda"
    )
    return sae, cfg_dict, sparsity

def get_model_inputs(item, model_name: str, tokenizer: AutoTokenizer, prompt_type: str, is_chat_model: bool = False, relevance_type: str = "relevant"):
    """
    Get model inputs for a question with a single context type.
    
    Args:
        item: Dataset item containing question, answers and context information
        model_name: Model name
        tokenizer: Tokenizer instance
        prompt_type: Type of prompt template to use
        is_chat_model: Whether the model is a chat model
        relevance_type: Either "relevant" or "irrelevant" to indicate the type of context
    
    Returns:
        dict: Contains model input and relevance information
    """
    question = item["question"]
    passages = item["ctxs"]
    
    # Check if this is a Llama 3.1 model
    use_llama_format = "llama-3.1" in model_name.lower()
    
    # Process the passage (assuming single passage per item now)
    passage = passages[0] if isinstance(passages, list) else passages
    prompt = format_prompt(question, passage_list=[passage], prompt_type=prompt_type, base_model_prompt=not is_chat_model)
    
    # Create the full conversation
    if is_chat_model:
        if use_llama_format:
            model_input = LLAMA_3_1_8B_INSTRUCT_PROMPT_FORMAT.format(prompt=prompt)
        else:
            model_input = tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        model_input = prompt
    
    # Determine relevance based on the type specified
    relevant = (relevance_type == "relevant")
    
    return {
        'model_input': model_input,
        'relevance': relevant,
        'question': question,
        'answers': item['answers']
    }

def get_features(
    sae_acts: Dict[str, torch.Tensor],
    metric: str = 'relative_difference',
    min_activations: Optional[List[float]] = None,
    eps: float = 1e-6,
):
    """
    Calculate feature scores that separate relevant from irrelevant entity activations.
    
    Args:
        sae_acts: Dictionary with 'relevant' and 'irrelevant' activation tensors
        metric: Scoring method ('relative_difference', 'absolute_difference', 't_test')
        min_activations: Optional minimum activation thresholds. A list of two float values between 0 and 1:
            - min_activations[0]: Threshold for irrelevant entity activations (freq_acts_1)
            - min_activations[1]: Threshold for relevant entity activations (freq_acts_0)
            When using 'absolute_difference' method, only features with activation frequencies 
            below these thresholds will be considered for scoring.
        eps: Small value to avoid division by zero
    
    Returns:
        scores_dict: Dictionary of scores for each feature
        freq_acts_dict: Dictionary of activation frequencies
        mean_features_acts: Dictionary of mean activation values
    """
    # Compute frequency of activations (how often a feature activates)
    if sae_acts['relevant'].shape[0] == 0:
        # If no activation for relevant, we set the frequency to zero
        freq_acts_0 = torch.zeros(sae_acts['irrelevant'].shape[1], device=sae_acts['irrelevant'].device)
    else:
        freq_acts_0 = (sae_acts['relevant'] > eps).float().mean(dim=0)
        
    if sae_acts['irrelevant'].shape[0] == 0:
        # If no activation for irrelevant, we set the frequency to zero
        freq_acts_1 = torch.zeros(sae_acts['irrelevant'].shape[1], device=sae_acts['irrelevant'].device)
    else:
        freq_acts_1 = (sae_acts['irrelevant'] > eps).float().mean(dim=0)

    # Calculate scores based on selected metric
    if metric == 'relative_difference':
        scores_0 = (freq_acts_0 - freq_acts_1) / (freq_acts_1 + eps)
        scores_1 = (freq_acts_1 - freq_acts_0) / (freq_acts_0 + eps)
    
    elif metric == 'absolute_difference':
        if min_activations is not None:
            mask_1 = freq_acts_1 < min_activations[0]
            mask_0 = freq_acts_0 < min_activations[1]
            scores_0 = ((freq_acts_0 - freq_acts_1) * mask_1)
            scores_1 = ((freq_acts_1 - freq_acts_0) * mask_0)
        else:
            scores_0 = (freq_acts_0 - freq_acts_1)
            scores_1 = (freq_acts_1 - freq_acts_0)
    
    elif metric == 't_test':
        scores_0 = []
        scores_1 = []
        for i in range(0, sae_acts['relevant'].shape[1]):
            if sae_acts['relevant'][:,i].sum() == 0 and sae_acts['irrelevant'][:,i].sum() == 0:
                scores_0.append(0.0)
                scores_1.append(0.0)
            else:
                scores_0.append(stats.ttest_ind(
                    sae_acts['relevant'][:,i].detach().cpu().numpy(), 
                    sae_acts['irrelevant'][:,i].detach().cpu().numpy(), 
                    axis=0, equal_var=False
                ).statistic)
                scores_1.append(stats.ttest_ind(
                    sae_acts['irrelevant'][:,i].detach().cpu().numpy(), 
                    sae_acts['relevant'][:,i].detach().cpu().numpy(), 
                    axis=0, equal_var=False
                ).statistic)
        scores_0 = torch.tensor(scores_0, device=freq_acts_0.device)
        scores_1 = torch.tensor(scores_1, device=freq_acts_1.device)
        scores_0 = torch.nan_to_num(scores_0, nan=0.0)
        scores_1 = torch.nan_to_num(scores_1, nan=0.0)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    # Prepare return values
    scores_dict = {}
    freq_acts_dict = {}
    mean_features_acts = {}
    
    for relevance_label, scores in zip(['relevant', 'irrelevant'], [scores_0, scores_1]):
        scores_dict[relevance_label] = scores.tolist() if isinstance(scores, torch.Tensor) else scores
        # Frequency of activation in relevant and irrelevant prompts
        freq_acts_dict[relevance_label] = (freq_acts_0.cpu().tolist(), freq_acts_1.cpu().tolist())
        mean_features_acts[relevance_label] = sae_acts[relevance_label].mean(0).detach().cpu().tolist()

    return scores_dict, freq_acts_dict, mean_features_acts

def format_layer_features(feats_per_layer, save_dir=None):
    """
    Format the feature scores and metrics for each layer.
    
    Args:
        feats_per_layer: Dictionary of features per layer and position
        save_dir: Optional directory to save results
    
    Returns:
        Dictionary of formatted features by layer and position
    """
    final_feats_dict = {}
    
    for layer in feats_per_layer.keys():
        final_feats_dict[layer] = {}
        
        for pos in feats_per_layer[layer].keys():
            final_feats_layer_dict = {}
            scores_dict, freq_acts_dict, mean_features_acts_dict = feats_per_layer[layer][pos]
            
            for relevance_label in ['relevant', 'irrelevant']:
                final_feats_layer_dict[relevance_label] = {}
                scores = scores_dict[relevance_label]
                freq_acts = freq_acts_dict[relevance_label]
                mean_features_acts = mean_features_acts_dict[relevance_label]
                
                # Get indices of all features
                full_top_feats = list(range(len(scores)))
                
                # Get ordered indices by score (descending)
                ordered_indices = np.argsort(np.array(scores))[::-1]
                
                # Format and store results for each feature
                for i, idx in enumerate(ordered_indices):
                    final_feats_layer_dict[relevance_label][str(i)] = {
                        'layer': layer,
                        'position': pos,
                        'latent_idx': idx,
                        'score': scores[idx],
                        'freq_acts_relevant': freq_acts[0][idx],
                        'freq_acts_irrelevant': freq_acts[1][idx],
                        'mean_features_acts': mean_features_acts[idx]
                    }
            
            # Save results if a directory is provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                output_file = os.path.join(save_dir, f"L_{str(layer)}_P_{str(pos)}.json")
                with open(output_file, 'w') as f:
                    json.dump(final_feats_layer_dict, f, indent=4, default=float)
            
            final_feats_dict[layer][pos] = final_feats_layer_dict
    
    return final_feats_dict

def get_top_k_features(feats_layers, k=10):
    """
    Get top k features across all layers that best separate relevant and irrelevant examples.
    
    Args:
        feats_layers: Dictionary of feature layers
        k: Number of top features to return (None for all)
    
    Returns:
        Dictionary of top features for relevant and irrelevant entity detection
    """
    final_feats_dict = {}
    
    for relevance_label in ['relevant', 'irrelevant']:
        final_feats_dict[relevance_label] = {}
        
        # Collect all features across layers
        full_scores = []
        full_layers_list = []
        full_latent_ids = []
        full_top_freq_acts_0 = []
        full_top_freq_acts_1 = []
        full_top_mean_features_acts = []
        
        for layer in feats_layers.keys():
            layer_relevance_label_dict = feats_layers[layer][relevance_label]
            
            for idx in layer_relevance_label_dict.keys():
                full_scores.append(layer_relevance_label_dict[idx]['score'])
                full_layers_list.append(layer)
                full_latent_ids.append(layer_relevance_label_dict[idx]['latent_idx'])
                full_top_freq_acts_0.append(layer_relevance_label_dict[idx]['freq_acts_relevant'])
                full_top_freq_acts_1.append(layer_relevance_label_dict[idx]['freq_acts_irrelevant'])
                full_top_mean_features_acts.append(layer_relevance_label_dict[idx]['mean_features_acts'])
        
        # Join scores across all layers and get top indices
        full_scores_array = np.array(full_scores)
        
        if k is not None:
            # Get only top k indices
            top_k_indices = np.argsort(full_scores_array)[-k:][::-1]
        else:
            # Get all indices sorted by score
            top_k_indices = np.argsort(full_scores_array)[::-1]
        
        # Format results for each top feature
        for i, idx in enumerate(top_k_indices):
            final_feats_dict[relevance_label][i] = {
                'layer': full_layers_list[idx],
                'latent_idx': full_latent_ids[idx],
                'score': full_scores_array[idx],
                'freq_acts_relevant': full_top_freq_acts_0[idx],
                'freq_acts_irrelevant': full_top_freq_acts_1[idx],
                'mean_features_acts': full_top_mean_features_acts[idx]
            }
    
    return final_feats_dict

def plot_top_features(features_dict, pos_to_analyze, output_dir=None):
    """
    Plot the top features for relevant and irrelevant entity recognition.
    
    Args:
        features_dict: Dictionary of top features
        pos_to_analyze: Position to analyze
        output_dir: Optional directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    colors = {'relevant': 'green', 'irrelevant': 'red'}
    
    for label in ['relevant', 'irrelevant']:
        x = []
        y = []
        
        for feature_id in features_dict[label].keys():
            x.append(features_dict[label][feature_id]['freq_acts_relevant'] * 100)
            y.append(features_dict[label][feature_id]['freq_acts_irrelevant'] * 100)
        
        plt.scatter(x, y, color=colors[label], alpha=0.6, label=f"{label.capitalize()} Latents")
    
    # Add diagonal line for reference
    plt.plot([0, 100], [0, 100], color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Activation Frequency for Relevant (%)')
    plt.ylabel('Activation Frequency for Irrelevant (%)')
    plt.title(f'Feature Separation')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"relevance_features_{pos_to_analyze}.png"), dpi=300, bbox_inches='tight')
        
    return plt.gcf()

def find_generation_pos(model_input_tokens, generation_tokens):
    """
    Find the position of the generation in the model input tokens

    Args:
        model_input_tokens: list of tokens of the model input
        generation_tokens: list of tokens of the generation

    Returns:
        pos: a tuple of (start, end) position of the generation in the model input tokens
    """
    model_input_tokens = model_input_tokens[0]
    generation_tokens = generation_tokens[0]

    for i in range(len(model_input_tokens) - len(generation_tokens) + 1):
        if model_input_tokens[i:i + len(generation_tokens)].equal(generation_tokens):
            return (i, i + len(generation_tokens))
    return None

def extract_exact_answer_token_generation(pos, generation_pos):
    """
    Extract the position of the exact answer in the generation

    Args:
        pos: a list of the position of the exact answer in the model input tokens
        generation_pos: the position of the generation in the model input tokens

    Returns:
        generation_pos: a list of the position of the exact answer in the generation
    """
    exact_positions = []
    for p in pos:
        if generation_pos[0] <= p[0] and p[-1] <= generation_pos[1]:
            exact_positions.append(p)
    return exact_positions

def save_model_activations(activations: Dict, save_path: str):
    """
    Save model activations to disk.
    
    Args:
        activations: Dictionary of activations from get_model_activations
        save_path: Path to save the activations
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(activations, save_path)
    print(f"Saved activations to {save_path}")

def load_model_activations(load_path: str, pos_to_analyze: Union[int, List[int]] = None) -> Dict:
    """
    Load model activations from disk.
    
    Args:
        load_path: Path to load the activations from
        pos_to_analyze: Optional position or list of positions to load. If None, loads all positions.
        
    Returns:
        Dictionary of activations
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Activation file not found at {load_path}")
    
    activations = torch.load(load_path)
    print(f"Loaded activations from {load_path}")
    
    # If specific positions requested, filter the activations
    if pos_to_analyze is not None:
        
        # Create a new dictionary with only the requested positions
        filtered_activations = {}
        for layer in activations:
            filtered_activations[layer] = {}
            for pos in pos_to_analyze:
                if pos in activations[layer]:
                    filtered_activations[layer][pos] = activations[layer][pos]
                else:
                    print(f"Warning: Position {pos} not found in saved activations")
        
        activations = filtered_activations
    
    return activations

def load_combined_activations(relevant_load_path: str, irrelevant_load_path: str, pos_to_analyze: Union[int, List[int]] = None) -> Dict:
    """
    Load model activations from two separate files (relevant and irrelevant) and combine them.
    
    Args:
        relevant_load_path: Path to load the relevant activations from
        irrelevant_load_path: Path to load the irrelevant activations from
        pos_to_analyze: Optional position or list of positions to load. If None, loads all positions.
        
    Returns:
        Dictionary of combined activations in the format:
        {layer: {pos: {'relevant': Tensor, 'irrelevant': Tensor}}}
    """
    if not os.path.exists(relevant_load_path):
        raise FileNotFoundError(f"Relevant activation file not found at {relevant_load_path}")
    if not os.path.exists(irrelevant_load_path):
        raise FileNotFoundError(f"Irrelevant activation file not found at {irrelevant_load_path}")
    
    # Load both activation files
    relevant_activations = torch.load(relevant_load_path)
    irrelevant_activations = torch.load(irrelevant_load_path)
    print(f"Loaded relevant activations from {relevant_load_path}")
    print(f"Loaded irrelevant activations from {irrelevant_load_path}")
    
    # Combine into expected format
    combined_activations = {}
    
    # Get all layers present in both files
    relevant_layers = set(relevant_activations.keys())
    irrelevant_layers = set(irrelevant_activations.keys())
    common_layers = relevant_layers.intersection(irrelevant_layers)
    
    if not common_layers:
        raise ValueError("No common layers found between relevant and irrelevant activation files")
    
    for layer in common_layers:
        combined_activations[layer] = {}
        
        # Get all positions present in both files for this layer
        relevant_positions = set(relevant_activations[layer].keys())
        irrelevant_positions = set(irrelevant_activations[layer].keys())
        common_positions = relevant_positions.intersection(irrelevant_positions)
        
        if not common_positions:
            print(f"Warning: No common positions found for layer {layer}")
            continue
        
        for pos in common_positions:
            # Filter by requested positions if specified
            if pos_to_analyze is not None:
                if isinstance(pos_to_analyze, int):
                    pos_to_analyze = [pos_to_analyze]
                if pos not in pos_to_analyze:
                    continue
            
            combined_activations[layer][pos] = {
                'relevant': relevant_activations[layer][pos],
                'irrelevant': irrelevant_activations[layer][pos]
            }
    
    return combined_activations

def get_model_activations(model, dataset, layers, pos_to_analyze=-1, save_relevant_path=None, save_irrelevant_path=None, load_relevant_path=None, load_irrelevant_path=None, hook_name="resid_post"):
    """
    Get model activations for a list of examples.
    
    Args:
        model: HookedTransformer model
        dataset: Dataset of input examples
        layers: List of layers to collect activations from
        pos_to_analyze: Position or list of positions to analyze
        save_relevant_path: Optional path to save relevant activations
        save_irrelevant_path: Optional path to save irrelevant activations
        load_relevant_path: Optional path to load relevant activations from
        load_irrelevant_path: Optional path to load irrelevant activations from
        hook_name: Name of the hook to extract activations from (default: resid_post)
    
    Returns:
        Dictionary of activations for relevant and irrelevant entities by layer and position
    """
    # Convert single position to list if needed
    if isinstance(pos_to_analyze, int):
        pos_to_analyze = [pos_to_analyze]
    
    # Try to load activations if both paths provided
    if load_relevant_path is not None and load_irrelevant_path is not None:
        try:
            return load_combined_activations(load_relevant_path, load_irrelevant_path, pos_to_analyze)
        except FileNotFoundError as e:
            print(f"Could not load activations: {e}. Computing them instead")
    
    # Initialize nested dictionary structure
    activations = {
        layer: {
            pos: {'relevant': [], 'irrelevant': []} 
            for pos in pos_to_analyze
        } 
        for layer in layers
    }
    
    for example in tqdm(dataset, desc="Getting model activations"):
        # Process each generation for this example
        assert len(example['model_inputs']) == len(example['relevances']), "Testing the passage with multiple model inputs, should have the same number of relevances"
        for model_input, relevant in zip(example['model_inputs'], example['relevances']):
            # Get model input tokens
            model_input_tokens = model.to_tokens(model_input, prepend_bos=False)
            
            # Run model with cache
            with torch.no_grad():
                _, cache = model.run_with_cache(model_input_tokens)
            
            for pos in pos_to_analyze:
                token_to_analyze = model.to_string(model_input_tokens[0][pos])
                print(f"Token to analyze at position {pos}: {token_to_analyze!r}")
            # For each layer, cache all positions at once
            for layer in layers:
                # Get all residual stream activations for this layer
                all_acts = cache[hook_name, layer][0]  # Remove batch dimension
                
                # Process each requested position
                for pos in pos_to_analyze:
                    
                    # Get activation for this position
                    act = all_acts[pos].cpu()
                    
                    # Add to appropriate list based on entity status
                    if relevant:
                        activations[layer][pos]['relevant'].append(act)
                    else:
                        activations[layer][pos]['irrelevant'].append(act)
                

    
    # Stack tensors for each category, layer, and position
    for layer in layers:
        for pos in pos_to_analyze:
            if activations[layer][pos]['relevant']:
                activations[layer][pos]['relevant'] = torch.stack(activations[layer][pos]['relevant'])
            else:
                activations[layer][pos]['relevant'] = torch.zeros((0, model.cfg.d_model), device=model.cfg.device)
            
            if activations[layer][pos]['irrelevant']:
                activations[layer][pos]['irrelevant'] = torch.stack(activations[layer][pos]['irrelevant'])
            else:
                activations[layer][pos]['irrelevant'] = torch.zeros((0, model.cfg.d_model), device=model.cfg.device)
            
    
    # Save activations to separate files if paths provided
    if save_relevant_path is not None or save_irrelevant_path is not None:
        # Extract relevant and irrelevant activations into separate formats
        relevant_only = {}
        irrelevant_only = {}
        
        for layer in layers:
            relevant_only[layer] = {}
            irrelevant_only[layer] = {}
            for pos in pos_to_analyze:
                relevant_only[layer][pos] = activations[layer][pos]['relevant']
                irrelevant_only[layer][pos] = activations[layer][pos]['irrelevant']
        
        if save_relevant_path is not None:
            save_model_activations(relevant_only, save_relevant_path)
            print(f"Saved relevant activations to {save_relevant_path}")
        
        if save_irrelevant_path is not None:
            save_model_activations(irrelevant_only, save_irrelevant_path)
            print(f"Saved irrelevant activations to {save_irrelevant_path}")
    
    return activations

def get_sae_activations(model_acts, model_alias, layers, sae_repo_id):
    """
    Get SAE activations for model activations.
    
    Args:
        model_acts: Dictionary of model activations by layer and position
        model_alias: Model alias string
        layers: List of layers
        sae_repo_id: Repository ID for pre-trained SAEs
    
    Returns:
        Dictionary of SAE activations by layer and position
    """
    sae_acts = {}
    
    for layer in layers:
        sae_acts[layer] = {}
        
        # Load SAE for this layer
        if 'gemma-2' in model_alias:
            sae_id = f"layer_{layer}/width_16k/canonical"
        elif 'llama-3.1-8b' in model_alias:
            sae_id = f"l{layer}r_8x"
        else:
            raise ValueError(f"Unsupported model: {model_alias}")
        
        # Load the SAE
        sae, cfg_dict, sparsity = load_sae(sae_repo_id, sae_id)
        
        # Process each position
        for pos in model_acts[layer].keys():
            # Encode activations for this position
            relevant_encoded = sae.encode(model_acts[layer][pos]['relevant'].to('cuda')) if model_acts[layer][pos]['relevant'].shape[0] > 0 else torch.zeros((0, sae.cfg.d_sae), device=model_acts[layer][pos]['relevant'].device)
            irrelevant_encoded = sae.encode(model_acts[layer][pos]['irrelevant'].to('cuda')) if model_acts[layer][pos]['irrelevant'].shape[0] > 0 else torch.zeros((0, sae.cfg.d_sae), device=model_acts[layer][pos]['irrelevant'].device)
            
            sae_acts[layer][pos] = {
                'relevant': relevant_encoded.cpu(),
                'irrelevant': irrelevant_encoded.cpu()
            }
            

        
        # # Clean up to save memory
        # del sae
        # torch.cuda.empty_cache()
    
    return sae_acts



def analyze_relevance_features(model, dataset, layers, args):
    """
    Analyze relevance features using SAEs.
    
    Args:
        model: HookedTransformer model
        examples: List of examples with relevant/irrelevant
        layers: List of layers to analyze
        args: Command line arguments
    
    Returns:
        Dictionary with analysis results
    """
    # Parse positions to analyze
    if args.test_third_last_newline: # the first position after the passage
        example = dataset[0]
        example_tokens = model.to_tokens(example['model_inputs'][0], prepend_bos=False)
        pos_to_analyze = [find_third_last_newline_token_pos(example_tokens[0], model.tokenizer)]
        # turn into the negative index for generality
        if pos_to_analyze[0] != -1: # Check if a valid position was found
            pos_to_analyze = [pos_to_analyze[0] - len(example_tokens[0])]
        else:
            print("Warning: Third last newline token not found. Defaulting to standard positions.")
            pos_to_analyze = [int(p) for p in args.pos_to_analyze.split(',')]
    else:
        pos_to_analyze = [int(p) for p in args.pos_to_analyze.split(',')]
    print(f"Analyzing positions: {pos_to_analyze}")
    
    # Get model activations for all positions in one run
    # Set up save paths if not provided
    if args.save_relevant_activations is None:
        args.save_relevant_activations = os.path.join(args.output_dir, f'relevant_activations_{args.prompt_type}_{"_".join([str(p) for p in pos_to_analyze])}.pt')
    if args.save_irrelevant_activations is None:
        args.save_irrelevant_activations = os.path.join(args.output_dir, f'irrelevant_activations_{args.prompt_type}_{"_".join([str(p) for p in pos_to_analyze])}.pt')
    
    model_acts = get_model_activations(
        model, 
        dataset, 
        layers,
        pos_to_analyze=pos_to_analyze,
        save_relevant_path=args.save_relevant_activations,
        save_irrelevant_path=args.save_irrelevant_activations,
        load_relevant_path=args.load_relevant_activations,
        load_irrelevant_path=args.load_irrelevant_activations,
        hook_name=args.hook_name
    )
    
    # Get SAE activations
    sae_acts = get_sae_activations(
        model_acts, 
        args.model_name, 
        layers, 
        args.sae_repo_id
    )
    
    # Parse min_activations if provided
    min_activations = None
    if args.min_activations:
        try:
            min_activations = [float(x) for x in args.min_activations.split(',')]
            if len(min_activations) != 2:
                print(f"Warning: min_activations should have exactly 2 values, got {len(min_activations)}. Using default None.")
                min_activations = None
        except ValueError:
            print(f"Warning: Could not parse min_activations value '{args.min_activations}'. Using default None.")
            min_activations = None
    
    # Calculate feature scores for each position
    feats_per_layer = {}
    for layer in layers:
        feats_per_layer[layer] = {}
        for pos in pos_to_analyze:
            feature_scores = get_features(
                sae_acts[layer][pos],
                metric=args.scoring_method,
                min_activations=min_activations
            )
            feats_per_layer[layer][pos] = feature_scores
    
    # Format results
    save_dir = os.path.join(args.output_dir, f'layer_features')
    formatted_features = format_layer_features(
        feats_per_layer, 
        save_dir=save_dir
    )
    

    
    # Get top features for each position
    top_features = {}
    for pos in pos_to_analyze:
        top_features[pos] = get_top_k_features(
            {layer: formatted_features[layer][pos] for layer in layers}, 
            k=args.top_k_features
        )
    
    # Plot top features for each position
    plot_dir = os.path.join(args.output_dir, f'plots')
    plots = {}
    for pos in pos_to_analyze:
        plot = plot_top_features(top_features[pos], pos_to_analyze=pos, output_dir=plot_dir)
        plots[pos] = plot
    
    # Save top features
    os.makedirs(os.path.join(args.output_dir, f'top_features'), exist_ok=True)
    for pos in pos_to_analyze:
        with open(os.path.join(args.output_dir, f'top_features', f'top_features_P_{str(pos)}.json'), 'w') as f:
            json.dump(top_features[pos], f, indent=4, default=float)
    
    return {
        'top_features': top_features,
        'formatted_features': formatted_features,
        'plots': plots
    }

# Temporary test function to find the position of the 3rd to last newline token
def find_third_last_newline_token_pos(token_ids: torch.Tensor, tokenizer) -> int:
    """
    Finds the index (position) of the third to last newline token in a sequence of token IDs.

    Args:
        token_ids: A 1D tensor of token IDs.
        tokenizer: The tokenizer instance used to encode the text.

    Returns:
        The index of the third to last newline token, or -1 if fewer than
        three newline tokens are found.
    """
    # Get the token ID for newline
    newline_token_id_list = tokenizer.encode('\n\n', add_special_tokens=False)
    if len(newline_token_id_list) != 1:
        # Handle cases where newline might be tokenized differently (e.g., multiple tokens)
        # For simplicity, we'll use the first token if it's split, but ideally,
        # this case should be handled based on specific tokenizer behavior.
        print(f"Warning: Newline token is not a single token: {tokenizer.convert_ids_to_tokens(newline_token_id_list)}. Using the first token ID: {newline_token_id_list[0]}")
    newline_token_id = newline_token_id_list[0]

    # Find all indices where the token ID matches the newline token ID
    # Ensure token_ids is on the CPU for comparison and indexing if it isn't already
    newline_indices = torch.where(token_ids.cpu() == newline_token_id)[0]

    # Check if we have at least three newline tokens
    if len(newline_indices) < 1:
        return -1

    # Return the index of the third-to-last newline token
    return newline_indices[-1].item()

def main(args):
    
    # Initialize W&B if enabled
    if args.use_wandb:
        if not args.run_name:
            model_short_name = args.model_name.replace('/', '--').replace('.', '-')
            relevant_file_name = os.path.basename(args.relevant_data_file).split('.')[0]
            irrelevant_file_name = os.path.basename(args.irrelevant_data_file).split('.')[0]
            args.run_name = f"{model_short_name}_relevance_detection_{relevant_file_name}_{irrelevant_file_name}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args)
        )
    
    # Create output directories
    args.output_dir = os.path.join(args.output_dir, args.run_name, args.hook_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    if args.model_name in MODEL_ALIAS:
        model_name = MODEL_ALIAS[args.model_name]
    else:
        model_name = args.model_name
    
    model = load_model(model_name, 'cpu') # Should put to cpu as we barely use the model
    
    model_name = model.cfg.model_name if hasattr(model.cfg, 'model_name') else str(model.cfg)
    is_chat_model = "it" in model_name.lower() or "instruct" in model_name.lower() if model_name else False

    print(f"Chat model: {is_chat_model}")
    
    # Determine layers to analyze
    if args.layers:
        layers = [int(l) for l in args.layers.split(',')]
    else:
        # TODO: May remove the layer filter, use all layers
        # filter out the last 20% of layers
        layers = list(range(model.cfg.n_layers))
        layers = layers[:int(len(layers) * 0.8)]
    
    print(f"Analyzing layers: {layers}")
    
    # Load datasets
    print(f"Loading relevant dataset from {args.relevant_data_file}...")
    relevant_dataset = load_dataset('json', data_files=args.relevant_data_file)['train']
    print(f"Loading irrelevant dataset from {args.irrelevant_data_file}...")
    irrelevant_dataset = load_dataset('json', data_files=args.irrelevant_data_file)['train']
    
    # Apply max_examples limit to both datasets
    if args.max_examples:
        max_per_type = args.max_examples // 2  # Split examples evenly between relevant and irrelevant
        relevant_dataset = relevant_dataset.select(range(min(max_per_type, len(relevant_dataset))))
        irrelevant_dataset = irrelevant_dataset.select(range(min(max_per_type, len(irrelevant_dataset))))
    
    print(f"Processing {len(relevant_dataset)} relevant examples and {len(irrelevant_dataset)} irrelevant examples...")
    
    # Prepare examples
    model_name = model.cfg.model_name if hasattr(model.cfg, 'model_name') else str(model.cfg)
    relevant_dataset = relevant_dataset.map(get_model_inputs, fn_kwargs={'model_name': model_name, 'tokenizer': model.tokenizer, 'prompt_type': args.prompt_type, 'is_chat_model': is_chat_model, 'relevance_type': 'relevant'})
    irrelevant_dataset = irrelevant_dataset.map(get_model_inputs, fn_kwargs={'model_name': model_name, 'tokenizer': model.tokenizer, 'prompt_type': args.prompt_type, 'is_chat_model': is_chat_model, 'relevance_type': 'irrelevant'})
    
    # Combine datasets into a single format expected by the rest of the code
    combined_examples = []
    
    # Pair up relevant and irrelevant examples
    min_len = min(len(relevant_dataset), len(irrelevant_dataset))
    for i in range(min_len):
        relevant_item = relevant_dataset[i]
        irrelevant_item = irrelevant_dataset[i]
        
        # Create combined item in the format expected by downstream code
        combined_item = {
            'model_inputs': [relevant_item['model_input'], irrelevant_item['model_input']],
            'relevances': [relevant_item['relevance'], irrelevant_item['relevance']],
            'question': relevant_item['question'],  # Assuming questions are the same
            'answers': relevant_item['answers']
        }
        combined_examples.append(combined_item)
    
    dataset = combined_examples
    
    # Analyze relevance features
    results = analyze_relevance_features(model, dataset, layers, args)

    # Log results to wandb
    wandb.log({
        'top_features': results['top_features'],
        'plots': results['plots']
    })
        
    # Finish wandb run
    wandb.finish()
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args) 
