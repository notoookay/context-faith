"""
Model Activation Extractor

This module focuses on extracting activations from specified layers and positions
of a transformer model for given input data. It adapts relevant functions from
'answerable_recognition.py' but excludes SAE-specific logic.
"""

import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
import einops
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any
from datasets import load_dataset
import random
import wandb

from transformer_lens import HookedTransformer, ActivationCache

from utils import load_jsonlines, save_file_jsonl, format_prompt, MODEL_ALIAS, LLAMA_3_1_INSTRUCT_PROMPT_FORMAT, QWEN_2_5_INSTRUCT_PROMPT_FORMAT

def parse_args():
    """Parse command line arguments for activation extraction."""
    parser = argparse.ArgumentParser(description="Extract activations from language models")
    
    # Required arguments
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name or alias to load (e.g., 'google/gemma-2-2b-it')")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the JSONL data file with questions and prompts")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="output/model_activations",
                        help="Directory to save the output activations")
    parser.add_argument("--prompt_type", type=str, default="with_passage",
                        help="Type of prompt template to use for formatting input")
    parser.add_argument("--dataset_with_modified_passages", action="store_true", help="The loaded dataset is with modified passages")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layers to extract activations from (e.g., '10,15'). If None, all layers are used.")
    parser.add_argument("--pos_to_analyze", type=str, default="-2",
                        help="Comma-separated list of positions to analyze (e.g., '-2,-1'). Supports negative indexing.")
    parser.add_argument("--save_activations", type=str, default=None,
                        help="Path to save the activations. If None, defaults to a file in output_dir.")
    parser.add_argument("--load_activations", type=str, default=None,
                        help="Path to load pre-computed activations from.")
    parser.add_argument("--test_third_last_newline", action="store_true",
                        help="Use the position of the third-to-last newline token as the analysis position.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run name for organizing output files.")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="get-model-activation",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name")

    return parser.parse_args()

def load_model(model_name: str):
    """Load model using HookedTransformer."""
    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Loading model {model_name} on {device}...")
    # Resolve alias if necessary
    resolved_model_name = MODEL_ALIAS.get(model_name, model_name)
    model = HookedTransformer.from_pretrained(resolved_model_name, device=device, dtype=torch.bfloat16)
    model.eval()
    return model

def prepare_model_inputs(item, tokenizer, prompt_type: str, dataset_with_modified_passages: bool, is_chat_model: bool=False, model_name: str=None):
    """
    Prepare model inputs from a dataset item.
    
    Args:
        item: Dataset item containing question, passages etc.
        model: HookedTransformer model instance
        prompt_type: Type of prompt template to use
        dataset_with_modified_passages: Whether the dataset is with modified passages
    
    Returns:
        dict: Contains model inputs strings and other relevant info like 'is_known'.
    """
    question = item["question"]
    passages = item["ctxs"] # assume there is only one passage

    model_input = None

    if prompt_type == "no_passage":
        if is_chat_model:
            native_prompt = format_prompt(question, prompt_type=prompt_type, format_passages=(not dataset_with_modified_passages))
            if "llama" in model_name:
                model_input = LLAMA_3_1_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
            elif "qwen" in model_name:
                model_input = QWEN_2_5_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
            else:
                model_input = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': native_prompt}],
                    tokenize=False,
                    add_generation_prompt=True, # Adds the prompt for the assistant's turn
                )
        else:
            model_input = format_prompt(question, prompt_type=prompt_type, format_passages=(not dataset_with_modified_passages), base_model_prompt=True)
        
    else:
        # Process each passage/generation
        for passage in passages:
            
            if is_chat_model:
                # Create prompt for chat/instruct models
                native_prompt = format_prompt(question, passage_list=[passage], prompt_type=prompt_type, format_passages=(not dataset_with_modified_passages))
                if "llama" in model_name:
                    model_input = LLAMA_3_1_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
                elif "qwen" in model_name:
                    model_input = QWEN_2_5_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
                else:
                    model_input = tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': native_prompt}],
                        tokenize=False,
                        add_generation_prompt=True, # Adds the prompt for the assistant's turn
                    )
            else:
                # Create prompt for base models
                model_input = format_prompt(question, passage_list=[passage], prompt_type=prompt_type, format_passages=(not dataset_with_modified_passages), base_model_prompt=True)
            
    return {
        'model_input': model_input,
        'question': question,
        # 'answers': item['answers']
    }

def save_model_activations(activations: Dict, save_path: str):
    """
    Save model activations to disk.
    
    Args:
        activations: Dictionary of activations
        save_path: Path to save the activations
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(activations, save_path)
    print(f"Saved activations to {save_path}")

def load_model_activations(load_path: str) -> Dict:
    """
    Load model activations from disk.
    
    Args:
        load_path: Path to load the activations from
        
    Returns:
        Dictionary of activations
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Activation file not found at {load_path}")
    
    activations = torch.load(load_path)
    print(f"Loaded activations from {load_path}")
    return activations


def get_model_activations(
    model: HookedTransformer, 
    dataset: List[Dict], 
    layers: List[int], 
    pos_to_analyze: List[int],
    use_third_last_newline: bool = False,
    save_path: Optional[str] = None, 
    load_path: Optional[str] = None
) -> Dict:
    """
    Get model activations for a list of examples.
    
    Args:
        model: HookedTransformer model
        dataset: List of processed examples (dictionaries with 'model_inputs')
        layers: List of layers to collect activations from
        pos_to_analyze: List of positions (indices) to analyze. Overridden if use_third_last_newline is True.
        use_third_last_newline: If True, dynamically find the third-last newline position.
        save_path: Optional path to save activations
        load_path: Optional path to load activations from
    
    Returns:
        Dictionary of activations by layer and position: {layer: {pos: Tensor}}
    """
    
    # Try to load activations if path provided
    if load_path is not None and os.path.exists(load_path):
        print(f"Attempting to load activations from {load_path}")
        try:
            loaded_data = load_model_activations(load_path)
            # Basic check if structure seems compatible
            if isinstance(loaded_data, dict) and layers[0] in loaded_data:
                 # Check if the loaded positions match requested ones (if not using dynamic newline finding)
                 first_layer_data = loaded_data[layers[0]]
                 loaded_positions = list(first_layer_data.keys())
                 
                 # Check if all requested positions are present, unless using dynamic newline
                 if use_third_last_newline or all(pos in loaded_positions for pos in pos_to_analyze):
                     print("Successfully loaded and verified activations.")
                     return loaded_data
                 else:
                     print(f"Warning: Loaded activation positions {loaded_positions} don't match requested {pos_to_analyze}. Recomputing.")
            else:
                print("Warning: Loaded activation structure mismatch. Recomputing.")
        except Exception as e:
            print(f"Could not load activations from {load_path}: {e}. Recomputing.")
    
    activations = defaultdict(lambda: defaultdict(list))
    
    hook_name = 'resid_post'
    
    print(f"Extracting activations for layers {layers} at positions {pos_to_analyze} {'(or third-last newline)' if use_third_last_newline else ''}")

    for example in tqdm(dataset, desc="Getting model activations"):
        model_input = example['model_input']
        
        # Tokenize the input
        model_input_tokens = model.to_tokens(model_input, prepend_bos=False)

        # Run model and cache activations for the specified layers
        with torch.no_grad():
            # Only use hook_resid_post for each layer
            
            _, cache = model.run_with_cache(
                model_input_tokens,
            )

        # Extract activations for the target positions
        for layer in layers:
            # Only use hook_resid_post
            all_acts = cache[hook_name, layer][0] # Remove batch dim
            
            # Iterate through the *valid original* positions
            for pos in pos_to_analyze:
                
                act = all_acts[pos].cpu() # Ensure CPU tensor
                # Create a nested key structure: layer -> pos
                if layer not in activations:
                    activations[layer] = {}
                if pos not in activations[layer]:
                    activations[layer][pos] = []
                
                # Append the activation
                activations[layer][pos].append(act)

    # Final processing: Stack tensors for each category
    final_activations = {}

    for layer in layers:
        final_activations[layer] = {}
        for pos in activations[layer]: # Iterate through positions actually found
            final_activations[layer][pos] = torch.stack(activations[layer][pos])

    # Save activations if path provided
    if save_path is not None:
        save_model_activations(final_activations, save_path)
    
    return final_activations


def main(args):

    # Initialize W&B if enabled
    if args.use_wandb:
        # Define default run name if not provided
        if not args.run_name:
            model_short_name = args.model_name.replace('/', '--').replace('.', '-')
            data_file_name = os.path.basename(args.data_file).split('.')[0]
            pos_str = args.pos_to_analyze.replace(',', '_')
            layers_str = args.layers.replace(',', '_') if args.layers else 'all'
            # modified_passages_str = 'modified_passages' if args.dataset_with_modified_passages else 'original_passages'
            args.run_name = f"{model_short_name}_{data_file_name}_{args.prompt_type}_layers_{layers_str}_pos_{pos_str}_activation"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args) # Log all arguments
        )
        print(f"Wandb logging enabled for run: {args.run_name}")

    # Create output directory
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Define save path if not given
    if args.save_activations is None:
        filename = "activations.pt"
        args.save_activations = os.path.join(args.output_dir, filename)
        print(f"Activations will be saved to: {args.save_activations}")

    # Load model
    model = load_model(args.model_name)
    model_name = model.cfg.model_name if hasattr(model.cfg, 'model_name') else str(model.cfg)
    is_chat_model = "it" in model_name.lower() or "instruct" in model_name.lower() if model_name else False

    print(f"Chat model: {is_chat_model}")

    # Determine layers to analyze
    if args.layers:
        try:
            layers = [int(l.strip()) for l in args.layers.split(',')]
        except ValueError:
            print(f"Error: Invalid layer specification '{args.layers}'. Please use comma-separated integers.")
            return
    else:
        layers = list(range(model.cfg.n_layers))  # Default to all layers
    print(f"Target layers: {layers}")

    # Determine positions to analyze
    try:
        pos_to_analyze = [int(p.strip()) for p in args.pos_to_analyze.split(',')]
    except ValueError:
        print(f"Error: Invalid position specification '{args.pos_to_analyze}'. Please use comma-separated integers.")
        return
    if args.test_third_last_newline:
        print("Position analysis method: Third-to-last newline token.")
        # pos_to_analyze will be ignored initially in get_model_activations
    else:
        print(f"Target positions: {pos_to_analyze}")

    # Load dataset
    print(f"Loading dataset from {args.data_file}...")
    dataset = load_dataset('json', data_files=args.data_file)['train']

    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print(f"Preparing {len(dataset)} examples...")
    tokenizer = model.tokenizer
    # Use map for potentially large datasets
    processed_dataset = dataset.map(
        prepare_model_inputs,
        fn_kwargs={
            'tokenizer': tokenizer,
            'prompt_type': args.prompt_type,
            'dataset_with_modified_passages': args.dataset_with_modified_passages,
            'is_chat_model': is_chat_model,
            'model_name': model.cfg.model_name.lower(),
        },
        # num_proc=4 # Optional: speed up processing
    )

    # Filter out examples where input preparation failed
    # processed_dataset = processed_dataset.filter(lambda example: example['model_inputs'] is not None and len(example['model_inputs']) > 0)
    # print(f"Processing {len(processed_dataset)} valid examples...")

    # Get model activations
    activations = get_model_activations(
        model=model,
        dataset=list(processed_dataset), # Convert map dataset to list for iteration
        layers=layers,
        pos_to_analyze=pos_to_analyze,
        use_third_last_newline=args.test_third_last_newline,
        save_path=args.save_activations,
        load_path=args.load_activations
    )

    print(f"Activation extraction complete.")
    if args.save_activations and not args.load_activations:
        print(f"Activations saved to {args.save_activations}")
    elif args.load_activations:
        print(f"Activations loaded from {args.load_activations}")

    # Example: Print shape of activations for the first layer/pos found
    if activations:
        first_layer = list(activations.keys())[0]
        if activations[first_layer]:
            first_pos = list(activations[first_layer].keys())[0]
            print(f"\nExample activation shapes for Layer {first_layer}, Position {first_pos}:")
            print(f"Shape: {activations[first_layer][first_pos].shape}")
        else:
            print("No activations found for the first layer.")
    else:
        print("No activations were extracted or loaded.")

    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args) 
