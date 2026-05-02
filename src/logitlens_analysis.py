"""
This is a combined script for logitlens analysis for both no_passage and relevant_passage variants.
It is used to analyze the logitlens of the model for both variants.
"""

import torch
from transformer_lens import HookedTransformer, ActivationCache
import einops
from jaxtyping import Float
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import argparse
import json
import wandb
from datasets import load_dataset
from tqdm import tqdm
from utils import format_prompt

def parse_args():
    
    parser = argparse.ArgumentParser(description="Analyze models using LogitLens technique")
    
    # Required arguments
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name to analyze")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the JSONL data file")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="output/logitlens_analysis",
                        help="Directory to save the output figures")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix for output filenames")
    parser.add_argument("--prompt_template", type=str, default="no_passage_no_refuse",
                        help="Type of prompt template to use")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process")
    
    # Wandb related arguments
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="logitlens",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name, defaults to model_name-prefix if not provided")
    
    return parser.parse_args()

def load_model(model_name: str):
    """Load model using HookedTransformer."""
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Loading model {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype='bfloat16')
    model.eval()
    return model

def get_model_input(item, model: HookedTransformer, prompt_template: str):
    """
    Get the full text of the question and generations from the data.
    Split generations into correct and incorrect ones.
    
    Args:
        item: Dataset item containing question and multiple generations
        model: HookedTransformer model instance
        prompt_template: Type of prompt template to use
    
    Returns:
        dict: Contains model inputs for both correct and incorrect generations
    """
    question = item["question"]
    prompt = format_prompt(question, None, prompt_template)
    
    # Initialize lists to store correct and incorrect generations
    model_input_correct_generations = []
    model_input_incorrect_generations = []
    correct_generations = []
    incorrect_generations = []
    
    # Process each generation
    for passage in item['passages']:
        generation = passage['response']
        is_correct = passage['evaluation']['acc']
        
        # Create the full conversation with this generation
        model_input_generation = model.tokenizer.apply_chat_template(
            [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': generation},
            ],
            tokenize=False,
        )
        
        # Add to appropriate list based on correctness
        if is_correct:
            model_input_correct_generations.append(model_input_generation)
            correct_generations.append(generation)
        else:
            model_input_incorrect_generations.append(model_input_generation)
            incorrect_generations.append(generation)
    
    # Create the base model input (just the question)
    model_input = model.tokenizer.apply_chat_template(
        [
            {'role': 'user', 'content': prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return {
        'model_input': model_input,
        'model_input_correct_generations': model_input_correct_generations,
        'model_input_incorrect_generations': model_input_incorrect_generations,
        'num_correct': len(model_input_correct_generations),
        'num_incorrect': len(model_input_incorrect_generations),
        'correct_generations': correct_generations,
        'incorrect_generations': incorrect_generations
    }

def find_exact_answer(answers, pred):
    """Find if any of the ground truth answers appear in the prediction."""
    for answer in answers:
        if answer in pred:
            return answer
    return None

def map_char_to_token_positions(text, answer, tokenizer):
    """
    Maps character-level positions to token-level positions.
    
    Args:
        text: Source text
        answer: Answer to locate
        tokenizer: The model's tokenizer
    
    Returns:
        List of token indices that cover the answer
    """
    # Find character positions of all answer occurrences
    answer_positions = []
    start_pos = 0
    
    while True:
        start_idx = text.find(answer, start_pos)
        if start_idx == -1:
            break
        end_idx = start_idx + len(answer)
        answer_positions.append((start_idx, end_idx))
        start_pos = start_idx + 1
    
    if not answer_positions:
        return []
    
    # Get token-to-char mapping from tokenizer
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_spans = encoding.offset_mapping
    
    # Find tokens that overlap with answer spans
    token_positions = []
    for start_char, end_char in answer_positions:
        tokens_for_this_span = []
        for i, (token_start, token_end) in enumerate(token_spans):
            if token_end > start_char and token_start < end_char:
                tokens_for_this_span.append(i)
        
        if tokens_for_this_span:
            token_positions.append((tokens_for_this_span[0], tokens_for_this_span[-1] + 1))
    
    return token_positions

def find_generation_pos(model_input_tokens, generation_tokens):
    """
    Find the position of the generation in the model input tokens

    Args:
    model_input_tokens: list of tokens of the model input
    generation_tokens: list of tokens of the generation

    Returns:
    pos: a list of the position of the generation in the model input tokens
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

def residual_stack_to_logit(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1).to(dtype=torch.bfloat16)
    return (
        einops.einsum(scaled_residual_stack, logit_directions, "... batch d_model, batch d_model -> ...")
        / batch_size
    )

def accumulated_logit_lens(cache: ActivationCache, logit_direction: Float[Tensor, "batch d_model"], layer: int = -1, pos_slice: slice | int = -1):
    accumulated_residual, labels = cache.accumulated_resid(layer=layer, incl_mid=True, pos_slice=pos_slice, return_labels=True)
    return residual_stack_to_logit(accumulated_residual, cache, logit_direction), labels

def decomposed_logit_lens(cache: ActivationCache, logit_direction: Float[Tensor, "batch d_model"], layer: int = -1, pos_slice: slice | int = -1):
    per_layer_residual, labels = cache.decompose_resid(layer=layer, pos_slice=pos_slice, return_labels=True)
    return residual_stack_to_logit(per_layer_residual, cache, logit_direction), labels

def head_logit_lens(model: HookedTransformer, cache: ActivationCache, logit_direction: Float[Tensor, "batch d_model"], layer: int = -1, pos_slice: slice | int = -1):
    per_head_residual, labels = cache.stack_head_results(layer=layer, pos_slice=pos_slice, return_labels=True)
    per_head_residual = einops.rearrange(per_head_residual, "(layer head) ... -> layer head ...", layer=model.cfg.n_layers)
    return residual_stack_to_logit(per_head_residual, cache, logit_direction), labels

def process_example_for_exact_answers(example, model):
    """Process an example to find exact answers in correct generations."""
    # Process all correct generations
    correct_answers = []
    correct_positions = []
    correct_generation_positions = []
    correct_exact_positions = []

    # Check if there are any correct generations
    if example['num_correct'] > 0:
        for i in range(example['num_correct']):
            # Find the exact answer in this generation
            answer = find_exact_answer(example['ground_truth'], example['correct_generations'][i])
            if answer:
                # Map the answer to token positions
                pos = map_char_to_token_positions(example['model_input_correct_generations'][i], answer, model.tokenizer)
                if pos:
                    # Find the generation position in the model input
                    generation_pos = find_generation_pos(
                        model.to_tokens(example['model_input_correct_generations'][i], prepend_bos=False), 
                        model.to_tokens(example['correct_generations'][i].strip(), prepend_bos=False)
                    )
                    if generation_pos:
                        # Extract the exact answer positions within the generation
                        exact_positions = extract_exact_answer_token_generation(pos, generation_pos)
                        if exact_positions:
                            correct_answers.append(answer)
                            correct_positions.append(pos)
                            correct_generation_positions.append(generation_pos)
                            correct_exact_positions.append(exact_positions)

    return {
        'correct_answers': correct_answers,
        'correct_positions': correct_positions,
        'correct_generation_positions': correct_generation_positions,
        'correct_exact_positions': correct_exact_positions
    }

def process_example(example, model):
    """Process a single example and return analysis results for each correct generation."""
    results = []
    
    # Process each correct generation
    for i in range(min(len(example['correct_answers']), len(example['correct_exact_positions']))):
        answer = example['correct_answers'][i]
        pos = example['correct_positions'][i]
        generation_pos = example['correct_generation_positions'][i]
        exact_positions_in_generation = example['correct_exact_positions'][i]
        
        # Run model with cache
        with torch.no_grad():
            logits, cache = model.run_with_cache(model.to_tokens(example['model_input'], prepend_bos=False))
        
        # Get first exact answer token and its logit direction
        model_input_generation_tokens = model.to_tokens(example['model_input_correct_generations'][i], prepend_bos=False)
        exact_answer_tokens = model_input_generation_tokens[0, exact_positions_in_generation[0][0]:exact_positions_in_generation[0][1]]
        first_exact_answer_token = exact_answer_tokens[0]
        first_exact_answer_logit_direction = einops.rearrange(
            model.tokens_to_residual_directions(first_exact_answer_token),
            "d_model -> 1 d_model"
        )
        
        # Get accumulated logits
        accumulated_logits, labels = accumulated_logit_lens(cache, first_exact_answer_logit_direction)
        
        # Get per-layer logits
        per_layer_logits, _ = decomposed_logit_lens(cache, first_exact_answer_logit_direction)
        
        # Get per-head logits
        per_head_logits, _ = head_logit_lens(model, cache, first_exact_answer_logit_direction)
        
        results.append({
            'generation_index': i,
            'answer': answer,
            'accumulated_logits': accumulated_logits.cpu().detach(),
            'per_layer_logits': per_layer_logits.cpu().detach(),
            'per_head_logits': per_head_logits.cpu().detach(),
            'labels': labels
        })
    
    return results

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize W&B if enabled
    if args.use_wandb:
        # Create a default run name if not provided
        if not args.wandb_run_name:
            # Format similar to rag_evaluation.py
            model_short_name = args.model_name.replace('/', '--')
            data_file_name = os.path.basename(args.data_file).split('.')[0]
            script_name = os.path.basename(__file__)[:-len('.py')]
            args.wandb_run_name = f"{model_short_name}__{args.prompt_template}__{data_file_name}_{script_name}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name,
                "data_file": os.path.basename(args.data_file),
                "prompt_template": args.prompt_template,
                "max_examples": args.max_examples
            }
        )
        print(f"Initialized W&B logging for project: {args.wandb_project}, run: {args.wandb_run_name}")
    
    # Load model
    model = load_model(args.model_name)
    
    # Load dataset
    print(f"Loading dataset from {args.data_file}...")
    dataset = load_dataset('json', data_files=args.data_file)
    dataset = dataset['train']
    
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    
    print(f"Processing {len(dataset)} examples...")
    
    # Prepare data
    dataset = dataset.map(lambda x: get_model_input(x, model, args.prompt_template), batched=False)
    dataset = dataset.map(lambda x: process_example_for_exact_answers(x, model), batched=False)
    # for example in tqdm(dataset):
        
    #     # Get model input
    #     example_data = get_model_input(example, model, args.prompt_template)
        
    #     # Find exact answers
    #     # example_data.update(process_example_for_exact_answers(example_data, model))
    #     example_data.map
        
    #     # Store ground truth
    #     example_data['ground_truth'] = example['ground_truth']
        
    #     processed_dataset.append(example_data)
    
    # Process examples
    all_results = []
    for example in tqdm(dataset):
        
        results = process_example(example, model)
        if results:
            all_results.extend(results)
    
    print(f"Collected {len(all_results)} results for analysis")
    
    if not all_results:
        print("No results to analyze. Exiting.")
        if args.use_wandb:
            wandb.finish()
        return
    
    # Calculate statistics across all generations
    accumulated_logits_stack = torch.stack([r['accumulated_logits'] for r in all_results])
    per_layer_logits_stack = torch.stack([r['per_layer_logits'] for r in all_results])
    per_head_logits_stack = torch.stack([r['per_head_logits'] for r in all_results])

    # Convert to numpy arrays for easier plotting
    accumulated_mean = accumulated_logits_stack.mean(dim=0).float().numpy()
    accumulated_min = accumulated_logits_stack.min(dim=0)[0].float().numpy()
    accumulated_max = accumulated_logits_stack.max(dim=0)[0].float().numpy()
    accumulated_std = accumulated_logits_stack.std(dim=0).float().numpy()

    per_layer_mean = per_layer_logits_stack.mean(dim=0).float().numpy()
    per_layer_min = per_layer_logits_stack.min(dim=0)[0].float().numpy()
    per_layer_max = per_layer_logits_stack.max(dim=0)[0].float().numpy()
    per_layer_std = per_layer_logits_stack.std(dim=0).float().numpy()

    per_head_mean = per_head_logits_stack.mean(dim=0).float().numpy()
    
    # Log statistics to W&B if enabled
    if args.use_wandb:
        # Log summary statistics
        wandb.log({
            "num_examples": len(dataset),
            "num_results": len(all_results),
            "accumulated_logits_max": float(accumulated_mean.max()),
            "per_layer_logits_max": float(per_layer_mean.max()),
            "per_head_logits_max": float(per_head_mean.max())
        })
        
        # Create and log tables for layer-wise data
        layer_table = wandb.Table(
            columns=["Layer", "Accumulated Mean", "Accumulated Min", "Accumulated Max", "Accumulated Std",
                     "Per Layer Mean", "Per Layer Min", "Per Layer Max", "Per Layer Std"]
        )
        
        for i, label in enumerate(all_results[0]['labels']):
            layer_table.add_data(
                label, 
                float(accumulated_mean[i]), float(accumulated_min[i]), float(accumulated_max[i]), float(accumulated_std[i]),
                float(per_layer_mean[i]), float(per_layer_min[i]), float(per_layer_max[i]), float(per_layer_std[i])
            )
        
        wandb.log({"layer_statistics": layer_table})

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_context("paper", font_scale=1.2)

    # Generate filename prefix
    prefix = args.prefix + "_" if args.prefix else ""
    
    # Plot accumulated logits
    plt.figure(figsize=(10, 6))
    x = np.arange(len(all_results[0]['labels']))
    plt.plot(x, accumulated_mean, marker='o', color='blue', label='Mean', zorder=2)
    plt.fill_between(x, accumulated_min, accumulated_max, alpha=0.2, color='blue', zorder=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Layer')
    plt.ylabel('Score')
    plt.title('Accumulated Logits Across Dataset')
    plt.legend()
    plt.tight_layout()
    fig_path = f"{args.output_dir}/{prefix}accumulated_logits.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    if args.use_wandb:
        wandb.log({"accumulated_logits_plot": wandb.Image(fig_path)})
    plt.close()

    # Plot per-layer logits
    plt.figure(figsize=(10, 6))
    plt.plot(x, per_layer_mean, marker='o', color='blue', label='Mean', zorder=2)
    plt.fill_between(x, per_layer_min, per_layer_max, alpha=0.2, color='blue', zorder=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Layer')
    plt.ylabel('Score')
    plt.title('Per-Layer Logits Across Dataset')
    plt.legend()
    plt.tight_layout()
    fig_path = f"{args.output_dir}/{prefix}per_layer_logits.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    if args.use_wandb:
        wandb.log({"per_layer_logits_plot": wandb.Image(fig_path)})
    plt.close()

    # Plot per-head logits heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(per_head_mean, 
                cmap='RdBu',
                center=0,
                cbar_kws={'label': 'Mean Logit Value'},
                xticklabels='auto',
                yticklabels='auto')
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.title('Mean Logit From Each Head Across Dataset')
    plt.tight_layout()
    fig_path = f"{args.output_dir}/{prefix}per_head_logits_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    if args.use_wandb:
        wandb.log({"per_head_logits_heatmap": wandb.Image(fig_path)})
    plt.close()
    
    # Compare multiple datasets/models side by side if this is wandb run
    if args.use_wandb:
        # Create a line plot for comparing results across models/datasets
        data = {
            "Layer": list(range(len(all_results[0]['labels']))),
            "Accumulated Score": accumulated_mean.tolist(),
            "Per Layer Score": per_layer_mean.tolist(),
            "Model/Dataset": [args.wandb_run_name] * len(all_results[0]['labels'])
        }
        wandb.log({"layer_scores": wandb.Table(dataframe=pd.DataFrame(data))})
        
        # Finish the wandb run
        wandb.finish()
    
    print(f"Analysis complete. Figures saved to {args.output_dir} directory.")

if __name__ == "__main__":
    main() 