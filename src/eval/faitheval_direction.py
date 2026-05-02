import argparse
import os
import json
import torch
from functools import partial
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from torch.utils.data import DataLoader
from transformers import set_seed
from datasets import load_dataset
import wandb
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from torch import Tensor

import sys
sys.path.append("..")
from exp_hallucination.src.utils import load_jsonlines, save_file_jsonl, format_prompt, MODEL_ALIAS
from exp_hallucination.src.eval.utils import normalize_answer, evaluate_faithfulness
from exp_hallucination.src.diff_in_mean import apply_hook_to_model
from exp_hallucination.src.eval.faitheval_generate import FaithEvalDataset, collate_fn

# def direction_hook(
#     activations: Float[Tensor, "batch pos d_in"],
#     hook: HookPoint,
#     direction: torch.Tensor,
#     coefficient: float,
#     positions: Optional[List[int]] = None,
#     ablate: bool = False,
# ) -> Tensor:
#     """
#     Applies a direction to model activations at specified positions.
    
#     Args:
#         activations: The input activations tensor
#         hook: The hook point
#         direction: The direction tensor to apply
#         coefficient: Scaling coefficient for the direction
#         positions: List of positions to apply the direction to. If None, applies to all positions.
#         ablate: Whether to ablate the direction (project out) instead of adding it
#     """
#     # Skip during generation (when processing one token at a time)
#     if activations.shape[1] == 1:
#         return activations

#     if positions is None:
#         positions = list(range(activations.shape[1]))
    
#     # Apply direction at specified positions
#     if isinstance(positions[0], list):
#         # Handle per-batch positions
#         for batch_idx, pos in enumerate(positions):
#             if pos < activations.shape[1]:  # Only apply if position exists in current sequence
#                 if ablate:
#                     # Project out the direction (ablation)
#                     direction_norm = direction / torch.norm(direction)
#                     proj = torch.sum(activations[batch_idx, pos] * direction_norm) * direction_norm
#                     activations[batch_idx, pos] -= proj * coefficient
#                 else:
#                     # Add the direction
#                     activations[batch_idx, pos] += direction * coefficient
#     else:
#         # Handle same positions for all batches
#         for pos in positions:
#             if pos < activations.shape[1]:  # Only apply if position exists in current sequence
#                 if ablate:
#                     # Project out the direction (ablation) for all items in batch
#                     direction_norm = direction / torch.norm(direction)
#                     for batch_idx in range(activations.shape[0]):
#                         proj = torch.sum(activations[batch_idx, pos] * direction_norm) * direction_norm
#                         activations[batch_idx, pos] -= proj * coefficient
#                 else:
#                     # Add the direction to all items in batch
#                     activations[:, pos] += direction * coefficient
    
#     return activations


def main(args):
    set_seed(args.seed)
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set model name
    if args.model_name in MODEL_ALIAS:
        model_name = MODEL_ALIAS[args.model_name]
    else:
        model_name = args.model_name
        print(f"Warning: Model name '{args.model_name}' is not in alias list, using as is.")

    # Task-specific prompt based on task type
    if args.task_type == "unanswerable":
        task_specific_prompt = "If there is no information available from the context, the answer should be 'unknown'."
        dataset_name = "Salesforce/FaithEval-unanswerable-v1.0"
    elif args.task_type == "inconsistent":
        task_specific_prompt = "If there is conflict information or multiple answers from the context, the answer should be 'conflict'."
        dataset_name = "Salesforce/FaithEval-inconsistent-v1.0"
    elif args.task_type == "counterfactual":
        # task_specific_prompt = "Given four answer candidates, A, B, C and D, choose the best answer choice based on your knowledge or the context. Your answer should be A, B, C or D."
        task_specific_prompt = ""
        dataset_name = "Salesforce/FaithEval-counterfactual-v1.0"
    
    # Process layer argument to convert string to list of integers
    if args.layer is not None:
        args.layer = [int(layer) for layer in args.layer.split(',')]
    
    # Initialize wandb if requested
    run_suffix = "ablate" if args.ablate else "add"
    layer_str = "_".join([str(layer) for layer in args.layer])
    args.run_name = f"{args.model_name.replace('/', '--')}__{args.task_type}__{os.path.basename(args.direction_file).split('.')[0]}_coef{args.coefficient}_{run_suffix}_layer{layer_str}_{args.seed}"
    if args.use_wandb:
        wandb.init(
            project="faitheval-direction", 
            name=args.run_name,
            config=vars(args)
        )
    
    # Load data from Huggingface
    print(f"Loading {args.task_type} data from Huggingface...")
    data = load_dataset(dataset_name, split="test")
    print(f"Loaded {len(data)} items.")

    # Limit the number of items if specified
    if args.max_items > 0:
        data = data.select(range(min(args.max_items, len(data))))
        print(f"Limited to {len(data)} items.")

    print(f"Greedy decoding: {not args.do_sample}")
    print(f"Number of generations per question: {args.n_samples}")
    print(f"Loading model {model_name} with TransformerLens...")
    
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        n_devices=args.n_gpu,
        dtype='bfloat16' if not args.fp16 else 'float16',
        default_padding_side='left',
    )
    print("Model loaded successfully")
    
    # Set model to evaluation mode
    model.eval()
    
    # Load direction
    print(f"Loading direction from {args.direction_file}...")
    direction = torch.load(args.direction_file, map_location=device)
    
    print(f"Direction shape: {direction.shape}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = FaithEvalDataset(
        data, 
        model.tokenizer, 
        args.n_samples, 
        task_specific_prompt=task_specific_prompt,
        dataset_name=dataset_name,
        model_name=args.model_name,
        task_type=args.task_type
    )
    print(f"Created dataset with {len(dataset)} items")

    # Create DataLoader with custom collate function
    custom_collate = lambda batch: collate_fn(batch, model.tokenizer, args.model_name, task_type=args.task_type)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=args.num_workers if args.num_workers > 0 else 0,
        pin_memory=True if torch.cuda.is_available() and args.num_workers > 0 else False
    )
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # Process data in batches
    results_map = {}  # Map to store results indexed by item_id
    
    # Process in batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating steered responses")):
        # Move input tensors to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata']
        
        # Parse positions for steering
        if args.pos_to_apply:
            positions = [int(pos) for pos in args.pos_to_apply.split(',')]
        else:
            # Default to last token of each prompt
            positions = [meta['prompt_length'] - 1 for meta in metadata]
            
        # Log steering positions for the first batch
        if batch_idx == 0:
            token_strs = model.to_str_tokens(input_ids[0])
            print(f"Applying direction at positions: {positions}")
            print(f"Token at position: {[token_strs[pos] for pos in positions]}")
            print(f"Using coefficient: {args.coefficient}, ablate: {args.ablate}")
        
        # Generate responses with transformer_lens
        with torch.no_grad():
            # Create direction hook
            hooks_dict = apply_hook_to_model(
                model=model,
                direction=direction,
                layer=args.layer,
                coeff=args.coefficient,
                ablate=args.ablate,
                position=positions,
                hook_name=args.hook_name
            )
            # Generate steered response
            with model.hooks(fwd_hooks=[(name, hook_fn) for name, hook_fn in hooks_dict.items()]):
                outputs_steered = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.do_sample else 0.0,
                    do_sample=args.do_sample,
                    verbose=False
                )
        
        # Process results
        for i, (meta, output_steered) in enumerate(zip(metadata, outputs_steered)):
            # Get the length of the input
            input_length = meta['prompt_length']
            
            # Extract only the generated tokens (excluding input)
            steered_tokens = output_steered[input_length:]
            
            # Find EOS token (if any) to truncate the responses
            eos_token_id = model.tokenizer.eos_token_id
            if eos_token_id is not None:
                # Find the first occurrence of EOS token in steered response
                eos_positions = (steered_tokens == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    # Truncate at the first EOS token
                    steered_tokens = steered_tokens[:eos_positions[0]]
            
            # Decode the responses
            steered_response = model.tokenizer.decode(steered_tokens, skip_special_tokens=True)
            
            # Store the data
            item_id = meta['item_id']
            if item_id not in results_map:
                results_map[item_id] = {
                    'id': item_id,
                    'question': meta['question'],
                    'context': meta['context'],
                    'answers': meta['answers'],
                    'responses': []
                }
            if args.task_type == "counterfactual":
                results_map[item_id]['answer_key'] = meta['answer_key']
            
            results_map[item_id]['responses'].append({
                'response': steered_response,
                'gen_idx': len(results_map[item_id]['responses'])
            })
    
    # Convert results map to list
    results = list(results_map.values())
    
    # Prepare for evaluation
    all_predictions = []
    for item in results:
        for response_data in item['responses']:
            all_predictions.append({
                'id': item['id'],
                'question': item['question'],
                'context': item['context'],
                'answers': item['answers'],
                'response': response_data['response']
            })
            if args.task_type == "counterfactual":
                all_predictions[-1]['answer_key'] = item['answer_key']
    
    # Evaluate faithfulness
    accuracy, correct_count, total_count = evaluate_faithfulness(
        all_predictions, 
        args.task_type,
        args.strict_match
    )
    
    print(f"\n=== Evaluation Results for {args.task_type} with Direction {'Ablation' if args.ablate else 'Addition'} ===")
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct_count}/{total_count})")
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count
        })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, args.task_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, f"{args.run_name}.jsonl")
    print(f"\nSaving generated responses to {output_file}...")
    save_file_jsonl(results, output_file)
    
    # Save evaluation metadata
    eval_metadata = {
        'model_name': args.model_name,
        'task_type': args.task_type,
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'do_sample': args.do_sample,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature if args.do_sample else None,
        'direction_file': args.direction_file,
        'coefficient': args.coefficient,
        'ablate': args.ablate,
        'layer': args.layer,
        'hook_name': args.hook_name,
        'pos_to_apply': args.pos_to_apply,
        'strict_match': args.strict_match,
        'seed': args.seed
    }
    
    metadata_file = os.path.splitext(output_file)[0] + '_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(eval_metadata, f, indent=2)
    
    print(f"Saved evaluation metadata to {metadata_file}")
    print("\nEvaluation complete!")
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on FaithEval benchmark with direction-based control")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--task_type", type=str, required=True, 
                        choices=["unanswerable", "inconsistent", "counterfactual"],
                        help="Type of FaithEval task to evaluate")
    parser.add_argument("--output_dir", type=str, default="output/eval/faitheval_direction", 
                        help="Directory to save output files")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--max_items", type=int, default=-1, help="Maximum number of items to process (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision instead of bfloat16")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing passages")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strict_match", action="store_true", help="Use strict phrase matching for evaluation")
    
    # Direction related arguments
    parser.add_argument("--direction_file", type=str, required=True, help="Path to saved direction file (.pt)")
    parser.add_argument("--direction_position", type=int, default=0, 
                        help="Position index in the saved directions tensor (if multiple directions are saved)")
    parser.add_argument("--coefficient", type=float, default=1.0, 
                        help="Coefficient to scale the direction by (can be negative)")
    parser.add_argument("--ablate", action="store_true", help="Ablate the direction instead of adding it")
    parser.add_argument("--layer", type=str, required=True, 
                        help="Layer to apply direction to")
    parser.add_argument("--hook_name", type=str, default="resid_post", 
                        help="Name of the hook point to apply the direction to")
    parser.add_argument("--pos_to_apply", type=str, default="-1",
                        help="Position to steer at (-1 for last position). Can be comma-separated for multiple positions.")
    
    args = parser.parse_args()
    
    main(args) 