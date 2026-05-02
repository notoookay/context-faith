import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import set_seed
from transformer_lens import HookedTransformer, ActivationCache
import torch.nn.functional as F
import wandb
import einops
from jaxtyping import Float
from torch import Tensor

from utils import load_jsonlines, save_file_jsonl, MODEL_ALIAS, format_prompt
from generation_utils import RAGDataset
from diff_in_mean import apply_hook_to_model


def get_prompts(
    results: List[Dict],
    model_name: str,
    tokenizer,
    prompt_type: str,
    max_passages_per_item: int = -1
) -> Dict[str, str]:
    """
    Use RAGDataset to construct prompts exactly as in apply_direction_generation.py.
    
    Args:
        results: Results from apply_direction_generation.py
        model_name: Model name
        tokenizer: Model tokenizer
        prompt_type: Type of prompt
        max_passages_per_item: Max passages per item
        
    Returns:
        Dictionary mapping item_id to prompt
    """
    # Create RAGDataset from results
    dataset = RAGDataset(
        data=results,
        max_passages_per_item=max_passages_per_item,
        tokenizer=tokenizer,
        prompt_type=prompt_type,
        n_samples=1,
        model_name=model_name
    )
    
    # Create mapping from item_id to prompt
    prompt_map = {}
    for task in dataset.tasks:
        item_id = task['item_id']
        prompt_map[item_id] = task['prompt']
    
    return prompt_map


def calculate_layer_top5_prob_sum_for_response(
    top5_prob_sums: Float[Tensor, "layer pos"],
) -> Dict[str, Float[Tensor, "layer"]]:
    """
    Calculate top-5 probability sum statistics for a given response.
    
    Args:
        top5_prob_sums: Top-5 probability sums for each layer and position
        
    Returns:
        Dictionary containing top-5 probability sum statistics
    """
    
    avg_top5_prob_sum = torch.mean(top5_prob_sums, dim=1)  # [layer]
    min_top5_prob_sum = torch.min(top5_prob_sums, dim=1)[0]  # [layer]
    max_top5_prob_sum = torch.max(top5_prob_sums, dim=1)[0]  # [layer]
    std_top5_prob_sum = torch.std(top5_prob_sums, dim=1)  # [layer]
    
    return {
        'avg_top5_prob_sum': avg_top5_prob_sum.cpu().numpy().tolist(),
        'min_top5_prob_sum': min_top5_prob_sum.cpu().numpy().tolist(),
        'max_top5_prob_sum': max_top5_prob_sum.cpu().numpy().tolist(),
        'std_top5_prob_sum': std_top5_prob_sum.cpu().numpy().tolist(),
    }


def compute_top5_prob_sum_stats(
    model: HookedTransformer, 
    cache: ActivationCache, 
    layer: int = -1, 
    pos_slice: slice | int = -1,
    k: int = 5
) -> Tuple[Float[Tensor, "layer pos"], List[str]]:
    """
    Compute the sum of probabilities of top-k tokens for each layer.
    
    Args:
        model: The HookedTransformer model
        cache: The activation cache from running the model
        layer: Which layer to analyze up to (default: -1 for all layers)
        pos_slice: Position slice to analyze (default: -1 for last position)
        k: Number of top tokens to consider (default: 5)
    
    Returns:
        top_k_prob_sums: Sum of top-k probabilities for each layer and position
        labels: Layer labels
    """
    # Get accumulated residual stream for each layer
    accumulated_residual, labels = cache.accumulated_resid(
        layer=layer, pos_slice=pos_slice, return_labels=True, apply_ln=True
    )  # [layer, batch, pos, d_model]
    accumulated_residual = accumulated_residual[:, 0, :, :].to(dtype=torch.bfloat16)
    
    # Get logits for all vocabulary tokens
    all_logits = einops.einsum(
        accumulated_residual, 
        model.W_U,
        "layer pos d_model, d_model vocab -> layer pos vocab"
    ).detach().cpu().to(dtype=torch.float32)
    
    # Compute probabilities using softmax
    probs = torch.softmax(all_logits, dim=-1)  # [layer, pos, vocab]
    
    # Get top-k probabilities and sum them
    top_k_probs, _ = torch.topk(probs, k=k, dim=-1)  # [layer, pos, k]
    top_k_prob_sums = torch.sum(top_k_probs, dim=-1)  # [layer, pos]
    
    return top_k_prob_sums, labels


def analyze_top5_prob_sum_for_response(
    model: HookedTransformer,
    prompt: str,
    response: str,
    direction: Optional[torch.Tensor] = None,
    layer: Optional[List[int]] = None,
    coefficient: float = 1.0,
    ablate: bool = False,
    position: int = -1,
    hook_name: str = "resid_post",
    k: int = 5,
) -> Dict:
    """
    Analyze top-k probability sums for a given response.
    
    Args:
        model: The transformer model
        prompt: The input prompt
        response: The generated response
        direction: Direction vector for intervention (optional)
        layer: Layers to apply intervention to
        coefficient: Coefficient for intervention
        ablate: Whether to ablate the direction
        position: Position to apply intervention
        hook_name: Name of the hook point
        k: Number of top tokens to consider
        
    Returns:
        Dictionary containing top-k probability sum analysis
    """

    # Tokenize prompt and full text
    prompt_tokens = model.to_tokens(prompt, padding_side='left', prepend_bos=False)
    full_text = prompt + response
    full_tokens = model.to_tokens(full_text, padding_side='left', prepend_bos=False)

    # Set up hooks if direction is provided
    hook_dict = None
    if direction is not None and layer is not None:
        hook_dict = apply_hook_to_model(
            model=model,
            direction=direction,
            layer=layer,
            coeff=coefficient,
            ablate=ablate,
            position=position,
            hook_name=hook_name
        )

    with torch.no_grad():
        # Set up hooks
        if hook_dict:
            hooks = [(name, hook_fn) for name, hook_fn in hook_dict.items()]
        else:
            hooks = []

        with model.hooks(fwd_hooks=hooks):
            # Run model and get cache
            logits, cache = model.run_with_cache(full_tokens)

            # Compute top-k probability sum statistics
            top_k_prob_sums, dist_labels = compute_top5_prob_sum_stats(
                model=model, 
                cache=cache, 
                layer=-1, 
                pos_slice=slice(prompt_tokens.shape[1] - 1, full_tokens.shape[1] - 1),
                k=k
            )

            layer_top5_prob_sum_stats = calculate_layer_top5_prob_sum_for_response(
                top5_prob_sums=top_k_prob_sums
            )

    return {
        "top5_prob_sums": layer_top5_prob_sum_stats,
        "labels": dist_labels,
        "k": k,
    }


def process_existing_results(
    results: List[Dict],
    model: HookedTransformer,
    prompt_type: str,
    direction: Optional[torch.Tensor] = None,
    layer: Optional[List[int]] = None,
    coefficient: float = 1.0,
    ablate: bool = False,
    position: int = -1,
    hook_name: str = "resid_post",
    max_passages_per_item: int = -1,
    k: int = 5,
) -> List[Dict]:
    """
    Process existing results and add top-k probability sum analysis.
    
    Args:
        results: List of results from apply_direction_generation.py
        model: The transformer model
        prompt_type: Type of prompt used
        direction: Direction vector for intervention (optional)
        layer: Layers to apply intervention to
        coefficient: Coefficient for intervention
        ablate: Whether to ablate the direction
        position: Position to apply intervention
        hook_name: Name of the hook point
        max_passages_per_item: Maximum number of passages per item
        k: Number of top tokens to consider
        
    Returns:
        List of results with top-k probability sum information added
    """
    # Generate prompts using RAGDataset for exact consistency
    print("Generating prompts using RAGDataset...")
    prompt_map = get_prompts(
        results=results,
        model_name=model.tokenizer.name_or_path,
        tokenizer=model.tokenizer,
        prompt_type=prompt_type,
        max_passages_per_item=max_passages_per_item
    )
    
    processed_results = []
    
    for item in tqdm(results, desc=f"Adding top-{k} probability sum analysis"):
        
        # Process each context/response
        item_results = []
        item_id = item['id']
        
        if 'ctxs' in item:
            for ctx in item['ctxs']:
                response = ctx.get('response', '')
                passage_id = ctx.get('passage_id', 'unknown')
                
                # Get the prompt for this specific context
                prompt = prompt_map.get(item_id)
                
                if prompt is None:
                    print(f"Warning: No prompt found for item {item_id}, passage {passage_id}")
                    continue
                
                top5_prob_sum_analysis = analyze_top5_prob_sum_for_response(
                    model=model,
                    prompt=prompt,
                    response=response,
                    direction=direction,
                    layer=layer,
                    coefficient=coefficient,
                    ablate=ablate,
                    position=position,
                    hook_name=hook_name,
                    k=k,
                )
                
                # Create result with top-k probability sum information
                result = {
                    'item_id': item_id,
                    'question': item['question'],
                    'answers': item['answers'],
                    'passage_id': passage_id,
                    'title': ctx.get('title', ''),
                    'text': ctx.get('text', ''),
                    'has_answer': ctx.get('has_answer', False),
                    'response': response,
                    'accuracy': ctx.get('accuracy'),
                    'layer_top5_prob_sum_stats': top5_prob_sum_analysis,
                }
                
                item_results.append(result)
        
        processed_results.extend(item_results)
    
    return processed_results


def parse_args():
    parser = argparse.ArgumentParser(description="Add top-k probability sum analysis to existing results")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to existing results file from apply_direction_generation.py")
    parser.add_argument("--output_dir", type=str, default="output/top5_prob_sum",
                        help="Path to save results with top-k probability sum analysis")
    parser.add_argument("--model_name", type=str, required=True, help="Model name used for generation")
    parser.add_argument("--prompt_type", type=str, default="with_passage",
                        choices=["with_passage", "with_passage_check_relevance", "no_passage", "no_passage_knowledge_check"],
                        help="Type of prompt used for generation")
    parser.add_argument("--max_passages_per_item", type=int, default=-1,
                        help="Maximum number of passages per item used during generation (-1 for all)")
    parser.add_argument("--max_items", type=int, default=-1,
                        help="Maximum number of items to process (-1 for all)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of top tokens to consider (default: 5)")
    
    # Optional direction parameters (if the original generation used direction intervention)
    parser.add_argument("--direction_file", type=str, help="Path to direction file (if used)")
    parser.add_argument("--layer", type=str, help="Layers where direction was applied")
    parser.add_argument("--coefficient", type=float, default=1.0, help="Coefficient used for direction")
    parser.add_argument("--ablate", action="store_true", help="Whether direction was ablated")
    parser.add_argument("--position", type=int, default=-1, help="Position where direction was applied")
    parser.add_argument("--hook_name", type=str, default="resid_post", help="Hook name used")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="top5_prob_sum_analysis", help="Wandb project name")
    
    return parser.parse_args()


def main(args):
    
    set_seed(args.seed)
    
    # Initialize wandb if requested
    run_name = f"{args.model_name}_{args.prompt_type}_top{args.k}_{'.'.join(args.input_file.split('/')[-1].split('.')[:-1])}"
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                'run_name': run_name,
                'model_name': args.model_name,
                'prompt_type': args.prompt_type,
                'max_passages_per_item': args.max_passages_per_item,
                'max_items': args.max_items,
                'k': args.k,
                'direction_file': args.direction_file,
                'layer': args.layer,
                'coefficient': args.coefficient,
                'ablate': args.ablate,
                'position': args.position,
                'hook_name': args.hook_name,
                'seed': args.seed,
                'input_file': args.input_file,
            }
        )
    
    # Load existing results
    print(f"Loading existing results from {args.input_file}")
    results = load_jsonlines(args.input_file)
    if args.max_items != -1:
        results = results[:args.max_items]
    print(f"Loaded {len(results)} items")
    
    # Load model
    if args.model_name in MODEL_ALIAS:
        model_name = MODEL_ALIAS[args.model_name]
    else:
        model_name = args.model_name
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name} on {device}")
    
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    
    # Load direction if specified
    direction = None
    layer = None
    if args.direction_file:
        print(f"Loading direction from {args.direction_file}")
        direction = torch.load(args.direction_file, map_location=device)
        
        if args.layer:
            layer = [int(l.strip()) for l in args.layer.split(',')]
    
    # Process results
    print(f"Processing results and calculating top-{args.k} probability sums...")
    processed_results = process_existing_results(
        results=results,
        model=model,
        prompt_type=args.prompt_type,
        direction=direction,
        layer=layer,
        coefficient=args.coefficient,
        ablate=args.ablate,
        position=args.position,
        hook_name=args.hook_name,
        max_passages_per_item=args.max_passages_per_item,
        k=args.k,
    )
    
    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(args.output_dir, f"{run_name}.jsonl")
    print(f"Saving results with top-{args.k} probability sum analysis to {output_file}")
    save_file_jsonl(processed_results, output_file)
    
    # Print summary
    if processed_results:
        avg_top5_prob_sums = [r['layer_top5_prob_sum_stats']['top5_prob_sums']['avg_top5_prob_sum'] for r in processed_results]
        
        overall_avg = np.mean([sums[-1] for sums in avg_top5_prob_sums])  # Last layer average
        overall_std = np.std([sums[-1] for sums in avg_top5_prob_sums])
        overall_min = np.min([sums[-1] for sums in avg_top5_prob_sums])
        overall_max = np.max([sums[-1] for sums in avg_top5_prob_sums])
        
        print(f"\nTop-{args.k} Probability Sum Analysis Summary:")
        print(f"Total processed items: {len(processed_results)}")
        print(f"Average last layer top-{args.k} prob sum: {overall_avg:.4f} ± {overall_std:.4f}")
        print(f"Min last layer top-{args.k} prob sum: {overall_min:.4f}")
        print(f"Max last layer top-{args.k} prob sum: {overall_max:.4f}")

        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'total_items': len(processed_results),
                f'top{args.k}_prob_sum/overall_last_layer_avg': overall_avg,
                f'top{args.k}_prob_sum/overall_last_layer_std': overall_std,
                f'top{args.k}_prob_sum/overall_last_layer_min': overall_min,
                f'top{args.k}_prob_sum/overall_last_layer_max': overall_max,
            })
    
    if args.use_wandb:
        wandb.finish()
    
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args) 