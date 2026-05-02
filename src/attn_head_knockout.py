"""
The script is modified from `data_analysis/knockout.py`
"""

import argparse
import os
import json
import torch
from functools import partial
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from torch import Tensor
import wandb

from utils import load_jsonlines, save_file_jsonl, MODEL_ALIAS
from generation_utils import RAGDataset, collate_fn
from generation_evaluate import evaluate_answer

def knockout_hook_fn(value, hook, head):
    """Zero out specific attention heads"""
    # value shape: [batch_size, seq_len, n_heads, d_head]
    # if value.shape[1] == 1:
    #     value[:, -1, head, :] = 0
    #     return value
    value[:, -1, head, :] = 0
    return value

def create_knockout_hooks(model: HookedTransformer, heads_to_knockout: List[Tuple[int, int]]):
    """
    Create hook functions for knocking out specified attention heads.
    
    Args:
        model: The transformer_lens model
        heads_to_knockout: List of (layer, head) tuples to knockout
        
    Returns:
        List of (hook_name, hook_function) tuples
    """
    hooks = []
    
    for layer, head in heads_to_knockout:
        # Get the hook name for the attention head output (z)
        hook_name = utils.get_act_name("z", layer)
        # Create partial function with the specific head to knockout
        hook_fn = partial(knockout_hook_fn, head=head)
        hooks.append((hook_name, hook_fn))
    
    return hooks

def main(args):
    set_seed(args.seed)
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonlines(args.input_file)
    print(f"Loaded {len(data)} items.")

    # Limit the number of items if specified
    if args.max_items > 0:
        data = data[:args.max_items]
        print(f"Limited to {len(data)} items.")

    # Set model name
    if args.model_name in MODEL_ALIAS:
        model_name = MODEL_ALIAS[args.model_name]
    else:
        model_name = args.model_name
        print(f"Warning: Model name '{args.model_name}' is not in alias list, using as is.")

    # Initialize wandb if requested
    args.run_name = f"{args.model_name.replace('/', '--').replace('.', '-')}__{args.prompt_type}__{args.input_file.split('/')[-1].split('.')[0]}__head_knockout_top-{len(COPYING_HEADS)}heads_{args.seed}"
    if args.use_wandb:
        wandb.init(
            project="attn-head-knockout", 
            name=args.run_name,
            config=vars(args)
        )

    print(f"Prompt type: {args.prompt_type}")
    print(f"Greedy decoding: {not args.do_sample}")
    print(f"Number of generations per question: {args.n_samples}")
    print(f"Knocking out {len(COPYING_HEADS)} heads: {COPYING_HEADS}")
    print(f"Loading model {model_name} with TransformerLens...")
    
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        n_devices=args.n_gpu,
        dtype='bfloat16',
        default_padding_side='left',
    )
    print("Model loaded successfully")

    if not ("it" in model_name.lower() or "instruct" in model_name.lower()):
        print("Using base model")
        model.tokenizer.eos_token_id = model.to_single_token("\n")
    
    model.eval()
    
    # Create dataset
    print("Creating dataset...")
    dataset = RAGDataset(data, args.max_passages_per_item, model.tokenizer, args.prompt_type, args.n_samples, model_name=args.model_name)
    print(f"Created dataset with {len(dataset)} items")

    # Create DataLoader with custom collate function
    custom_collate = lambda batch: collate_fn(batch, model.tokenizer, args.model_name)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=args.num_workers if args.num_workers > 0 else 0,
        pin_memory=True if torch.cuda.is_available() and args.num_workers > 0 else False
    )
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # Create knockout hooks
    knockout_hooks = create_knockout_hooks(model, COPYING_HEADS)
    print(f"Created {len(knockout_hooks)} knockout hooks")
    
    # Process data in batches
    results_map = {}  # Map to store results indexed by item_id
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating responses with head knockout")):
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata']
            
        # Log information for the first batch
        if batch_idx == 0:
            token_strs = model.to_str_tokens(input_ids[0])
            print(f"Batch size: {input_ids.shape[0]}")
            print(f"Sequence length: {input_ids.shape[1]}")
            print(f"Last 10 tokens: {token_strs[-10:]}")  # Show last 10 tokens
        
        with torch.no_grad():
            with model.hooks(fwd_hooks=knockout_hooks):
                outputs_knockout = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.do_sample else 0.0,
                    do_sample=args.do_sample,
                    verbose=False
                )
        
        # Process results
        for i, (meta, output_knockout) in enumerate(zip(metadata, outputs_knockout)):
            input_length = meta['prompt_length']
            
            knockout_tokens = output_knockout[input_length:]
            
            eos_token_id = model.tokenizer.eos_token_id
            if eos_token_id is not None:
                eos_positions = (knockout_tokens == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    knockout_tokens = knockout_tokens[:eos_positions[0]]
            
            knockout_response = model.tokenizer.decode(knockout_tokens, skip_special_tokens=True)
            
            eval_results = evaluate_answer(knockout_response, meta['answers'])
            
            item_id = meta['item_id']
            if item_id not in results_map:
                results_map[item_id] = {
                    'id': item_id,
                    'question': meta['question'],
                    'answers': meta['answers'],
                    'ctxs': []
                }
            
            if args.prompt_type.startswith("no_passage"):
                passage_result = {
                    'passage_id': 'none',
                    'title': 'N/A',  
                    'text': 'N/A',
                    'has_answer': True,  # We always consider it as "has answer" for no_passage mode
                    'response': knockout_response,
                    'accuracy': eval_results['acc'],
                }
            else:
                passage_result = {
                    'passage_id': meta['passage'].get('id', 'unknown'),
                    'title': meta['passage'].get('title', ''),
                    'text': meta['passage']['text'],
                    'has_answer': meta['passage']['hasanswer'] if 'hasanswer' in meta['passage'] else meta['passage']['has_answer'],
                    'response': knockout_response,
                    'accuracy': eval_results['acc'],
                }
            
            results_map[item_id]['ctxs'].append(passage_result)
    
    results = list(results_map.values())
    
    print("\n=== Head Knockout Generation Summary ===")
    print(f"Total items processed: {len(results)}")
    print(f"Heads knocked out: {len(COPYING_HEADS)}")
    
    total_ctxs = sum(len(item['ctxs']) for item in results)
    print(f"Total passages/examples: {total_ctxs}")
    
    total_accuracy = sum(sum(p['accuracy'] for p in item['ctxs']) for item in results)
    avg_accuracy = total_accuracy / total_ctxs if total_ctxs > 0 else 0
    print(f"Average accuracy with head knockout: {avg_accuracy:.4f}")
    
    if args.use_wandb:
        wandb.log({
            "total_examples": total_ctxs,
            "avg_accuracy_knockout": avg_accuracy,
            "heads_knocked_out": len(COPYING_HEADS),
        })
    
    output_dir = os.path.join(args.output_dir, args.model_name.replace('/', '--').replace('.', '-'))
    os.makedirs(output_dir, exist_ok=True)
    
    knockout_str = f"_head_knockout_{len(COPYING_HEADS)}heads"
    output_file = os.path.join(output_dir, f"{args.input_file.split('/')[-1].split('.')[0]}_{args.prompt_type}{knockout_str}.jsonl")
    
    print(f"\nSaving generated responses to {output_file}...")
    save_file_jsonl(results, output_file)
    
    print("\nHead knockout generation complete!")
    
    if args.use_wandb:
        wandb.finish()
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model responses with attention head knockout")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="output/attn_head_knockout", help="Directory to save output files")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--prompt_type", type=str, default="with_passage", 
                        choices=["with_passage", "with_passage_check_relevance", "no_passage", "no_passage_knowledge_check"],
                        help="Type of prompt to use")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--max_items", type=int, default=-1, help="Maximum number of items to process (-1 for all)")
    parser.add_argument("--max_passages_per_item", type=int, default=-1, help="Maximum number of passages per item (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(args) 
