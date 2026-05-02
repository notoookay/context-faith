import argparse
import os
import json
import torch
from functools import partial
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from torch import Tensor
import wandb

from utils import load_jsonlines, save_file_jsonl, MODEL_ALIAS
from generation_utils import RAGDataset, collate_fn
from generation_evaluate import evaluate_answer
from diff_in_mean import apply_hook_to_model

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
    run_suffix = "ablate" if args.ablate else "add"
    args.run_name = f"{args.model_name.replace('/', '--').replace('.', '-')}__{args.prompt_type}__{os.path.basename(args.direction_file).split('.')[0]}_coef{args.coefficient}_{run_suffix}_layer{args.layer}_{args.seed}"
    if args.use_wandb:
        wandb.init(
            project="direction-generation", 
            name=args.run_name,
            config=vars(args)
        )

    print(f"Prompt type: {args.prompt_type}")
    print(f"Greedy decoding: {not args.do_sample}")
    print(f"Number of generations per question: {args.n_samples}")
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

    # Set model to evaluation mode
    model.eval()

    # Load direction
    print(f"Loading direction from {args.direction_file}...")
    direction = torch.load(args.direction_file, map_location=device)

    print(f"Direction shape: {direction.shape}")

    if args.layer is not None:
        args.layer = [int(layer) for layer in args.layer.split(',')]

    # Create dataset
    print("Creating dataset...")
    dataset = RAGDataset(
        data,
        args.max_passages_per_item,
        model.tokenizer,
        args.prompt_type,
        args.n_samples,
        model_name=args.model_name,
    )
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

    # Process data in batches
    results_map = {}  # Map to store results indexed by item_id

    # Process in batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating steered responses")):
        # Move input tensors to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata']

        # Log steering positions for the first batch
        if batch_idx == 0:
            token_strs = model.to_str_tokens(input_ids[0])
            print(f"Applying direction at position: {args.pos_to_apply}")
            print(f"Token at position: {token_strs[args.pos_to_apply]!r}")
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
                position=args.pos_to_apply,
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

            # Evaluate the answer using string matching
            eval_results = evaluate_answer(steered_response, meta['answers'])

            # Store the data
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
                    'response': steered_response,
                    'accuracy': eval_results['acc'],
                }
            else:
                passage_result = {
                    'passage_id': meta['passage'].get('id', 'unknown'),
                    'title': meta['passage'].get('title', ''),
                    'text': meta['passage']['text'],
                    'has_answer': meta['passage']['hasanswer'] if 'hasanswer' in meta['passage'] else meta['passage']['has_answer'],
                    'response': steered_response,
                    'accuracy': eval_results['acc'],
                }

            results_map[item_id]['ctxs'].append(passage_result)

    # Convert results map to list
    results = list(results_map.values())

    # Calculate and print statistics
    print("\n=== Generation Summary ===")
    print(f"Total items processed: {len(results)}")

    # Gather statistics
    total_ctxs = sum(len(item['ctxs']) for item in results)
    print(f"Total passages/examples: {total_ctxs}")

    # Calculate accuracy
    total_accuracy = sum(sum(p['accuracy'] for p in item['ctxs']) for item in results)
    avg_accuracy = total_accuracy / total_ctxs if total_ctxs > 0 else 0
    print(f"Average accuracy: {avg_accuracy:.4f}")

    if args.use_wandb:
        wandb.log({
            "total_examples": total_ctxs,
            "avg_accuracy": avg_accuracy,
        })

    output_dir = os.path.join(args.output_dir, args.model_name.replace('/', '--').replace('.', '-'))
    os.makedirs(output_dir, exist_ok=True)

    # Include direction information in the output filename
    direction_str = f"_{'ablate' if args.ablate else 'add'}_coef{args.coefficient}"
    position_str = f"_pos{args.pos_to_apply}"
    layer_str = f"_layer{'_'.join(str(l) for l in args.layer)}" if args.layer else ""
    output_file = os.path.join(output_dir, f"{args.input_file.split('/')[-1].split('.')[0]}_{args.prompt_type}{direction_str}{position_str}{layer_str}.jsonl")

    print(f"\nSaving generated responses to {output_file}...")
    save_file_jsonl(results, output_file)

    print("\nGeneration complete!")

    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()

    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model responses with direction-based steering")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="output/direction_generation", help="Directory to save output files")
    parser.add_argument("--model_name", type=str, default="gpt2-small", help="Hugging Face model name")
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
    
    # Direction related arguments
    parser.add_argument("--direction_file", type=str, required=True, help="Path to saved direction file (.pt)")
    parser.add_argument("--coefficient", type=float, default=1.0, 
                        help="Coefficient to scale the direction by (can be negative)")
    parser.add_argument("--ablate", action="store_true", help="Ablate the direction instead of adding it")
    parser.add_argument("--layer", type=str, required=True,
                        help="Layer to apply direction to")
    parser.add_argument("--hook_name", type=str, default="resid_post", 
                        help="Name of the hook point to apply the direction to")
    parser.add_argument("--pos_to_apply", type=int, default=-1,
                        help="Position to apply direction at")
    
    args = parser.parse_args()
    
    main(args) 
