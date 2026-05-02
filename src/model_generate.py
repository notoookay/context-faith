import json
import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import set_seed
import numpy as np
from datasets import load_dataset
import wandb
from vllm import LLM, SamplingParams

from utils import save_file_jsonl, format_prompt, MODEL_ALIAS
from generation_evaluate import evaluate_answer
from generation_utils import RAGDataset


def main(args):
    
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for model generation with vLLM")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_dataset('json', data_files=args.input_file, split='train')
    print(f"Loaded {len(data)} items.")

    # Limit the number of items if specified
    if args.max_items > 0:
        data = data.select(range(min(args.max_items, len(data))))
        print(f"Limited to {len(data)} items.")

    # Set model name
    if args.model_name in MODEL_ALIAS:
        args.model_name = MODEL_ALIAS[args.model_name]
    else:
        print(f"Warning: Model name '{args.model_name}' is not in alias list, using as is.")

    print(f"Prompt type: {args.prompt_type}")
    print(f"Greedy decoding: {not args.do_sample}")
    print(f"Number of generations per question: {args.n_samples}")
    

    
    # Load model with vLLM
    print(f"Loading model {args.model_name} with vLLM...")
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.n_gpu,
        dtype="float16" if args.fp16 else "bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
    )
    tokenizer = model.get_tokenizer()
    print("vLLM model loaded successfully")
    
    # Initialize wandb if requested
    args.run_name = f"{args.model_name.replace('/', '--').replace('.', '-')}__{args.prompt_type}__{args.n_samples}_{args.input_file.split('/')[-1].split('.')[0]}_generation"
    if args.dataset_with_modified_passages:
        args.run_name += "_with_modified_passages"
    if args.use_wandb:
        wandb.init(
            project="model-generation", 
            name=args.run_name,
            config=vars(args)
        )

    # Create dataset
    print("Creating dataset...")
    dataset = RAGDataset(
        data, 
        args.max_passages_per_item, 
        tokenizer, 
        args.prompt_type, 
        args.n_samples, 
        args.dataset_with_modified_passages,
        args.model_name
    )
    print(f"Created dataset with {len(dataset)} items")

    # Process with vLLM
    print("Processing prompts with vLLM...")
    all_prompts = []
    metadata_map = {}
    
    # Collect all prompts and metadata
    for idx, item in enumerate(dataset):
        all_prompts.append(item['prompt'])
        # Store metadata with the prompt's index
        metadata_map[idx] = {
            'item_id': item['item_id'],
            'question': item['question'],
            'answers': item['answers'],
            'passage': item['passage'],
            'passage_idx': item['passage_idx']
        }
    
    print(f"Processing {len(all_prompts)} prompts...")
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature if args.do_sample else 0,
        top_p=args.top_p if args.do_sample and args.top_p < 1.0 else 1.0,
        max_tokens=args.max_new_tokens,
    )
    
    # Generate responses using vLLM (handles batching internally)
    results_map = {}
    
    # Generate in batches for progress tracking
    batch_size = args.batch_size
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating responses"):
        batch_prompts = all_prompts[i:i+batch_size]
        batch_outputs = model.generate(batch_prompts, sampling_params)
        
        # Process batch results
        for j, output in enumerate(batch_outputs):
            prompt_idx = i + j
            meta = metadata_map[prompt_idx]
            
            # Extract response (vLLM removes the prompt automatically)
            response = output.outputs[0].text
            
            # Evaluate the answer
            eval_results = evaluate_answer(response, meta['answers'])
            
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
                    'response': response,
                    'accuracy': eval_results['acc'],
                }
            else:
                passage_result = {
                    'passage_id': meta['passage'].get('id', 'unknown'),
                    'title': meta['passage'].get('title', ''),
                    'text': meta['passage']['text'],
                    "modified_passage": meta['passage']['modified_passage'] if args.dataset_with_modified_passages else None,
                    'has_answer': meta['passage']['hasanswer'] if 'hasanswer' in meta['passage'] else meta['passage']['has_answer'],
                    'response': response,
                    'accuracy': eval_results['acc'],
                }
            
            results_map[item_id]['ctxs'].append(passage_result)
    
    # Convert results map to list
    results = list(results_map.values())
    
    # Calculate and print statistics
    print("\n=== Generation Summary ===")
    print(f"Total items processed: {len(results)}")
    
    # Gather statistics
    total_passages = sum(len(item['ctxs']) for item in results)
    print(f"Total passages/examples: {total_passages}")

    # Calculate accuracy
    total_accuracy = sum(sum(p['accuracy'] for p in item['ctxs']) for item in results)
    avg_accuracy = total_accuracy / total_passages if total_passages > 0 else 0
    print(f"Average accuracy: {avg_accuracy:.4f}")
    
    if args.use_wandb:
        wandb.log({
            "total_examples": total_passages,
            "avg_accuracy": avg_accuracy,
        })
        
    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.run_name}.jsonl")
    
    print(f"\nSaving generated responses to {output_file}...")
    save_file_jsonl(results, output_file)
    
    print("\nGeneration complete!")
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model responses")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="output/model_generation", help="Directory to save output files")
    parser.add_argument("--model_name", type=str, default="gpt2-small", help="Hugging Face model name")
    parser.add_argument("--prompt_type", type=str, default="with_passage", 
                        choices=["with_passage", "with_passage_check_relevance", "no_passage", "no_passage_knowledge_check"],
                        help="Type of prompt to use")
    parser.add_argument("--dataset_with_modified_passages", action="store_true", help="The loaded dataset is with modified passages")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--max_items", type=int, default=-1, help="Maximum number of items to process (-1 for all)")
    parser.add_argument("--max_passages_per_item", type=int, default=-1, help="Maximum number of passages per item (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision instead of bfloat16")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(args) 