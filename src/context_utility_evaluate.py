import argparse
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, set_seed
import numpy as np
from datasets import load_dataset
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from vllm import LLM, SamplingParams

from utils import load_jsonlines, save_file_jsonl, format_passage
from generation_evaluate import LLMEvaluator


def create_utility_visualizations(results, output_dir, run_name=None):
    """Create visualizations based on the context utility evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = run_name or "context_utility_results"
    
    # Prepare data
    data = []
    for item in results:
        for passage_result in item['ctxs']:
            data.append({
                'question_id': item['id'],
                'question': item['question'],
                'context_utility': passage_result['context_utility'],
            })
    
    df = pd.DataFrame(data)
    
    # Context utility distribution
    plt.figure(figsize=(12, 8))
    
    # Overall utility distribution
    plt.subplot(2, 2, 1)
    utility_counts = df['context_utility'].value_counts().sort_index()
    plt.bar(['Not Useful (0)', 'Useful (1)'], [utility_counts.get(0, 0), utility_counts.get(1, 0)], 
            color=['red', 'green'], alpha=0.7)
    plt.title('Overall Context Utility Distribution')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    df['context_utility'].hist(bins=2, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Context Utility Histogram')
    plt.xlabel('Utility Score')
    plt.ylabel('Frequency')
    plt.xticks([0, 1], ['Not Useful', 'Useful'])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{base_filename}_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Load generated responses
    print(f"Loading generated responses from {args.input_file}...")
    data = load_dataset('json', data_files=args.input_file, split='train')
    print(f"Loaded {len(data)} items with generated responses.")
    
    # Convert to list
    results = [item for item in data]
    
    # Initialize wandb if requested
    if args.run_name:
        run_name = args.run_name
    else:
        input_file_base = '.'.join(os.path.basename(args.input_file).split('.')[:-1]) # remove the file extension
        run_name = f"{input_file_base}_context_utility_eval_{args.llm_eval_model.replace('/', '--').replace('.', '-')}"
    
    if args.use_wandb:
        wandb.init(
            project="context-utility-evaluation", 
            name=run_name,
            config={
                "input_file": args.input_file,
                "evaluation_model": args.llm_eval_model,
                "batch_size": args.batch_size,
            }
        )
    
    # Initialize LLM evaluator
    print(f"Using LLM-based context utility evaluation with model: {args.llm_eval_model}")
    try:
        llm_evaluator = LLMEvaluator(
            model_name=args.llm_eval_model,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"Error initializing LLM evaluator: {e}")
        raise e
        
    # Evaluate context utility
    print("Evaluating context utility...")
    
    # Store evaluation results
    evaluated_results = []
    
    # Process in batches if using batch evaluation
    if args.batch_eval:
        # Prepare mapping to track original items and passages
        passage_map = {}  # Maps passage ID to (item_idx, passage_idx)
        reverse_passage_map = {}  # Maps (item_idx, passage_idx) to passage_id
        passage_id_counter = 0
        
        # Collect all passages from all items for efficient batch processing
        all_questions = []
        all_contexts = []
        all_model_answers = []
        all_passage_ids = []
        
        # Collect all passages across items
        for item_idx, item in enumerate(results):
            for passage_idx, passage in enumerate(item['ctxs']):
                all_questions.append(item['question'])
                passage_text = format_passage(passage['title'], passage['text'])
                all_contexts.append(passage_text)
                all_model_answers.append(passage['response'])
                
                # Generate a unique ID for this passage
                passage_id = passage_id_counter
                passage_map[passage_id] = (item_idx, passage_idx)
                reverse_passage_map[(item_idx, passage_idx)] = passage_id
                all_passage_ids.append(passage_id)
                passage_id_counter += 1
        
        total_passages = len(all_passage_ids)
        print(f"Total passages to evaluate: {total_passages}")
        
        # Process in optimized batches across items
        utility_results = {}  # Map of passage_id -> utility score
        
        # Process in batches
        for i in tqdm(range(0, len(all_questions), args.batch_size), desc="Processing batches"):
            batch_questions = all_questions[i:i+args.batch_size]
            batch_contexts = all_contexts[i:i+args.batch_size]
            batch_answers = all_model_answers[i:i+args.batch_size]
            batch_passage_ids = all_passage_ids[i:i+args.batch_size]
            
            # Evaluate batch
            batch_utility_scores = llm_evaluator.batch_evaluate_context_utility(
                batch_questions,
                batch_contexts,
                batch_answers
            )
            
            # Store results with their passage IDs
            for j, utility in enumerate(batch_utility_scores):
                passage_id = batch_passage_ids[j]
                utility_results[passage_id] = utility
        
        # Now reconstruct the original structure with evaluation results
        for item_idx, item in enumerate(results):
            item_result = {
                'id': item['id'],
                'question': item['question'],
                'answers': item['answers'],
                'ctxs': []
            }
            
            # Add results to each passage
            for passage_idx, passage in enumerate(item['ctxs']):
                # Get the passage ID directly from the reverse mapping
                passage_id = reverse_passage_map[(item_idx, passage_idx)]
                utility = utility_results[passage_id]
                
                # Create a copy of the passage with evaluation added
                passage_result = passage.copy()
                passage_result['context_utility'] = utility
                
                item_result['ctxs'].append(passage_result)
            
            evaluated_results.append(item_result)
    else:
        # Process each item individually
        for item_data in tqdm(results, desc="Evaluating items"):
            item_results = {
                'id': item_data['id'],
                'question': item_data['question'],
                'answers': item_data['answers'],
                'ctxs': []
            }
            
            # Evaluate each passage
            for passage in item_data['ctxs']:
                passage_text = format_passage(passage['title'], passage['text'])
                utility = llm_evaluator.evaluate_context_utility(
                    item_data['question'], 
                    passage_text,
                    passage['response']
                )
                
                passage_result = passage.copy()
                passage_result['context_utility'] = utility
                
                item_results['ctxs'].append(passage_result)
                
            evaluated_results.append(item_results)
    
    # Calculate and print statistics
    print("\n=== Context Utility Evaluation Results ===")
    print(f"Total items processed: {len(evaluated_results)}")
    print(f"Evaluation model: {args.llm_eval_model}")

    # Gather statistics
    total_passages = sum(len(item['ctxs']) for item in evaluated_results)

    # Calculate utility statistics
    total_useful = sum(sum(1 for p in item['ctxs'] if p['context_utility']) for item in evaluated_results)
    
    print(f"Total passages: {total_passages}")
    print(f"Contexts marked as useful: {total_useful} ({total_useful/total_passages*100:.2f}%)")
    print(f"Contexts marked as not useful: {total_passages - total_useful} ({(total_passages - total_useful)/total_passages*100:.2f}%)")
    

    # Log summary metrics to wandb
    if args.use_wandb:
        wandb.log({
            "total_passages": total_passages,
            "total_useful_contexts": total_useful,
            "overall_utility_rate": total_useful/total_passages if total_passages > 0 else 0,
        })
        
    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{run_name}.jsonl")
    
    print(f"\nSaving context utility evaluation results to {output_file}...")
    save_file_jsonl(evaluated_results, output_file)

    # Create visualizations
    if args.create_plots:
        print("Creating context utility visualizations...")
        create_utility_visualizations(evaluated_results, output_dir, run_name=run_name)

    print("\nContext utility evaluation complete!")

    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate context utility using LLM-based assessment")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Path to generated responses JSONL file from model_generate.py")
    parser.add_argument("--output_dir", type=str, default="output/context_utility_evaluation", 
                       help="Directory to save output files")
    parser.add_argument("--run_name", type=str, default=None, 
                       help="Run name for output files and Weights & Biases")
    parser.add_argument("--create_plots", action="store_true", 
                       help="Create visualizations")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Enable Weights & Biases logging")
    
    # LLM evaluation arguments
    parser.add_argument("--llm_eval_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Model to use for LLM-based context utility evaluation")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for LLM evaluation across all items. Higher values improve throughput " 
                            "but require more GPU memory. Recommended values: 8-32 depending on model size and GPU memory.")
    parser.add_argument("--batch_eval", action="store_true", 
                       help="Perform LLM evaluation in batches across all items. This is more efficient when evaluating many passages.")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    main(args) 