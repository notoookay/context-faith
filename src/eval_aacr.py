import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset
import wandb
from typing import List, Dict, Any

from aacr import AACR
from utils import save_file_jsonl, MODEL_ALIAS
from generation_evaluate import evaluate_answer


def evaluate_accuracy(predictions: List[Dict]) -> Dict[str, float]:
    """
    Evaluate accuracy of predictions.
    
    Args:
        predictions: List of dicts with 'answers' (ground truth) and 'response' (predicted)
        
    Returns:
        Dictionary with accuracy metrics
    """
    correct_count = 0
    total_count = len(predictions)
    
    for pred in predictions:
        predicted = pred['response']
        ground_truth_answers = pred['answers']
        results = evaluate_answer(predicted, ground_truth_answers)
        if results['acc'] == 1:
            correct_count += 1
        pred['accuracy'] = results['acc']
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count
    }


def main(args):
    set_seed(args.seed)
    
    # Set model name
    if args.model_name in MODEL_ALIAS:
        model_name = MODEL_ALIAS[args.model_name]
    else:
        model_name = args.model_name
        print(f"Warning: Model name '{args.model_name}' is not in alias list, using as is.")
    
    # Load data
    print(f"Loading data from {args.input_path}...")
    data = load_dataset('json', data_files=args.input_path, split='train')
    print(f"Loaded {len(data)} items.")
    
    # Limit the number of items if specified
    if args.max_items > 0:
        data = data.select(range(min(args.max_items, len(data))))
        print(f"Limited to {len(data)} items.")
    
    print(f"Loading AACR system with model {model_name}...")
    
    # Initialize AACR system
    aacr = AACR(
        model_name=model_name,
        prober_path=args.prober_path,
        optimal_layer=args.optimal_layer,
        context_threshold=args.context_threshold,
        batch_size=args.batch_size
    )
    print("AACR system loaded successfully")
    
    # Initialize wandb if requested
    args.run_name = f"{args.model_name.replace('/', '--')}__aacr__{args.input_path.split('/')[-1].split('.')[0]}__{args.seed}"
    if args.use_wandb:
        wandb.init(
            project="aacr-evaluation",
            name=args.run_name,
            config=vars(args)
        )
    
    # Extract questions, contexts, and metadata directly from data
    all_questions = []
    all_contexts = []
    all_metadata = []
    
    for item in data:
        # Handle different possible data formats
        if 'ctxs' in item and len(item['ctxs']) > 0:
            # Format with multiple contexts - use the first one or iterate through all
            for ctx in item['ctxs']:
                all_questions.append(item['question'])
                all_contexts.append(ctx)
                all_metadata.append({
                    'id': item.get('id', item.get('qid', len(all_metadata))),
                    'question': item['question'],
                    'answers': item['answers'],
                    'passage': ctx
                })
        else:
            raise ValueError("No context found in item")
    
    print(f"Processing all {len(all_questions)} items with AACR...")
    batch_results = aacr.batch_answer(all_questions, all_contexts)
    
    # Process results
    all_results = []
    decision_stats = {
        'use_context': 0,
        'use_parametric_knowledge': 0,
        'unable_to_answer': 0
    }
    
    print("Processing results...")
    for (answer, decision_info), question, context, meta in zip(batch_results, all_questions, all_contexts, all_metadata):
        # Update decision statistics
        decision_stats[decision_info['decision']] += 1
        
        result = {
            'id': meta['id'],
            'question': question,
            'context': context,
            'answers': meta['answers'],
            'response': answer,
            'decision_info': decision_info
        }
        all_results.append(result)
    
    # Evaluate overall accuracy
    overall_metrics = evaluate_accuracy(all_results)
    
    # Evaluate by decision type
    metrics_by_decision = {}
    for decision_type in decision_stats.keys():
        decision_results = [r for r in all_results if r['decision_info']['decision'] == decision_type]
        if decision_results:
            metrics_by_decision[decision_type] = evaluate_accuracy(decision_results)
    
    # Print results
    print(f"\n=== AACR Evaluation Results ===")
    print(f"Overall Accuracy: {overall_metrics['accuracy'] * 100:.2f}% ({overall_metrics['correct_count']}/{overall_metrics['total_count']})")
    
    print(f"\n=== Decision Statistics ===")
    total_decisions = sum(decision_stats.values())
    for decision_type, count in decision_stats.items():
        percentage = count / total_decisions * 100 if total_decisions > 0 else 0
        print(f"{decision_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    if metrics_by_decision:
        print(f"\n=== Accuracy by Decision Type ===")
        for decision_type, metrics in metrics_by_decision.items():
            print(f"{decision_type.replace('_', ' ').title()}: {metrics['accuracy'] * 100:.2f}% ({metrics['correct_count']}/{metrics['total_count']})")
    
    # Log to wandb if enabled
    if args.use_wandb:
        log_data = {
            "overall_accuracy": overall_metrics['accuracy'],
            "overall_correct_count": overall_metrics['correct_count'],
            "overall_total_count": overall_metrics['total_count']
        }
        
        # Add decision statistics
        for decision_type, count in decision_stats.items():
            log_data[f"decision_{decision_type}_count"] = count
            log_data[f"decision_{decision_type}_percentage"] = count / total_decisions * 100 if total_decisions > 0 else 0
        
        # Add decision type accuracies
        for decision_type, metrics in metrics_by_decision.items():
            log_data[f"accuracy_{decision_type}"] = metrics['accuracy']
        
        wandb.log(log_data)
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, f"{args.run_name}.jsonl")
    print(f"\nSaving results to {output_file}...")
    save_file_jsonl(all_results, output_file)
    
    print("\nEvaluation complete!")
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    return overall_metrics['accuracy']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data and model arguments
    parser.add_argument("--input_path", type=str, required=True, help="Path to JSONL data file")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--prober_path", type=str, required=True, help="Path to trained context utility prober")
    parser.add_argument("--optimal_layer", type=int, required=True, help="Layer number for context utility detection")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    
    # AACR specific arguments
    parser.add_argument("--context_threshold", type=float, default=0.5, help="Context utility threshold")
    
    # Evaluation arguments
    parser.add_argument("--output_dir", type=str, default="output/eval/aacr",
                        help="Directory to save output files")
    parser.add_argument("--max_items", type=int, default=-1, help="Maximum number of items to process (-1 for all)")
    
    # Logging and misc
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(args) 