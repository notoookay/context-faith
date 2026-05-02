import json
import argparse
import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
import numpy as np
from datasets import load_dataset
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from transformer_lens import HookedTransformer

from utils import load_jsonlines, save_file_jsonl, format_prompt, MODEL_ALIAS


class RAGDataset(Dataset):
    def __init__(self, data, max_passages_per_item=-1, tokenizer=None, prompt_type="with_passage", n_samples=1):
        self.tasks = []
        self.prompt_type = prompt_type
        self.n_samples = n_samples
        
        # Process all items and create tasks
        for item_idx, item in enumerate(data):
            question = item['question']
            ground_truths = item['answers']
            item_id = item.get('id', str(item_idx))
            
            # For no passage variants, create just one task per question
            if prompt_type.startswith("no_passage"):
                # Create prompt
                native_prompt = format_prompt(question, prompt_type=prompt_type)
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": native_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                # Add task
                for _ in range(n_samples):
                    self.tasks.append({
                        'item_id': item_id,
                        'question': question,
                        'ground_truths': ground_truths,
                        'passage': None,  # No passage for this mode
                        'native_prompt': native_prompt,
                        'prompt': prompt,
                        'passage_idx': -1  # No passage index
                })
            
            # For with passage variant, create task for each passage
            else:
                if 'ctxs' not in item or not item['ctxs']:
                    continue
                
                # Get passages
                passages = item['ctxs']
                if max_passages_per_item > 0:
                    passages = passages[:max_passages_per_item]
                
                # Create a task for each passage
                for passage_idx, passage in enumerate(passages):
                    # Create prompt
                    native_prompt = format_prompt(question, [passage], prompt_type="with_passage")
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": native_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    
                    # Add task
                    self.tasks.append({
                        'item_id': item_id,
                        'question': question,
                        'ground_truths': ground_truths,
                        'passage': passage,
                        'native_prompt': native_prompt,
                        'prompt': prompt,
                        'passage_idx': passage_idx
                    })
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]


def collate_fn(batch, tokenizer, model_name):
    """
    Custom collate function to process and batch items together.
    
    Args:
        batch: List of items from the dataset
        tokenizer: Tokenizer to use for encoding
        model_name: Name of the model (used to determine prompt formatting)
    
    Returns:
        Dictionary with batched tensors and metadata
    """
    # Format prompts for the model
    prompts = [item['prompt'] for item in batch]
    
    # Tokenize all prompts in the batch
    encoded_inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,  # Don't add BOS/EOS tokens automatically
    )
    
    # Preserve metadata
    metadata = [{
        'item_id': item['item_id'],
        'question': item['question'],
        'ground_truths': item['ground_truths'],
        'passage': item['passage'],
        'prompt_length': len(encoded_inputs['input_ids'][i])
    } for i, item in enumerate(batch)]
    
    return {
        'input_ids': encoded_inputs['input_ids'],
        'attention_mask': encoded_inputs['attention_mask'],
        'metadata': metadata
    }


def evaluate_answer(predicted, labels):
    """
    Evaluate the predicted answer.
    """
    if isinstance(labels, str):
        labels = [labels]
    
    accuracy = max([1 if label in predicted else 0 for label in labels])
    results = {
        'acc': accuracy,
    }
    
    return results


def create_visualizations(results, output_dir, args, run_name=None):
    """Create visualizations based on the evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use run_name for consistent file naming
    base_filename = run_name or "results"
    
    # Prepare data
    data = []
    for item in results:
        for passage_result in item['passages']:
            data.append({
                'question_id': item['id'],
                'question': item['question'],
                'has_answer': passage_result['has_answer'],
                'acc': passage_result['evaluation']['acc'],
            })
    
    df = pd.DataFrame(data)
    
    # Performance comparison between relevant and irrelevant passages
    plt.figure(figsize=(12, 6))
    metrics = ['acc']
    
    for i, metric in enumerate(metrics):
        relevant_scores = df[df['has_answer']][metric].mean()
        irrelevant_scores = df[~df['has_answer']][metric].mean()
        
        plt.subplot(1, 2, i+1)
        plt.bar(['Relevant', 'Irrelevant'], [relevant_scores, irrelevant_scores], color=['green', 'red'])
        plt.title(f'Average {metric.replace("_", " ").title()}')
        plt.ylabel('Score')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{base_filename}_performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log({"performance_comparison": wandb.Image(plot_path)})
    
    plt.close()
    
    # ROC curve (if we consider this a binary classification problem)
    plt.figure(figsize=(8, 8))
    
    for metric, color, label in zip(
        ['acc'], 
        ['blue'],
        ['Accuracy']
    ):
        if metric in df.columns and not df[metric].isna().all():
            fpr, tpr, _ = roc_curve(df['has_answer'], df[metric])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Passage Relevance Detection')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, f"{base_filename}_roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log({"roc_curve": wandb.Image(roc_path), "auc": roc_auc})
    
    plt.close()


def main(args):
    
    set_seed(args.seed)
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(f"Loading model {args.model_name} with TransformerLens...")
    
    # Load the model with TransformerLens
    try:
        model = HookedTransformer.from_pretrained(
            args.model_name,
            device=device,
            n_devices=args.n_gpu,
            dtype='bfloat16' if not args.fp16 else 'float16',
            default_padding_side='left',
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        if "unknown model" in str(e).lower():
            print("TransformerLens might not support this model directly.")
            print("Trying to load with default HuggingFace configuration...")
            model = HookedTransformer.from_pretrained_no_processing(
                args.model_name,
                device=device,
                n_devices=args.n_gpu,
                dtype='bfloat16' if not args.fp16 else 'float16',
                default_padding_side='left',
            )
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize wandb if requested
    args.run_name = f"{args.model_name.replace('/', '--')}__{args.prompt_type}__{args.n_samples}_{args.input_file.split('/')[-1].split('.')[0]}_{os.path.basename(__file__)[: -len('.py')]}"
    if args.use_wandb:
        wandb.init(
            project="rag-evaluation", 
            name=args.run_name,
            config=vars(args)
        )

    # Create dataset
    print("Creating dataset...")
    dataset = RAGDataset(data, args.max_passages_per_item, model.tokenizer, args.prompt_type, args.n_samples)
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
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Move input tensors to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata']
        
        # Generate responses with transformer_lens
        with torch.no_grad():
            # TransformerLens batch generation - for simplicity we'll do greedy or sampling
            if args.do_sample:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p if args.top_p < 1.0 else None,
                    verbose=False
                )
            else:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    verbose=False
                )
        
        # Process results
        for i, (meta, output) in enumerate(zip(metadata, outputs)):
            # Get the length of the input
            input_length = meta['prompt_length']
            
            # Extract only the generated tokens (excluding input)
            new_tokens = output[input_length:]
            
            # Find EOS token (if any) to truncate the response
            eos_token_id = model.tokenizer.eos_token_id
            if eos_token_id is not None:
                # Find the first occurrence of EOS token
                eos_positions = (new_tokens == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    # Truncate at the first EOS token
                    first_eos_pos = eos_positions[0].item()
                    new_tokens = new_tokens[:first_eos_pos]
            
            # Decode the response and remove any special tokens that might remain
            response = model.tokenizer.decode(new_tokens)
            
            # Evaluate the response
            evaluation = evaluate_answer(response, meta['ground_truths'])
            
            # Create result object
            if args.prompt_type.startswith("no_passage"):
                passage_result = {
                    'passage_id': 'none',
                    'title': 'N/A',  
                    'text': 'N/A',
                    'has_answer': True,  # We always consider it as "has answer" for no_passage mode
                    'response': response,
                    'evaluation': evaluation
                }
            else:
                passage_result = {
                    'passage_id': meta['passage'].get('id', 'unknown'),
                    'title': meta['passage'].get('title', ''),
                    'text': meta['passage']['text'],
                    'has_answer': meta['passage']['hasanswer'],
                    'response': response,
                    'evaluation': evaluation
                }
            
            # Add to results map
            item_id = meta['item_id']
            if item_id not in results_map:
                results_map[item_id] = {
                    'id': item_id,
                    'question': meta['question'],
                    'ground_truth': meta['ground_truths'],
                    'passages': []
                }
            
            results_map[item_id]['passages'].append(passage_result)

    # Convert results map to list
    results = list(results_map.values())
    
    # Calculate and print statistics
    print("\n=== Results Summary ===")
    print(f"Total items processed: {len(results)}")

    # Gather statistics
    total_passages = sum(len(item['passages']) for item in results)

    # If we're using no_passage modes, we don't have the concept of passages with/without answers
    if args.prompt_type.startswith("no_passage"):
        print(f"Total examples: {total_passages}")
        
        # Calculate overall accuracy
        total_correct = sum(sum(1 for p in item['passages'] if p['evaluation']['acc']) for item in results)
        print(f"Overall accuracy: {total_correct} ({total_correct/total_passages*100:.2f}%)")
        
        # Log summary metrics to wandb
        if args.use_wandb:
            wandb.log({
                "total_examples": total_passages,
                "accuracy": total_correct/total_passages if total_passages > 0 else 0,
            })
            
        # Save a simple summary
        summary = {
            'total_items': len(results),
            'total_examples': total_passages,
            'correct_answers': total_correct,
            'accuracy': total_correct/total_passages if total_passages > 0 else 0,
            'model_name': args.model_name,
            'prompt_type': args.prompt_type,
            'timestamp': time.strftime('%Y-%m-%d-%H-%M-%S'),
        }
    else:
        # Original statistics for passage-based evaluation
        passages_with_answer = sum(sum(1 for p in item['passages'] if p['has_answer']) for item in results)
        passages_without_answer = total_passages - passages_with_answer

        print(f"Total passages: {total_passages}")
        print(f"Passages with answer: {passages_with_answer} ({passages_with_answer/total_passages*100:.2f}%)")
        print(f"Passages without answer: {passages_without_answer} ({passages_without_answer/total_passages*100:.2f}%)")

        # Calculate metrics
        acc_relevant = sum(sum(1 for p in item['passages'] if p['has_answer'] and p['evaluation']['acc']) for item in results)
        
        if passages_without_answer > 0:
            acc_irrelevant = sum(sum(1 for p in item['passages'] if not p['has_answer'] and p['evaluation']['acc']) for item in results)
            print(f"Accuracy with relevant passages: {acc_relevant} ({acc_relevant/passages_with_answer*100:.2f}% of relevant)")
            print(f"Accuracy with irrelevant passages: {acc_irrelevant} ({acc_irrelevant/passages_without_answer*100:.2f}% of irrelevant)")
        else:
            acc_irrelevant = 0
            print(f"Accuracy with relevant passages: {acc_relevant} ({acc_relevant/passages_with_answer*100:.2f}% of relevant)")
            print(f"No irrelevant passages to evaluate.")

        # Log summary metrics to wandb
        if args.use_wandb:
            wandb.log({
                "total_passages": total_passages,
                "passages_with_answer": passages_with_answer,
                "passages_without_answer": passages_without_answer,
                "accuracy_relevant": acc_relevant/passages_with_answer if passages_with_answer > 0 else 0,
                "accuracy_irrelevant": acc_irrelevant/passages_without_answer if passages_without_answer > 0 else 0,
                "total_accuracy": (acc_relevant + acc_irrelevant)/total_passages if total_passages > 0 else 0,
            })
            
        # Save a detailed summary
        summary = {
            'total_items': len(results),
            'total_passages': total_passages,
            'passages_with_answer': passages_with_answer,
            'passages_without_answer': passages_without_answer,
            'acc_relevant': acc_relevant,
            'acc_irrelevant': acc_irrelevant,
            'model_name': args.model_name,
            'prompt_type': args.prompt_type,
            'timestamp': time.strftime('%Y-%m-%d-%H-%M-%S'),
        }
        
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.run_name}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nSaving results to {output_file}...")
    save_file_jsonl(results, output_file)

    # Create visualizations
    if args.create_plots:
        print("Creating visualizations...")
        create_visualizations(results, os.path.dirname(output_file), args, run_name=args.run_name)

    summary_file = os.path.splitext(output_file)[0] + '_summary.jsonl'
    save_file_jsonl([summary], summary_file)

    # Log summary file to wandb if enabled
    if args.use_wandb:
        wandb.save(summary_file)

    print(f"Saved summary to {summary_file}")
    print("\nDone!")

    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default=f"output/{os.path.basename(__file__)[: -len('.py')]}", help="Directory to save output files")
    parser.add_argument("--run_name", type=str, default="rag_evaluation", help="Run name for Weights & Biases")
    parser.add_argument("--model_name", type=str, default="gpt2-small", help="Hugging Face model name")
    parser.add_argument("--prompt_type", type=str, default="with_passage", 
                        choices=["with_passage", "no_passage", "no_passage_no_refuse"],
                        help="Type of prompt to use")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--max_items", type=int, default=-1, help="Maximum number of items to process (-1 for all)")
    parser.add_argument("--max_passages_per_item", type=int, default=-1, help="Maximum number of passages per item (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--log_every", type=int, default=10, help="Log progress every N batches")
    parser.add_argument("--create_plots", action="store_true", help="Create visualizations")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing passages")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision instead of bfloat16")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    main(args)