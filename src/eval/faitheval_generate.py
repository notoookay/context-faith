import argparse
import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from datasets import load_dataset
import wandb
from transformer_lens import HookedTransformer

import sys
sys.path.append("..")
from exp_hallucination.src.utils import load_jsonlines, save_file_jsonl, format_prompt, MODEL_ALIAS, LLAMA_3_1_INSTRUCT_PROMPT_FORMAT, QWEN_2_5_INSTRUCT_PROMPT_FORMAT
from exp_hallucination.src.eval.utils import normalize_answer, evaluate_faithfulness

class FaithEvalDataset(Dataset):
    def __init__(self, data, tokenizer=None, n_samples=1, task_specific_prompt="", dataset_name="", model_name="", task_type="counterfactual"):
        self.tasks = []
        self.n_samples = n_samples
        self.task_specific_prompt = task_specific_prompt
        self.dataset_name = dataset_name
        
        # Process all items and create tasks
        for item_idx, item in enumerate(data):
            question = item['question']
            context = item['context']
            answers = item['answers'] if 'answers' in item else item['answer'] # Arc uses 'answer'
            item_id = item.get('id') or str(item_idx)

            if task_type == "counterfactual":
                choices = item["choices"]
                answer_key = item["answerKey"]
                if answer_key == "1":
                    answer_key = "A"
                if answer_key == "2":
                    answer_key = "B"
                if answer_key == "3":
                    answer_key = "C"
                if answer_key == "4":
                    answer_key = "D"
                answer_labels = {}
                choices["label"] = ["A", "B", "C", "D"]
                if len(choices["text"]) < 4:
                    continue
                for text, choice in zip(choices["text"], choices["label"]):
                    answer_labels[choice] = text
                input_question = "{0}\nA: {1}\nB: {2}\nC: {3}\nD: {4}".format(question, answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
                native_prompt = f"""Please answer the following question:
{task_specific_prompt}
Context: {context}
Question: {input_question}
Answer:"""
            else:
                native_prompt = f"""You are an expert in retrieval question answering. 
Please respond with the exact answer only. Do not be verbose or provide extra information.
{task_specific_prompt}
Context: {context}
Question: {question}
Answer:"""
            
            if "llama" in model_name.lower():
                prompt = LLAMA_3_1_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
            elif "qwen" in model_name.lower():
                prompt = QWEN_2_5_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
            else:
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
                    'context': context,
                    'answers': answers,
                    'native_prompt': native_prompt,
                    'prompt': prompt,
                })
                if task_type == "counterfactual":
                    self.tasks[-1]['answer_key'] = answer_key
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]


def collate_fn(batch, tokenizer, model_name, task_type):
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
        'context': item['context'],
        'answers': item['answers'] if 'answers' in item else item['answer'],
        'prompt_length': len(encoded_inputs['input_ids'][i])
    } for i, item in enumerate(batch)]

    if task_type == "counterfactual":
        for i, item in enumerate(batch):
            metadata[i]['answer_key'] = item['answer_key']
    
    return {
        'input_ids': encoded_inputs['input_ids'],
        'attention_mask': encoded_inputs['attention_mask'],
        'metadata': metadata
    }


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
    print(f"Loading model {model_name}...")
    
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
    
    # Initialize wandb if requested
    args.run_name = f"{args.model_name.replace('/', '--')}__{args.task_type}_{args.n_samples}_{args.seed}"
    if args.use_wandb:
        wandb.init(
            project="faitheval-generation", 
            name=args.run_name,
            config=vars(args)
        )

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
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating responses")):
        # Move input tensors to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata']
        
        with torch.no_grad():
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
            response = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
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
                'response': response,
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
    
    print(f"\n=== Evaluation Results for {args.task_type} ===")
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct_count}/{total_count})")
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count
        })
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
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
        'top_p': args.top_p if args.do_sample else None,
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
    parser = argparse.ArgumentParser(description="Evaluate model on FaithEval benchmark")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--task_type", type=str, required=True, 
                        choices=["unanswerable", "inconsistent", "counterfactual"],
                        help="Type of FaithEval task to evaluate")
    parser.add_argument("--output_dir", type=str, default="output/eval/faitheval", 
                        help="Directory to save output files")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--max_items", type=int, default=-1, help="Maximum number of items to process (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing passages")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision instead of bfloat16")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strict_match", action="store_true", help="Use strict phrase matching for evaluation")
    
    args = parser.parse_args()
    
    main(args) 