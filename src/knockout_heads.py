import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer, ActivationCache
from collections import defaultdict
from datasets import load_dataset
import sys
import argparse
import json
from transformer_lens import utils
from functools import partial

sys.path.append("..")

from utils import (
    format_prompt,
    LLAMA_3_1_INSTRUCT_PROMPT_FORMAT,
    QWEN_2_5_INSTRUCT_PROMPT_FORMAT,
    MODEL_ALIAS,
)

def load_model(model_name: str):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype='bfloat16')
    model.eval()

    return model


def generate_with_model(model, input_tokens, hooks=None, max_new_tokens=50):
    """Generate text with the model, optionally applying hooks"""
    # Use greedy decoding
    
    if hooks:
        with model.hooks(hooks):
            output = model.generate(
                input_tokens, 
                max_new_tokens=max_new_tokens,
                stop_at_eos=True,
                do_sample=False,
                verbose=False,
            )
    else:
        output = model.generate(
            input_tokens, 
            max_new_tokens=max_new_tokens,
            stop_at_eos=True,
            do_sample=False,
            verbose=False,
        )
    
    output = output[:, input_tokens.size(1):]  # Remove input tokens
    return output


def is_answer_correct(generated_text, correct_answers):
    """Check if the generated text contains any of the correct answers"""
    for answer in correct_answers:
        if answer.lower() in generated_text.lower():
            return True
    return False


def get_model_input(item, model: HookedTransformer):
    """
    Get the full text of the passage and question from the data
    """
    ctxs = item["ctxs"]
    question = item["question"]
    prompts = []
    for ctx in ctxs:
        # Create prompt
        native_prompt = format_prompt(question, [ctx], prompt_type="with_passage")
        if "llama-3.1" in model.cfg.model_name:
            prompt = LLAMA_3_1_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
        elif "qwen" in model.cfg.model_name:
            prompt = QWEN_2_5_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
        else:
            prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": native_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
            )
        prompts.append(prompt)
    
    return {'model_input': prompts}


def knockout_heads(
    model: HookedTransformer,
    examples: list,
    copying_heads: list,
    threshold: float = 0.3,
    batch_size: int = 4
):
    """
    Perform knockout analysis on identified copying heads.
    
    Args:
        model: The transformer_lens model
        examples: List of examples with formatted text and answers
        copying_heads: List of (layer, head) tuples to knockout
        threshold: Score threshold for identifying successful copying
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with results of knockout experiment
    """
    # Create hook functions for attention head knockout
    def knockout_hook_fn(value, hook, head):
        """Zero out specific attention heads"""
        # if value.shape[1] == 1:
        #     value[:, -1, head, :] = 0
        #     return value
        # value shape: [batch_size, seq_len, n_heads, d_head]
        value[:, -1, head, :] = 0
        return value
        
    # Process examples in batches
    total_examples = len(examples)
    results = {
        "original_correct": 0,
        "knockout_correct": 0,
        "example_results": []
    }
    
    for i in tqdm(range(0, total_examples, batch_size)):
        batch = examples[i:min(i+batch_size, total_examples)]
        
        # Get original outputs
        original_outputs = []
        knockout_outputs = []
        
        for example in batch:
            # Format input for generation
            # formatted_text = example["formatted_text"]
            model_input = example["model_input"][0]
            
            prompt_tokens = model.to_tokens(model_input, prepend_bos=False, padding_side='left')
            
            # Original generation (without knockout)
            with torch.no_grad():
                original_output = generate_with_model(
                    model, 
                    prompt_tokens, 
                    max_new_tokens=100  # Adjust based on expected output length
                )
            
            # Knockout generation (with copying heads disabled)
            with torch.no_grad():
                knockout_output = generate_with_model(
                    model, 
                    prompt_tokens, 
                    # hooks=[("blocks.*.attn.hook_pattern", knockout_hook_fn)],
                    hooks=[
                        (utils.get_act_name("z", layer), partial(knockout_hook_fn, head=head)) for layer, head in copying_heads
                    ],
                    max_new_tokens=100
                )
            
            # Convert outputs to text
            original_text = model.tokenizer.decode(original_output[0])
            knockout_text = model.tokenizer.decode(knockout_output[0])
            
            # Evaluate correctness
            original_correct = is_answer_correct(original_text, example["answers"])
            knockout_correct = is_answer_correct(knockout_text, example["answers"])
            
            results["original_correct"] += int(original_correct)
            results["knockout_correct"] += int(knockout_correct)
            
            # Store individual example results
            example_result = {
                # "id": example.get("id", f"example_{i}"),
                "question": example.get("question", ""),
                "passage": example.get("passage", {"text": ""}).get("text", ""),
                "answers": example["answers"],
                "original_output": original_text,
                "knockout_output": knockout_text,
                "original_correct": original_correct,
                "knockout_correct": knockout_correct,
                "changed": original_correct != knockout_correct
            }
            
            results["example_results"].append(example_result)
    
    # Calculate aggregate statistics
    results["total_examples"] = total_examples
    results["original_accuracy"] = results["original_correct"] / total_examples
    results["knockout_accuracy"] = results["knockout_correct"] / total_examples
    results["accuracy_change"] = results["original_accuracy"] - results["knockout_accuracy"]
    results["changed_count"] = sum(1 for res in results["example_results"] if res["changed"])
    results["changed_percentage"] = results["changed_count"] / total_examples * 100
    
    return results


def main():
    """Main function to run knockout heads analysis."""
    parser = argparse.ArgumentParser(description='Run knockout heads analysis on attention heads')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-2b-it',
                        help='Model name to load (default: google/gemma-2-2b-it)')
    parser.add_argument('--data_file', type=str, 
                        default='../data/exp-hallu/output/model_generation/google--gemma-2-9b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_1_generation_filtered.jsonl',
                        help='Path to the dataset file')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for processing')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples to evaluate (default: all)')
    parser.add_argument('--output_file', type=str, default='knockout_results.json',
                        help='Output file for results (default: knockout_results.json)')
    parser.add_argument('--heads', type=str, nargs='*',
                        help='Specific heads to knockout in format "layer,head" (e.g., "0,5" "1,10").')
    parser.add_argument('--layers', type=int, nargs='*',
                        help='Layers to knockout heads from (e.g., 0 1 2). Must be used with --head_indices.')
    parser.add_argument('--head_indices', type=int, nargs='*',
                        help='Head indices to knockout within specified layers (e.g., 5 10 15). Must be used with --layers.')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    model = load_model(MODEL_ALIAS[args.model_name])
    
    print(f"Loading dataset from: {args.data_file}")
    dataset = load_dataset(
        "json",
        data_files=args.data_file,
        split="train",
    )
    
    print("Processing dataset with model input formatting...")
    dataset = dataset.map(
        lambda x: get_model_input(x, model),
        desc="Formatting inputs"
    )
    
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
        print(f"Limited to {len(dataset)} examples")
    
    print(f"Total examples to evaluate: {len(dataset)}")
    
    # Determine which heads to knockout
    if args.heads:
        # Parse user-specified heads from command line arguments
        copying_heads = []
        for head_str in args.heads:
            try:
                layer, head = map(int, head_str.split(','))
                copying_heads.append((layer, head))
            except ValueError:
                raise ValueError(f"Invalid head format '{head_str}'. Expected format: 'layer,head' (e.g., '0,5')")
        print(f"Using user-specified heads: {copying_heads}")
    elif args.layers and args.head_indices:
        # Create all combinations of specified layers and head indices
        if len(args.layers) == 0 or len(args.head_indices) == 0:
            raise ValueError("Both --layers and --head_indices must be non-empty when used together")
        
        copying_heads = []
        for layer in args.layers:
            for head_idx in args.head_indices:
                copying_heads.append((layer, head_idx))
        print(f"Using layers {args.layers} with head indices {args.head_indices}: {copying_heads}")
    elif args.layers or args.head_indices:
        # Error if only one of layers/head_indices is provided
        raise ValueError("--layers and --head_indices must be used together")
    else:
        # No heads specified - require user to specify heads
        raise ValueError(
            "No heads specified for knockout. Please use one of the following options:\n"
            "  --heads: Specify exact (layer,head) pairs (e.g., --heads '0,5' '1,10' '2,15')\n"
            "  --layers and --head_indices: Specify layers and head indices separately (e.g., --layers 0 1 2 --head_indices 5 10 15)"
        )
    
    print(f"Running knockout analysis with batch size: {args.batch_size}")
    
    # Convert dataset to list for processing
    examples = list(dataset)
    
    # Run knockout analysis on the entire dataset
    knockout_results = knockout_heads(
        model, 
        examples, 
        copying_heads, 
        batch_size=args.batch_size
    )
    
    # Print summary results
    print("\n" + "="*50)
    print("KNOCKOUT ANALYSIS RESULTS")
    print("="*50)
    print(f"Total examples: {knockout_results['total_examples']}")
    print(f"Original accuracy: {knockout_results['original_accuracy']:.3f}")
    print(f"Knockout accuracy: {knockout_results['knockout_accuracy']:.3f}")
    print(f"Accuracy change: {knockout_results['accuracy_change']:.3f}")
    print(f"Examples with changed predictions: {knockout_results['changed_count']} ({knockout_results['changed_percentage']:.1f}%)")
    
    # Save detailed results
    print(f"\nSaving detailed results to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(knockout_results, f, indent=2, ensure_ascii=False)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
