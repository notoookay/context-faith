import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch as t
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm
from collections import defaultdict

from utils import format_prompt, MODEL_ALIAS, QWEN_2_5_INSTRUCT_PROMPT_FORMAT, LLAMA_3_1_INSTRUCT_PROMPT_FORMAT
from diff_in_mean import apply_hook_to_model


def load_model(model_name: str):
    """Load the specified transformer model."""
    device = (
        "mps"
        if t.backends.mps.is_available()
        else "cuda" if t.cuda.is_available() else "cpu"
    )
    if model_name not in MODEL_ALIAS:
        raise ValueError(f"Model {model_name} not found in MODEL_ALIAS")
    model_name = MODEL_ALIAS[model_name]
    print(f"Loading model {model_name} on device: {device}")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype='bfloat16')
    model.eval()
    return model


def get_model_input(item, tokenizer, model_name: str, prompt_type: str = "with_passage"):
    """Get the full text of the passage and question from the data."""
    passage = item["ctxs"][0]
    question = item["question"]
    prompt = format_prompt(question, [passage], prompt_type)

    if "qwen" in model_name.lower():
        model_input = QWEN_2_5_INSTRUCT_PROMPT_FORMAT.format(prompt=prompt)
    elif "llama" in model_name.lower():
        model_input = LLAMA_3_1_INSTRUCT_PROMPT_FORMAT.format(prompt=prompt)
    else:
        model_input = tokenizer.apply_chat_template(
            [
                {'role': 'user', 'content': prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    return {'model_input': model_input}


def map_char_to_token_positions(text, answer, tokenizer):
    """
    Maps character-level positions to token-level positions more robustly.
    
    Args:
        text: Source text
        answer: Answer to locate
        tokenizer: The model's tokenizer
    
    Returns:
        List of token indices that cover the answer
    """
    # Find character positions of all answer occurrences
    answer_positions = []
    start_pos = 0
    
    while True:
        start_idx = text.lower().find(answer.lower(), start_pos)
        if start_idx == -1:
            break
        end_idx = start_idx + len(answer)
        answer_positions.append((start_idx, end_idx))
        start_pos = start_idx + 1
    
    if not answer_positions:
        print(f"Answer '{answer}' not found in text!")
        return []
    
    # Get token-to-char mapping from tokenizer
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_spans = encoding.offset_mapping
    
    # Find tokens that overlap with answer spans
    token_positions = []
    for start_char, end_char in answer_positions:
        tokens_for_this_span = []
        for i, (token_start, token_end) in enumerate(token_spans):
            # Check if this token overlaps with the answer span
            if token_end > start_char and token_start < end_char:
                tokens_for_this_span.append(i)
        
        if tokens_for_this_span:
            token_positions.append((tokens_for_this_span[0], tokens_for_this_span[-1] + 1))
    
    return token_positions


def attn_detector(cache, token_position, num_layers, num_heads, top_k=10):
    """
    Detect the attention heads that attend most to the answer token.
    
    Returns:
        attn_heads: List of attention head names that focus on the token
        attn_scores: List of attention scores for those heads
    """
    attn_heads = []
    attn_scores = []
    
    for layer in range(num_layers):
        for head in range(num_heads):
            attention_pattern = cache['pattern', layer][head]
            scores = attention_pattern[-1, :].cpu()
            
            top_k_pos = np.argsort(scores)[-top_k:]
            if token_position in top_k_pos:
                attn_heads.append(f"L{layer}H{head}")
                attn_scores.append(scores[token_position])
    
    return attn_heads, attn_scores


def find_answer(response, answers):
    """Find which answer appears in the response."""
    for answer in answers:
        if answer.lower() in response.lower():
            return answer
    return None


def create_attention_heatmap(extracted_info_heads, num_layers, num_heads, total_examples, output_path=None):
    """
    Create a heatmap showing the ratio of attention head focus on answer tokens using discrete classes.
    
    Args:
        extracted_info_heads: Dictionary with head names as keys and counts as values
        num_layers: Number of layers in the model
        num_heads: Number of heads per layer
        total_examples: Total number of examples processed
        output_path: Path to save the heatmap (optional)
    """
    # Create a 2D array for the heatmap (heads x layers for transposed view)
    ratio_data = np.zeros((num_heads, num_layers))
    
    # Fill the heatmap with ratios
    for head_name, count in extracted_info_heads.items():
        # Parse layer and head numbers from format "L{layer}H{head}"
        layer_str, head_str = head_name.split('H')
        layer = int(layer_str[1:])  # Remove 'L' and convert to int
        head = int(head_str)
        
        ratio = count / total_examples if total_examples > 0 else 0
        ratio_data[head, layer] = ratio
    
    # Define ratio classes and their corresponding values
    def classify_ratio(ratio):
        if ratio < 0.1:
            return 0  # 0~0.1
        elif ratio < 0.3:
            return 1  # 0.1~0.3
        elif ratio < 0.5:
            return 2  # 0.3~0.5
        else:
            return 3  # 0.5~1.0
    
    # Convert ratios to classes
    heatmap_data = np.zeros_like(ratio_data)
    for i in range(num_heads):
        for j in range(num_layers):
            heatmap_data[i, j] = classify_ratio(ratio_data[i, j])
    
    # Define class labels and colors
    class_labels = ['0~0.1', '0.1~0.3', '0.3~0.5', '0.5~1.0']
    colors = ['#440154', '#31688e', '#35b779', '#fde725']  # Viridis-like discrete colors
    
    # Create custom discrete colormap
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(colors)
    
    # Create the heatmap
    plt.figure(figsize=(max(12, num_layers), max(8, num_heads * 0.5)))
    im = plt.imshow(
        heatmap_data,
        aspect='auto',
        cmap=custom_cmap,
        vmin=0,
        vmax=3
    )
    
    # Set ticks and labels
    plt.xticks(range(num_layers), [f'L{i}' for i in range(num_layers)])
    plt.yticks(range(num_heads), [f'H{i}' for i in range(num_heads)])
    
    # Create custom colorbar with class labels
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(class_labels)
    cbar.set_label('Ratio Classes of Focus on Answer Tokens')
    
    plt.title('Attention Head Focus on Answer Tokens\n(Classified ratio of examples where head focuses on answer token)')
    plt.xlabel('Layer')
    plt.ylabel('Head')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
    else:
        plt.show()
    
    return ratio_data  # Return original ratio data for analysis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the JSONL data file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name to analyze')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples to process')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-k tokens to consider for attention')
    parser.add_argument('--output_dir', type=str, default='output/attention_heads_analysis',
                        help='Output directory for results')
    parser.add_argument('--save_heatmap', action='store_true',
                        help='Save heatmap to file instead of displaying')
    
    # Direction-related arguments
    parser.add_argument('--direction_file', type=str, default=None,
                        help='Path to direction file (.pt) to apply during analysis')
    parser.add_argument('--coefficient', type=float, default=1.0,
                        help='Coefficient to scale the direction by')
    parser.add_argument('--ablate', action='store_true',
                        help='Ablate the direction instead of adding it')
    parser.add_argument('--layer', type=str, default=None,
                        help='Comma-separated list of layers to apply direction to')
    parser.add_argument('--hook_name', type=str, default='resid_post',
                        help='Name of the hook point to apply the direction to')
    parser.add_argument('--pos_to_apply', type=int, default=-1,
                        help='Position to apply direction at')
    
    return parser.parse_args()


def main(args):
    # Create output directory if it doesn't exist
    args.output_dir = os.path.join(args.output_dir, args.model_name, args.data_file.split('/')[-1].split('.')[0])
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from: {args.data_file}")
    data_rel = load_dataset('json', data_files=args.data_file, split='train')
    
    if args.max_examples:
        data_rel = data_rel.select(range(min(args.max_examples, len(data_rel))))
    
    print(f"Processing {len(data_rel)} examples")
    
    # Load model
    model = load_model(args.model_name)
    
    # Prepare data with model inputs
    print("Preparing model inputs...")
    data_rel = data_rel.map(get_model_input, fn_kwargs={'tokenizer': model.tokenizer, 'model_name': args.model_name})
    
    # Get model configuration
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    print(f"Model has {num_layers} layers and {num_heads} heads per layer")
    
    # Load direction and setup hooks if specified
    direction = None
    hooks_dict = {}
    if args.direction_file:
        print(f"Loading direction from {args.direction_file}...")
        direction = t.load(args.direction_file, map_location=model.cfg.device)
        print(f"Direction shape: {direction.shape}")
        
        # Parse layers if specified
        if args.layer is not None:
            layers = [int(layer) for layer in args.layer.split(',')]
        else:
            layers = None
            
        # Create hooks
        hooks_dict = apply_hook_to_model(
            model=model,
            direction=direction,
            layer=layers,
            coeff=args.coefficient,
            ablate=args.ablate,
            position=args.pos_to_apply,
            hook_name=args.hook_name
        )
        
        direction_info = f"_{'ablate' if args.ablate else 'add'}_coef{args.coefficient}_pos{args.pos_to_apply}"
        if layers:
            direction_info += f"_layer{'_'.join(str(l) for l in layers)}"
        print(f"Direction will be applied with: {direction_info}")
    
    # Process examples and collect attention head statistics
    extracted_info_heads = defaultdict(int)
    processed_examples = 0
    
    print("Processing examples...")
    for i in tqdm(range(len(data_rel))):
        
        example = data_rel[i]
        
        # Tokenize input
        example_tokens = model.to_tokens(example['model_input'], prepend_bos=False)
        
        # Run model with cache to get attention patterns (with optional hooks)
        if hooks_dict:
            # Apply hooks when getting cache
            with model.hooks(fwd_hooks=[(name, hook_fn) for name, hook_fn in hooks_dict.items()]):
                example_logits, example_cache = model.run_with_cache(example_tokens, remove_batch_dim=True)
        else:
            # Run without hooks
            example_logits, example_cache = model.run_with_cache(example_tokens, remove_batch_dim=True)
        example_cache.to('cpu') # in-place operation
        
        # Find answer in the response
        answer = find_answer(example['model_input'], example['answers'])
        if answer is None:
            print(f"No answer found for example {i}")
            continue
        
        # Map answer to token positions
        token_positions = map_char_to_token_positions(example['model_input'], answer, model.tokenizer)
        if not token_positions:
            continue
        
        # Use the last token position of the answer
        token_pos = token_positions[0][-1] - 1
        
        # Detect attention heads focusing on this token
        attn_heads, attn_scores = attn_detector(example_cache, token_pos, num_layers, num_heads, args.top_k)
        
        # Update counts
        for head in attn_heads:
            extracted_info_heads[head] += 1
        
        processed_examples += 1
    
    print(f"\nProcessed {processed_examples} examples successfully")
    
    # Print top attention heads
    sorted_heads = sorted(extracted_info_heads.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 20 attention heads by focus frequency:")
    for head, count in sorted_heads[:20]:
        ratio = count / processed_examples
        print(f"{head}: {count}/{processed_examples} ({ratio:.3f})")
    
    # Create heatmap
    output_path = None
    if args.save_heatmap:
        filename = 'attention_heads_heatmap'
        if args.direction_file:
            filename += direction_info
        filename += '.png'
        output_path = os.path.join(args.output_dir, filename)
    
    heatmap_data = create_attention_heatmap(
        extracted_info_heads, 
        num_layers, 
        num_heads, 
        processed_examples,
        output_path
    )
    
    # Save detailed results as JSON
    results_filename = 'attention_analysis_results'
    if args.direction_file:
        results_filename += direction_info
    results_filename += '.json'
    results_file = os.path.join(args.output_dir, results_filename)
    
    # Prepare data for JSON export
    results_data = {
        'metadata': {
            'model': args.model_name,
            'dataset': args.data_file,
            'processed_examples': processed_examples,
            'top_k': args.top_k,
            'num_layers': num_layers,
            'num_heads': num_heads
        },
        'extracted_info_heads': dict(extracted_info_heads),
        'sorted_heads': []
    }
    
    # Add direction information if applicable
    if args.direction_file:
        results_data['metadata']['direction_config'] = {
            'direction_file': args.direction_file,
            'coefficient': args.coefficient,
            'ablate': args.ablate,
            'layers': args.layer,
            'position': args.pos_to_apply,
            'hook_name': args.hook_name
        }
    
    # Add sorted results with ratios
    for head, count in sorted_heads:
        ratio = count / processed_examples
        results_data['sorted_heads'].append({
            'head': head,
            'count': count,
            'ratio': ratio
        })
    
    # Save to JSON file
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == '__main__':
    args = parse_args()
    main(args) 