import os
import argparse
import torch
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments for direction extraction."""
    parser = argparse.ArgumentParser(description="Extract diff-in-mean directions between two groups of pre-computed activations")
    
    parser.add_argument("--load_activations_group1", type=str, required=True,
                        help="Path to load pre-computed activations for group 1.")
    parser.add_argument("--load_activations_group2", type=str, required=True,
                        help="Path to load pre-computed activations for group 2.")
    parser.add_argument("--group1_name", type=str, required=True,
                        help="Name of group 1.")
    parser.add_argument("--group2_name", type=str, required=True,
                        help="Name of group 2.")
    parser.add_argument("--output_dir", type=str, default="./output/directions",
                        help="Directory to save the extracted directions")
    parser.add_argument("--extract_pos", type=int, default=-1, 
                        help="Position to extract the directions from (-1 means last token)")
    parser.add_argument("--visualize", action="store_true",
                        help="Whether to visualize the direction norms across layers")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layers to extract activations from (e.g., '10,15'). If None, all layers from the activation files are used.")
    
    return parser.parse_args()


def load_model_activations(load_path: str) -> Dict:
    """Load model activations from disk."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Activation file not found at {load_path}")
    
    activations = torch.load(load_path)
    print(f"Loaded activations from {load_path}")
    return activations


def compute_diff_in_mean_directions(activations_group1: Dict, activations_group2: Dict, layers: List[int], positions: List[int]) -> torch.Tensor:
    """
    Compute diff-in-mean directions from two groups of activations.
    
    Args:
        activations_group1: Dictionary of activations for group 1 {layer: {pos: {model_input_idx: Tensors}}}
        activations_group2: Dictionary of activations for group 2 {layer: {pos: {model_input_idx: Tensors}}}
        layers: List of layers to compute directions for
        positions: List of positions to compute directions for (should contain only one position)
    
    Returns:
        Tensor of shape (n_layers, d_model) containing the directions
    """
    print("Computing diff-in-mean directions...")
    
    # Since we only extract one position, use the first (and only) position
    pos = positions[0]
    directions_list = []
    d_model = activations_group1[layers[0]][pos].shape[-1]
    
    for layer in layers:
        # Check if activations exist for both groups
        if (layer not in activations_group1 or pos not in activations_group1[layer] or
            layer not in activations_group2 or pos not in activations_group2[layer]):
            print(f"Warning: No activations found for layer {layer}, position {pos}")
            
            direction = torch.zeros(d_model)
            directions_list.append(direction)
            continue
        
        # Collect from group 1
        group1_acts_list = activations_group1[layer][pos]
        
        # Collect from group 2  
        group2_acts_list = activations_group2[layer][pos]
        
        if len(group1_acts_list) == 0 or len(group2_acts_list) == 0:
            print(f"Warning: Empty activation group for layer {layer}, position {pos}")
            direction = torch.zeros(d_model)
            directions_list.append(direction)
            continue
        
        # Compute means for each group
        group1_mean = group1_acts_list.mean(dim=0)
        group2_mean = group2_acts_list.mean(dim=0)
        
        # Compute difference (group1 - group2)
        # This will suppress group2 characteristics and enhance group1 characteristics
        direction = group1_mean - group2_mean
        directions_list.append(direction)
    
    directions = torch.stack(directions_list)  # Shape: (n_layers, d_model)
    return directions

def visualize_directions(directions: torch.Tensor, title: str, output_path: str):
    """
    Visualize direction norms across layers.
    
    Args:
        directions: Tensor of shape (n_layers, d_model)
        title: Title for the plot
        output_path: Path to save the visualization
    """
    
    # Compute norms for each layer
    norms = torch.norm(directions, dim=-1)  # Shape: (n_layers,)
    
    plt.figure(figsize=(10, 6))
    
    n_layers = norms.shape[0]
    
    plt.plot(range(n_layers), norms.float().numpy())
    plt.xlabel('Layer')
    plt.ylabel('Direction Norm')
    plt.title(title)
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    args = parse_args()
    
    # Create output directory based on activation file names
    group1_name = args.group1_name
    group2_name = args.group2_name
    args.output_dir = os.path.join(args.output_dir, f"{group1_name}__{group2_name}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load activations for both groups
    print(f"Loading Group 1 activations from {args.load_activations_group1}")
    activations_group1 = load_model_activations(args.load_activations_group1)
    
    print(f"Loading Group 2 activations from {args.load_activations_group2}")
    activations_group2 = load_model_activations(args.load_activations_group2)
    
    # Determine layers to analyze
    if args.layers:
        try:
            layers = [int(l.strip()) for l in args.layers.split(',')]
        except ValueError:
            print(f"Error: Invalid layer specification '{args.layers}'. Please use comma-separated integers.")
            return
    else:
        # Use all layers from the activation files
        layers = list(activations_group1.keys())
    
    # Filter layers to only include those present in both activation files
    layers = [layer for layer in layers if layer in activations_group1 and layer in activations_group2]
    print(f"Target layers: {layers}")
    
    if not layers:
        print("Error: No common layers found between the two activation files.")
        return
    
    # Determine positions to analyze
    positions = [args.extract_pos]
    
    # Verify that the position exists in both activation files
    for layer in layers:
        if args.extract_pos not in activations_group1[layer] or args.extract_pos not in activations_group2[layer]:
            print(f"Warning: Position {args.extract_pos} not found in layer {layer} for one or both groups.")
    
    # Compute diff-in-mean directions
    print(f"Computing diff-in-mean directions (Group 1 - Group 2)...")
    directions = compute_diff_in_mean_directions(activations_group1, activations_group2, layers, positions)
    
    # Save directions
    directions_path = os.path.join(args.output_dir, f'diff_in_mean_{args.extract_pos}.pt')
    torch.save(directions, directions_path)
    
    print(f"Directions shape: {directions.shape}")
    print(f"Directions saved to {directions_path}")    
    
    # Visualize direction norms if requested
    if args.visualize:
        vis_path = os.path.join(args.output_dir, f"direction_norms_{args.extract_pos}.png")
        print(f"Visualizing direction norms to {vis_path}")
        visualize_directions(
            directions=directions,
            title=f"Direction Norms",
            output_path=vis_path
        )
    
if __name__ == "__main__":
    main()