import torch
import os
from typing import List, Optional, Tuple, Dict, Union, Callable
from tqdm import tqdm
import functools
from torch import Tensor
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from einops import einsum, rearrange

def get_mean_activations(
    model: HookedTransformer, 
    instructions: List[str], 
    batch_size: int = 32, 
    positions: List[int] = [-1],
    hook_name: str = 'resid_post'
) -> torch.Tensor:
    """
    Get mean activations for a list of instructions across specified positions and layers.
    
    Parameters:
    - model: The transformer_lens model
    - instructions: List of instruction strings
    - batch_size: Batch size for processing
    - positions: List of positions to extract activations from
    - hook_name: Name of the hook point to extract activations from
    
    Returns:
    - Mean activations tensor of shape (n_positions, n_layers, d_model)
    """
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.cfg.n_layers  # Use transformer_lens config
    n_samples = len(instructions)
    d_model = model.cfg.d_model   # Use transformer_lens config

    # Store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64)

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i:i+batch_size]
        batch_size_actual = len(batch_instructions)
        
        for j, instruction in enumerate(batch_instructions):
            tokens = model.to_tokens(instruction)
            
            _, cache = model.run_with_cache(tokens)
            
            for pos_idx, pos in enumerate(positions):
                for layer in range(n_layers):
                    activation = cache[hook_name, layer][0].cpu()  # [0] to remove batch dimension
                    
                    actual_pos = pos if pos >= 0 else activation.shape[0] + pos
                    
                    mean_activations[pos_idx, layer] += (1.0 / n_samples) * activation[actual_pos].to(torch.float64)

    return mean_activations

def get_diff_in_mean(
    model: HookedTransformer,
    group1_instructions: List[str],
    group2_instructions: List[str],
    batch_size: int = 32,
    positions: List[int] = [-1],
    hook_name: str = 'resid_post'
) -> torch.Tensor:
    """
    Calculate the difference in mean activations between two groups of instructions.
    
    Parameters:
    - model: The transformer_lens model
    - group1_instructions: First group of instructions 
    - group2_instructions: Second group of instructions
    - batch_size: Batch size for processing
    - positions: List of positions to extract activations from
    - hook_name: Name of the hook point to extract activations from
    
    Returns:
    - Difference in mean activations tensor of shape (n_positions, n_layers, d_model)
    """
    mean_activations_group1 = get_mean_activations(
        model, 
        group1_instructions, 
        batch_size=batch_size, 
        positions=positions,
        hook_name=hook_name
    )
    
    mean_activations_group2 = get_mean_activations(
        model, 
        group2_instructions, 
        batch_size=batch_size, 
        positions=positions,
        hook_name=hook_name
    )

    diff_in_mean = mean_activations_group1 - mean_activations_group2

    return diff_in_mean

def generate_directions(
    model: HookedTransformer,
    group1_instructions: List[str],
    group2_instructions: List[str],
    positions: List[int],
    batch_size: int = 32,
    hook_name: str = 'resid_post'
) -> torch.Tensor:
    """
    Generate directions using the diff-in-mean.
    
    Parameters:
    - model: The transformer_lens model
    - group1_instructions: First group of instructions
    - group2_instructions: Second group of instructions
    - positions: List of positions to extract activations from
    - batch_size: Batch size for processing
    - hook_name: Name of the hook point to extract activations from
    
    Returns:
    - Difference in mean activations tensor of shape (n_positions, n_layers, d_model)
    """

    diff_in_mean = get_diff_in_mean(
        model, 
        group1_instructions, 
        group2_instructions, 
        batch_size=batch_size, 
        positions=positions,
        hook_name=hook_name
    )

    # Validate dimensions and check for NaN values
    assert diff_in_mean.shape == (len(positions), model.cfg.n_layers, model.cfg.d_model)
    assert not diff_in_mean.isnan().any()

    return diff_in_mean

def activation_addition_hook(
    activation: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    vector: Float[Tensor, "layer d_model"],
    coeff: float,
    position: int = -1,
) -> Tensor:
    """
    Hook function that adds a scaled vector to activations.
    
    Parameters:
    - activation: The activation tensor
    - hook: The hook point
    - vector: Direction vector to add
    - coeff: Scaling coefficient
    - position: Optional specific position to apply to (if None, applies to all positions)
    Returns:
    - Modified activation tensor
    """
    
    # vector = vector[:, hook.layer()].squeeze(0) # [d_model]
    vector = vector[hook.layer()] # [d_model]

    # # Don't need to normalize
    # vector = vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)
    vector = vector.to(activation.device)

    # proj = activation[:, position] @ vector # [batch,]
    # coeff = 2 * proj.abs()

    if activation.shape[1] == 1:
        # proj = activation[:, 0] @ vector # [batch,]
        # coeff = 2 * proj.abs()
        # activation[:, 0] += einsum(coeff, vector, 'batch, d_model -> batch d_model') # Just erase the component along the direction
        activation[:, 0] += coeff * vector
        return activation
    
    # Convert negative indices
    # pos = position if position >= 0 else activation.shape[1] + position
    # Only modify the specific position
    # activation[:, position] += einsum(coeff, vector, 'batch, d_model -> batch d_model')
    activation[:, position] += coeff * vector
    
    return activation

def direction_ablation_hook(
    activation: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    direction: Float[Tensor, "layer d_model"],
    position: int = -1,
) -> Tensor:
    """
    Hook function that ablates (removes) a direction from activations.
    
    Parameters:
    - activation: The activation tensor
    - hook: The hook point
    - direction: Direction vector to ablate
    - position: Optional specific position to apply to (if None, applies to all positions)
    Returns:
    - Modified activation tensor
    """
    
    # direction = direction[:, hook.layer()].squeeze(0) # [d_model]
    direction = direction[hook.layer()] # [d_model]

    # Normalize direction
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    direction = direction.to(activation.device)
    
    if activation.shape[1] == 1:
        activation[:, 0] -= (activation[:, 0].to(direction.dtype) @ direction).unsqueeze(-1) * direction
        return activation
    
    # Convert negative indices
    # pos = position if position >= 0 else (activation.shape[1] + position)
    # Project out the component along the direction
    activation[:, position] -= (activation[:, position].to(direction.dtype) @ direction).unsqueeze(-1) * direction
    # activation[:, pos] -= (activation[:, pos] @ direction) * direction
    return activation

def apply_hook_to_model(
    model: HookedTransformer,
    direction: Float[Tensor, "layer d_model"],
    layer: Optional[List[int]] = None,
    coeff: float = 1.0,
    ablate: bool = False,
    position: int = -1,
    hook_name: str = 'resid_post'
) -> HookedTransformer:
    """
    Apply a hook to the model to modify activations using a direction.
    
    Parameters:
    - model: The transformer_lens model
    - direction: The direction vector 
    - layer: Optional specific layer to apply to (if None, applies to all layers)
    - coeff: Coefficient for scaling the direction (only used if ablate=False)
    - ablate: Whether to ablate the direction instead of adding it
    - position: Specific position to apply to
    - hook_name: Name of the hook point to modify
    
    Returns:
    - Dictionary of hooks
    """
    hooks_dict = {}
    
    if ablate:
        hook_fn = functools.partial(direction_ablation_hook, direction=direction, position=position)
    else:
        hook_fn = functools.partial(activation_addition_hook, vector=direction, coeff=coeff, position=position)
    
    # Apply to all layers or a specific layer
    if layer is not None:
        for l in layer:
            key = get_act_name(hook_name, layer=l)
            hooks_dict[key] = hook_fn
    else:
        for l in range(model.cfg.n_layers):
            key = get_act_name(hook_name, layer=l)
            hooks_dict[key] = hook_fn
            
    return hooks_dict

def visualize_directions(directions, token_labels=None, title="Direction Norms Across Layers", output_path=None):
    """
    Visualize the norms of extracted direction vectors across layers.
    
    Parameters:
    - directions: Directions tensor of shape (n_positions, n_layers, d_model)
    - token_labels: Optional labels for positions
    - title: Plot title
    - output_path: Path to save the visualization
    """
    n_pos, n_layer, _ = directions.shape
    
    # Calculate the norm of each direction vector
    direction_norms = torch.norm(directions, dim=2).cpu().detach().numpy()
    
    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add a trace for each position
    for i in range(n_pos):
        label = f"Position {i}" if token_labels is None else f"Position {i}: {token_labels[i]}"
        ax.plot(range(n_layer), direction_norms[i], label=label)
    
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Direction Norm")
    ax.legend(title="Position")
    
    if output_path:
        plt.savefig(output_path)
    
    plt.close() 