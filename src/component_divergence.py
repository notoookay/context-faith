"""
Component Divergence Analysis

This module analyzes model activations to identify components that diverge significantly
between two kinds of inputs. It could uses t-statistics to rank components by their
degree of divergence.
"""

import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
import wandb
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def parse_args():
    """Parse command line arguments for activation analysis."""
    parser = argparse.ArgumentParser(description="Analyze model activations to identify fact-sensitive pathways")
    
    # Required arguments - change to use two separate activation files
    parser.add_argument("--load_activations_group1", type=str, required=True,
                        help="Path to load pre-computed activations for group 1")
    parser.add_argument("--load_activations_group2", type=str, required=True,
                        help="Path to load pre-computed activations for group 2")
    parser.add_argument("--group1_name", type=str, required=True,
                        help="Name of group 1")
    parser.add_argument("--group2_name", type=str, required=True,
                        help="Name of group 2")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="output/component_divergence",
                        help="Directory to save analysis results")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top components to analyze")
    parser.add_argument("--position", type=int, default=None,
                        help="Specific position to analyze. If None, the first available position will be used.")
    parser.add_argument("--min_abs_t", type=float, default=0.0,
                        help="Minimum absolute t-statistic value to consider a component significant")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run name for organizing output files")
    parser.add_argument("--layer_range", type=str, default=None,
                        help="Comma-separated range of layers to analyze (e.g., '0,12'). If None, all layers are used.")
    
    # Visualization options
    parser.add_argument("--create_heatmap", action="store_true",
                        help="Create a heatmap visualization of t-statistics across layers")
    parser.add_argument("--plot_distribution", action="store_true",
                        help="Plot the distribution of t-statistics")
    
    # Wandb related arguments
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="rag-fact-pathway-analysis",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name")

    return parser.parse_args()

def load_model_activations(load_path: str) -> Dict:
    """
    Load model activations from disk.
    
    Args:
        load_path: Path to load the activations from
        
    Returns:
        Dictionary of activations
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Activation file not found at {load_path}")
    
    activations = torch.load(load_path)
    print(f"Loaded activations from {load_path}")
    return activations

def compare_means(sample1, sample2, alpha=0.05):
    """
    Performs a one-tailed Welch's t-test to determine if sample1's mean is 
    significantly larger than sample2's mean.
    
    Parameters:
    sample1, sample2: arrays or lists of numerical values
    alpha: significance level (default 0.05)
    
    Returns:
    dict containing test results and descriptive statistics
    """
    # Calculate descriptive statistics
    stats_dict = {
        'mean1': np.mean(sample1),
        'mean2': np.mean(sample2),
        'std1': np.std(sample1, ddof=1),
        'std2': np.std(sample2, ddof=1),
        'n1': len(sample1),
        'n2': len(sample2)
    }
    
    # Perform Welch's t-test (not assuming equal variances)
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    
    # Convert to one-tailed p-value
    # If t-stat is positive (mean1 > mean2), we use p_value/2
    # If t-stat is negative (mean1 < mean2), we use 1 - p_value/2
    one_tailed_p = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    
    # Add test results to dictionary
    stats_dict.update({
        't_statistic': t_stat,
        'p_value': one_tailed_p,
        'significant': one_tailed_p < alpha
    })
    
    return stats_dict

def test_mean_difference(data, alternative='two-sided', alpha=0.05):
    """
    Performs a one-sample t-test to determine if the mean is different from zero.
    
    Parameters:
    data (array-like): Sample data to test
    alternative (str): Type of test to perform: 'two-sided', 'greater', or 'less'
    alpha (float): Significance level, default is 0.05
    
    Returns:
    dict: Dictionary containing test results including:
        - t_statistic: The t-statistic
        - p_value: The p-value
        - mean: Sample mean
        - significant: Boolean indicating if result is significant
        - effect_size: Cohen's d effect size
        - test_type: String indicating the type of test performed
    """
    # Perform t-test
    t_stat, p_val = stats.ttest_1samp(data, popmean=0, alternative=alternative)
    
    # Calculate effect size (Cohen's d)
    effect_size = np.mean(data) / np.std(data, ddof=1)
    
    results = {
        't_statistic': t_stat,
        'p_value': p_val,
        'mean': np.mean(data),
        'significant': p_val < alpha,
        'effect_size': effect_size,
        'test_type': alternative
    }
    
    return results

def apply_pca_to_activations(
    group1_activations: Dict,
    group2_activations: Dict,
    layer_hook_name: str,
    position: int,
    n_components: int = 2,
    output_dir: str = None
):
    """
    Apply PCA to compare activation patterns between original and modified inputs.
    
    Args:
        original_activations: Dict of original model activations
        modified_activations: Dict of modified model activations
        layer_hook_name: Layer and hook type to analyze (e.g., "20_hook_resid_post")
        position: Position in the sequence to analyze
        n_components: Number of PCA components to use (default: 2)
        output_dir: Directory to save visualization
        
    Returns:
        Tuple of (pca_model, projected_data, labels)
    """
    # Parse layer and hook from layer_hook_name
    parts = layer_hook_name.split('_', 1)
    layer = int(parts[0])
    hook_type = parts[1]
    
    # Extract activations
    orig_acts = group1_activations[layer][position][hook_type].cpu().float()
    mod_acts = group2_activations[layer][position][hook_type].cpu().float()
    
    # Reshape to 2D for PCA: [batch_size, hidden_dim]
    batch_size_orig = orig_acts.shape[0]
    batch_size_mod = mod_acts.shape[0]
    
    orig_2d = orig_acts.reshape(batch_size_orig, -1).numpy()
    mod_2d = mod_acts.reshape(batch_size_mod, -1).numpy()
    
    # Combine for PCA
    combined = np.vstack([orig_2d, mod_2d])
    
    # Create labels
    labels = np.array(['group1'] * batch_size_orig + ['group2'] * batch_size_mod)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(combined)
    
    # Create visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        for label, color in zip(['group1', 'group2'], ['blue', 'red']):
            mask = labels == label
            plt.scatter(
                projected[mask, 0], 
                projected[mask, 1],
                c=color,
                label=label,
                alpha=1.0,
                linewidth=0.5
            )
        
        # Add labels and legend
        plt.xlabel(f'PC1')
        plt.ylabel(f'PC2')
        plt.title(f'PCA of Activations at Layer {layer}, {hook_type.split("_")[-1]}, Position {position}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Add variance explained text
        # total_var = sum(pca.explained_variance_ratio_[:n_components])
        # plt.annotate(
        #     f'Total variance explained: {total_var:.2%}',
        #     xy=(0.02, 0.02),
        #     xycoords='axes fraction',
        #     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        # )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pca_layer{layer}_{hook_type}_pos{position}.png'), dpi=300)
        plt.close()
    
    return pca, projected, labels

def apply_tsne_to_activations(
    group1_activations: Dict,
    group2_activations: Dict,
    layer_hook_name: str,
    position: int,
    n_components: int = 2,
    perplexity: int = 30,
    output_dir: str = None
):
    """
    Apply t-SNE to compare activation patterns between original and modified inputs.
    
    Args:
        original_activations: Dict of original model activations
        modified_activations: Dict of modified model activations
        layer_hook_name: Layer and hook type to analyze (e.g., "20_hook_resid_post")
        position: Position in the sequence to analyze
        n_components: Number of t-SNE components to use (default: 2)
        perplexity: t-SNE perplexity parameter (default: 30)
        output_dir: Directory to save visualization
        
    Returns:
        Tuple of (projected_data, labels)
    """
    # Parse layer and hook from layer_hook_name
    parts = layer_hook_name.split('_', 1)
    layer = int(parts[0])
    hook_type = parts[1]
    
    # Extract activations
    orig_acts = group1_activations[layer][position][hook_type].cpu().float()
    mod_acts = group2_activations[layer][position][hook_type].cpu().float()
    
    # Reshape to 2D for t-SNE: [batch_size, hidden_dim]
    batch_size_orig = orig_acts.shape[0]
    batch_size_mod = mod_acts.shape[0]
    
    orig_2d = orig_acts.reshape(batch_size_orig, -1).numpy()
    mod_2d = mod_acts.reshape(batch_size_mod, -1).numpy()
    
    # Combine for t-SNE
    combined = np.vstack([orig_2d, mod_2d])
    
    # Create labels
    labels = np.array(['group1'] * batch_size_orig + ['group2'] * batch_size_mod)
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    projected = tsne.fit_transform(combined)
    
    # Create visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        for label, color in zip(['group1', 'group2'], ['blue', 'red']):
            mask = labels == label
            plt.scatter(
                projected[mask, 0], 
                projected[mask, 1],
                c=color,
                label=label,
                alpha=0.7,
                edgecolor='k'
            )
        
        # Add labels and legend
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f't-SNE of Activations at Layer {layer}, {hook_type.split("_")[-1]}, Position {position}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_layer{layer}_{hook_type}_pos{position}.png'), dpi=300)
        plt.close()
    
    return projected, labels

def visualize_activation_changes(
    group1_activations: Dict,
    group2_activations: Dict,
    position: int,
    output_dir: str,
    n_components: int = 2
):
    """
    Apply PCA and t-SNE to all components to visualize activation differences.
    
    Args:
        original_activations: Dict of original model activations
        modified_activations: Dict of modified model activations
        position: Position in the sequence to analyze
        output_dir: Directory to save visualizations
        n_components: Number of components to use
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each visualization method
    pca_output_dir = os.path.join(output_dir, 'pca')
    tsne_output_dir = os.path.join(output_dir, 'tsne')
    os.makedirs(pca_output_dir, exist_ok=True)
    os.makedirs(tsne_output_dir, exist_ok=True)
    
    # Get all available layers
    available_layers = sorted([int(l) for l in group1_activations.keys()])
    # Only use hook_resid_post
    hook_type = 'hook_resid_post'
    
    print(f"Visualizing activation changes for all components")
    
    for layer in available_layers:
        layer_hook_name = f"{layer}_{hook_type}"
        
        # Apply PCA
        try:
            pca, projected, labels = apply_pca_to_activations(
                group1_activations,
                group2_activations,
                layer_hook_name,
                position,
                n_components,
                pca_output_dir
            )
            
            print(f"PCA for {layer_hook_name}: "
                  f"Explained variance: {pca.explained_variance_ratio_[:n_components].sum():.2%}")
            
        except Exception as e:
            print(f"Error applying PCA to {layer_hook_name}: {e}")
            continue
        
        # Apply t-SNE
        # try:
        #     projected, labels = apply_tsne_to_activations(
        #         original_activations,
        #         modified_activations,
        #         layer_hook_name,
        #         position,
        #         n_components,
        #         perplexity=min(30, len(labels)-1),  # Adjust perplexity based on sample size
        #         output_dir=tsne_output_dir
        #     )
            
        #     print(f"t-SNE completed for {layer_hook_name}")
            
        # except Exception as e:
        #     print(f"Error applying t-SNE to {layer_hook_name}: {e}")
        #     continue

def rank_components_by_t_statistic(
    group1_activations: Dict, 
    group2_activations: Dict, 
    position: int = None,
    layer_range: Tuple[int, int] = None,
    top_k: int = 100,
    min_abs_t: float = 0.0
) -> Tuple[List[Dict], Dict]:
    """
    Rank model components by t-statistic magnitude to identify largest divergences.
    
    Args:
        group1_activations: Dict mapping layer names to activation tensors
        group2_activations: Dict with same structure as group1_activations
        position: Specific position to analyze. If None, first available one is used.
        layer_range: Tuple of (start_layer, end_layer) to limit analysis
        top_k: Number of top components to return
        min_abs_t: Minimum absolute t-statistic threshold
        
    Returns:
        Tuple of (top_components, all_t_stats_by_layer) where:
        - top_components: List of component dicts with metadata
        - all_t_stats_by_layer: Dict mapping layers to lists of t-statistics
    """
    all_components = []
    all_t_stats_by_layer = {}
    
    # Get available layers and filter by range if specified
    available_layers = sorted([int(l) for l in group1_activations.keys()])
    if layer_range:
        start_layer, end_layer = layer_range
        available_layers = [l for l in available_layers if start_layer <= l <= end_layer]
    
    # If no position specified, use the first available one
    if position is None:
        first_layer = available_layers[0]
        available_positions = list(group1_activations[first_layer].keys())
        if not available_positions:
            raise ValueError(f"No positions found in activations for layer {first_layer}")
        position = available_positions[0]
        print(f"No position specified, using position {position}")
    
    print(f"Analyzing divergence at position {position} across {len(available_layers)} layers")
    
    # Only use hook_resid_post
    hook_type = 'hook_resid_post'
    
    for layer in tqdm(available_layers, desc="Computing t-statistics by layer"):
        # Check if the specified position exists in both activation sets
        if position not in group1_activations[layer] or position not in group2_activations[layer]:
            print(f"Position {position} not found in layer {layer}, skipping")
            continue
        
        # Check if this hook type exists for this layer and position
        if (hook_type not in group1_activations[layer][position].keys() or 
            hook_type not in group2_activations[layer][position].keys()):
            print(f"{hook_type} not found in layer {layer}, position {position}, skipping")
            continue
        
        # Get activations for the specified position and hook type
        # Check if activations are empty
        if group1_activations[layer][position][hook_type].shape[0] == 0 or group2_activations[layer][position][hook_type].shape[0] == 0:
            print(f"Empty activations found for layer {layer}, position {position}, hook {hook_type}, skipping")
            continue
        
        # Create a combined layer id (numeric for sorting)
        combined_layer_id = layer
        layer_hook_name = f"{layer}_{hook_type}"
        
        # Calculate difference vectors for each example
        diff_vectors = group1_activations[layer][position][hook_type].cpu().float() - group2_activations[layer][position][hook_type].cpu().float()
        
        # Compute vector-level t-statistic (across all examples)
        # We flatten all activation differences into a single vector
        flattened_diff = diff_vectors.reshape(-1).numpy()
        t_test_results = test_mean_difference(flattened_diff, alternative='two-sided')
        
        # Store the overall t-statistic for this layer+hook
        t_stat = t_test_results['t_statistic']
        p_val = t_test_results['p_value']
        
        # Calculate mean difference
        mean_diff = float(group1_activations[layer][position][hook_type].mean() - group2_activations[layer][position][hook_type].mean())
        
        # Add component to list if it meets threshold
        if abs(t_stat) >= min_abs_t:
            all_components.append({
                'layer': layer,
                'hook_type': hook_type,
                'layer_hook_id': combined_layer_id,  # For sorting
                'layer_hook_name': layer_hook_name,  # For display
                't_statistic': float(t_stat),
                'abs_t_statistic': float(abs(t_stat)),
                'p_value': float(p_val),
                'mean_diff': mean_diff,
                'group1_mean': float(group1_activations[layer][position][hook_type].mean()),
                'group2_mean': float(group2_activations[layer][position][hook_type].mean()),
                'position': position
            })
        
        # Store t-statistic for this layer-hook
        all_t_stats_by_layer[layer_hook_name] = [float(t_stat)]
    
    # Sort by absolute t-statistic value (descending)
    all_components.sort(key=lambda x: x['abs_t_statistic'], reverse=True)
    
    # Return top-k components and all t-statistics by layer
    return all_components[:top_k], all_t_stats_by_layer

def visualize_top_components(
    top_components: List[Dict], 
    output_dir: str,
    title: str = "Top Components by t-statistic"
):
    """
    Visualize the top components and their t-statistics.
    
    Args:
        top_components: List of component dictionaries
        output_dir: Directory to save visualization
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data for visualization
    if len(top_components) > 20:
        display_components = top_components[:20]
    else:
        display_components = top_components
        
    # Create more descriptive component labels that include hook type
    component_labels = [f"L{comp['layer']}-{comp['hook_type'].split('_')[-1]}" 
                       for comp in display_components]
    t_stats = [comp['t_statistic'] for comp in display_components]
    
    # Plot t-statistics for top components
    colors = ['blue' if t > 0 else 'red' for t in t_stats]
    plt.bar(range(len(component_labels)), t_stats, color=colors)
    plt.xticks(range(len(component_labels)), component_labels, rotation=90)
    plt.xlabel('Component')
    plt.ylabel('t-statistic')
    plt.title(title)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add a legend explaining colors
    plt.legend(['Zero', 'Group 1 > Group 2', 'Group 2 > Group 1'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_components.png'), dpi=300)
    plt.close()
    
    # Visualize layer distribution of top components
    # Group by layer and hook type
    layer_hook_counts = {}
    for comp in top_components:
        layer_hook_name = comp['layer_hook_name']
        if layer_hook_name in layer_hook_counts:
            layer_hook_counts[layer_hook_name] += 1
        else:
            layer_hook_counts[layer_hook_name] = 1
    
    sorted_items = sorted(
        [(comp['layer_hook_id'], comp['layer_hook_name']) for comp in top_components],
        key=lambda x: x[0]
    )
    unique_layer_hooks = []
    for id_val, name in sorted_items:
        if name not in unique_layer_hooks:
            unique_layer_hooks.append(name)
    
    # Get counts in the proper order
    counts = [layer_hook_counts[name] for name in unique_layer_hooks]
    
    # Create more descriptive labels
    display_labels = []
    for name in unique_layer_hooks:
        print(name)
        layer, hook_type = name.split('_', 1)
        hook_short = hook_type.split('_')[-1]  # 'mid' or 'post'
        display_labels.append(f"L{layer}-{hook_short}")
    
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(display_labels)), counts)
    plt.xticks(range(len(display_labels)), display_labels, rotation=90)
    plt.xlabel('Layer-Hook')
    plt.ylabel('Count in Top Components')
    plt.title(f'Distribution of Top {len(top_components)} Components Across Layer-Hooks')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_distribution.png'), dpi=300)
    plt.close()

def create_t_statistic_heatmap(
    all_t_stats_by_layer: Dict, 
    output_dir: str,
    title: str = "t-statistics Distribution Across Layers"
):
    """
    Create a heatmap visualization of t-statistics across layers.
    
    Args:
        all_t_stats_by_layer: Dict mapping layer-hook names to lists of t-statistics
        output_dir: Directory to save visualization
        title: Plot title
    """
    # Find the layer with the most components to determine heatmap dimensions
    max_components = max(len(stats) for stats in all_t_stats_by_layer.values())
    
    # Create a matrix of t-statistics
    layer_hooks = sorted(all_t_stats_by_layer.keys(), 
                         key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))  # Sort by layer, then hook type
    t_stat_matrix = np.zeros((len(layer_hooks), max_components))
    
    for i, layer_hook in enumerate(layer_hooks):
        t_stats = all_t_stats_by_layer[layer_hook]
        t_stat_matrix[i, :len(t_stats)] = t_stats
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    
    # Use a diverging colormap centered at 0
    vmax = max(abs(np.min(t_stat_matrix)), abs(np.max(t_stat_matrix)))
    vmin = -vmax
    
    # Create more descriptive labels
    display_labels = []
    for name in layer_hooks:
        layer, hook_type = name.split('_', 1)
        hook_short = hook_type.split('_')[-1]  # 'mid' or 'post' (residual)
        display_labels.append(f"L{layer}-{hook_short}")
    
    sns.heatmap(
        t_stat_matrix, 
        cmap='coolwarm', 
        center=0, 
        vmin=vmin, 
        vmax=vmax,
        xticklabels=100 if max_components > 500 else 50,  # Show fewer x tick labels if many components
        yticklabels=display_labels
    )
    
    plt.title(title)
    plt.xlabel('Component Index')
    plt.ylabel('Layer-Hook')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 't_statistic_heatmap.png'), dpi=300)
    plt.close()
    
    # Create a distribution plot of t-statistics for each layer
    plt.figure(figsize=(14, 8))
    
    # Calculate t-statistic density for each layer-hook
    for layer_hook in layer_hooks:
        t_stats = all_t_stats_by_layer[layer_hook]
        layer, hook_type = layer_hook.split('_', 1)
        hook_short = hook_type.split('_')[-1]  # 'mid' or 'post' (residual)
        label = f"L{layer}-{hook_short}"
        sns.kdeplot(t_stats, label=label)
    
    plt.title("Distribution of t-statistics by Layer-Hook")
    plt.xlabel("t-statistic Value")
    plt.ylabel("Density")
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 't_statistic_distribution.png'), dpi=300)
    plt.close()

def plot_t_statistic_analysis(
    top_components: List[Dict],
    all_t_stats_by_layer: Dict,
    output_dir: str
):
    """
    Create comprehensive visualizations of t-statistic analysis.
    
    Args:
        top_components: List of top component dictionaries
        all_t_stats_by_layer: Dict mapping layer-hook names to lists of t-statistics
        output_dir: Directory to save visualizations
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    visualize_top_components(top_components, output_dir)
    
    create_t_statistic_heatmap(all_t_stats_by_layer, output_dir)
    
    plt.figure(figsize=(12, 6))
    
    # Flatten all t-statistics into a single list
    all_t_stats = []
    for layer_hook, stats in all_t_stats_by_layer.items():
        all_t_stats.extend(stats)
    
    sns.histplot(all_t_stats, kde=True, bins=100)
    plt.title("Distribution of all t-statistics")
    plt.xlabel("t-statistic")
    plt.ylabel("Count")
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 't_statistic_histogram.png'), dpi=300)
    plt.close()
    
    # Plot absolute t-statistic values by layer-hook (mean, max)
    plt.figure(figsize=(14, 6))
    
    # Calculate statistics for each layer
    layer_hooks = sorted(all_t_stats_by_layer.keys(), 
                         key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))  # Sort by layer, then hook
    mean_abs_t = [np.mean(np.abs(all_t_stats_by_layer[lh])) for lh in layer_hooks]
    max_abs_t = [np.max(np.abs(all_t_stats_by_layer[lh])) for lh in layer_hooks]
    
    # Create more descriptive x-labels
    display_labels = []
    for name in layer_hooks:
        layer, hook_type = name.split('_', 1)
        hook_short = hook_type.split('_')[-1]  # 'mid' or 'post' (residual)
        display_labels.append(f"L{layer}-{hook_short}")
    
    x = np.arange(len(display_labels))
    plt.plot(x, mean_abs_t, 'o-', label='Mean |t|')
    plt.plot(x, max_abs_t, 's-', label='Max |t|')
    plt.xticks(x, display_labels, rotation=90)
    plt.xlabel('Layer-Hook')
    plt.ylabel('|t-statistic|')
    plt.title('Absolute t-statistic Values by Layer-Hook')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 't_statistic_by_layer.png'), dpi=300)
    plt.close()

def save_analysis_results(
    top_components: List[Dict],
    output_dir: str,
    run_name: str = None
):
    """
    Save analysis results to disk.
    
    Args:
        top_components: List of top component dictionaries
        output_dir: Directory to save results
        run_name: Optional run name for the file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    if run_name:
        filename = f"{run_name}_top_components.json"
    else:
        filename = "top_components.json"
    
    # Save top components
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(top_components, f, indent=2)
    
    print(f"Saved top components to {output_path}")
    
    # Save as CSV for easier analysis
    csv_path = os.path.join(output_dir, filename.replace('.json', '.csv'))
    
    # Create CSV header and rows
    header = list(top_components[0].keys())
    rows = []
    for comp in top_components:
        rows.append(','.join([str(comp[h]) for h in header]))
    
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        f.write('\n'.join(rows))
    
    print(f"Saved top components CSV to {csv_path}")

def main(args):
    # Initialize W&B if enabled
    if args.use_wandb:
        # Define default run name if not provided
        if not args.run_name:
            args.run_name = f"comparison_{args.group1_name}__{args.group2_name}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args)
        )
        print(f"Wandb logging enabled for run: {args.run_name}")
    
    # Create output directory based on group names
    group1_name = args.group1_name
    group2_name = args.group2_name
    args.output_dir = os.path.join(args.output_dir, f"{group1_name}__{group2_name}")

    # Create output directory
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load activations for both groups
    print(f"Loading Group 1 ({args.group1_name}) activations from {args.load_activations_group1}")
    group1_activations = load_model_activations(args.load_activations_group1)
    
    print(f"Loading Group 2 ({args.group2_name}) activations from {args.load_activations_group2}")
    group2_activations = load_model_activations(args.load_activations_group2)
    
    # Convert activation format to be compatible with existing analysis functions
    # The loaded activations might be in format {layer: {pos: tensor}} 
    # but we need {layer: {pos: {'hook_resid_post': tensor}}}
    group1_activations_formatted = {}
    group2_activations_formatted = {}
    
    for layer in group1_activations:
        group1_activations_formatted[layer] = {}
        for pos in group1_activations[layer]:
            group1_activations_formatted[layer][pos] = {
                'hook_resid_post': group1_activations[layer][pos]
            }
    
    for layer in group2_activations:
        group2_activations_formatted[layer] = {}
        for pos in group2_activations[layer]:
            group2_activations_formatted[layer][pos] = {
                'hook_resid_post': group2_activations[layer][pos]
            }
    
    print(f"Successfully loaded {args.group1_name} and {args.group2_name} activations.")
    
    # Determine layer range if specified
    layer_range = None
    if args.layer_range:
        try:
            start_layer, end_layer = map(int, args.layer_range.split(','))
            layer_range = (start_layer, end_layer)
            print(f"Analyzing layers {start_layer} to {end_layer}")
        except ValueError:
            print(f"Error: Invalid layer range '{args.layer_range}'. Using all available layers.")
    
    # Compute t-statistics and rank components
    top_components, all_t_stats_by_layer = rank_components_by_t_statistic(
        group1_activations=group1_activations_formatted,
        group2_activations=group2_activations_formatted,
        position=args.position,
        layer_range=layer_range,
        top_k=args.top_k,
        min_abs_t=args.min_abs_t
    )
    
    # Print top components
    print(f"\nTop {min(10, len(top_components))} components with highest |t-statistic|:")
    for i, comp in enumerate(top_components[:10]):
        hook_short = comp['hook_type'].split('_')[-1]  # 'mid' or 'post'
        print(f"{i+1}. Layer {comp['layer']}-{hook_short}")
        print(f"   t-statistic: {comp['t_statistic']:.4f}")
        print(f"   Group 1 mean: {comp['original_mean']:.4f}")
        print(f"   Group 2 mean: {comp['modified_mean']:.4f}")
        print(f"   Mean difference: {comp['mean_diff']:.4f}")
        print(f"   p-value: {comp['p_value']:.4f}")
        print()
    
    # Create visualizations
    if args.create_heatmap or args.plot_distribution:
        plot_t_statistic_analysis(top_components, all_t_stats_by_layer, args.output_dir)
    else:
        visualize_top_components(top_components, args.output_dir)
    
    # Apply PCA and t-SNE to visualize activation changes for top components
    vis_output_dir = os.path.join(args.output_dir, "visualizations")
    visualize_activation_changes(
        group1_activations_formatted, 
        group2_activations_formatted, 
        args.position, 
        vis_output_dir
    )
    
    # Save results
    save_analysis_results(top_components, args.output_dir, args.run_name)
    
    # Log to wandb if enabled
    if args.use_wandb:
        # Log images
        for img_file in os.listdir(args.output_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(args.output_dir, img_file)
                wandb.log({img_file: wandb.Image(img_path)})
        
        # Log PCA and t-SNE visualizations
        vis_dirs = ['pca', 'tsne']
        for vis_dir in vis_dirs:
            full_vis_dir = os.path.join(vis_output_dir, vis_dir)
            if os.path.exists(full_vis_dir):
                for img_file in os.listdir(full_vis_dir):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(full_vis_dir, img_file)
                        wandb.log({f"{vis_dir}/{img_file}": wandb.Image(img_path)})
        
        # Log top components table
        top_comp_table = wandb.Table(
            columns=list(top_components[0].keys()),
            data=[[comp[k] for k in top_components[0].keys()] for comp in top_components]
        )
        wandb.log({"top_components": top_comp_table})
        
        # Log summary metrics
        wandb.log({
            "total_components_analyzed": len(top_components),
            "max_abs_t_statistic": max([comp["abs_t_statistic"] for comp in top_components]),
            "mean_abs_t_statistic": np.mean([comp["abs_t_statistic"] for comp in top_components]),
        })
        
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)