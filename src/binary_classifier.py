"""
Binary Classifier for Layer-wise Activation Analysis

This module trains binary classifiers for each layer using activation data
to distinguish between original and modified inputs. It's designed to work
with the same data format as component_divergence.py.
"""

import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.utils import shuffle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import wandb
import pandas as pd
from sklearn.model_selection import cross_val_score

def parse_args():
    """Parse command line arguments for binary classifier training."""
    parser = argparse.ArgumentParser(description="Train binary classifiers for each layer using activation data")
    
    # Required arguments
    parser.add_argument("--load_activations_group1_train", type=str, required=True, 
                        help="Path to the training activations file for group 1 (negative class)")
    parser.add_argument("--load_activations_group2_train", type=str, required=True,
                        help="Path to the training activations file for group 2 (positive class)")
    parser.add_argument("--load_activations_group1_test", type=str, required=False, default=None,
                        help="Path to the test activations file for group 1 (negative class)")
    parser.add_argument("--load_activations_group2_test", type=str, required=False, default=None,
                        help="Path to the test activations file for group 2 (positive class)")
    parser.add_argument("--group1_name", type=str, default="group1",
                        help="Name for group 1")
    parser.add_argument("--group2_name", type=str, default="group2",
                        help="Name for group 2")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="output/binary_classifiers",
                        help="Directory to save trained classifiers and results")
    parser.add_argument("--position", type=int, default=-1,
                        help="Specific position to analyze. If None, the first available position will be used.")
    parser.add_argument("--layer_range", type=str, default=None,
                        help="Comma-separated range of layers to analyze (e.g., '0,12'). If None, all layers are used.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_classifiers", action="store_true",
                        help="Whether to save trained classifier models")
    
    # Classifier hyperparameters for high-dimensional data
    parser.add_argument("--regularization_strength", type=float, default=1.0,
                        help="Regularization strength (C parameter). Smaller values = stronger regularization")
    parser.add_argument("--penalty", type=str, default='l2', choices=['l1', 'l2'],
                        help="Regularization penalty type")
    parser.add_argument("--solver", type=str, default='liblinear', 
                        choices=['liblinear', 'saga', 'lbfgs'],
                        help="Solver algorithm (liblinear recommended for high-dim data)")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of iterations for convergence")
    parser.add_argument("--tune_hyperparams", action="store_true",
                        help="Automatically tune regularization strength using cross-validation")
    
    # Visualization options
    parser.add_argument("--create_heatmap", action="store_true",
                        help="Create a heatmap visualization of classifier performance across layers")
    
    # Wandb related arguments
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="layer-binary-classifiers",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name")

    return parser.parse_args()

def load_activations(load_path: str) -> Dict:
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

def prepare_data_for_classifier(
    group1_activations: Dict, 
    group2_activations: Dict,
    layer_idx: int,
    position: int,
    random_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training a binary classifier for a specific layer.
    
    Args:
        group1_activations: Dict of group1 model activations
        group2_activations: Dict of group2 model activations
        layer_idx: Layer index to prepare data for
        position: Position in the sequence to analyze
        random_seed: Random seed for shuffling the data
        
    Returns:
        X, y arrays for classifier training/testing
    """
    
    # Check if the specified layer, position exists
    if position not in group1_activations[layer_idx] or position not in group2_activations[layer_idx]:
        raise ValueError(f"Position {position} not found in layer {layer_idx}")
    
    # Extract activations
    group1_acts = group1_activations[layer_idx][position].cpu().float()
    group2_acts = group2_activations[layer_idx][position].cpu().float()
    
    # Reshape to 2D if needed: [batch_size, hidden_dim]
    if len(group1_acts.shape) > 2:
        batch_size_group1 = group1_acts.shape[0]
        group1_acts = group1_acts.reshape(batch_size_group1, -1)
        
        batch_size_group2 = group2_acts.shape[0]
        group2_acts = group2_acts.reshape(batch_size_group2, -1)
    
    # Create labels: 0 for group1, 1 for group2
    group1_labels = np.zeros(len(group1_acts))
    group2_labels = np.ones(len(group2_acts))
    
    # Combine data
    X = np.vstack([group1_acts.numpy(), group2_acts.numpy()])
    y = np.concatenate([group1_labels, group2_labels])
    
    # Shuffle the data to prevent the model from learning the order
    # X, y = shuffle(X, y, random_state=random_seed)
    
    return X, y

def train_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    random_seed: int = 42,
    C: float = 1.0,
    penalty: str = 'l2',
    solver: str = 'liblinear',
    max_iter: int = 1000
) -> LogisticRegression:
    """
    Train a logistic regression classifier optimized for high-dimensional data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_seed: Random seed for reproducibility
        C: Regularization strength (smaller values = stronger regularization)
        penalty: Regularization type ('l1', 'l2', or 'elasticnet')
        solver: Algorithm to use ('liblinear' recommended for high-dim data)
        max_iter: Maximum number of iterations for convergence
        
    Returns:
        Trained classifier
    """
    # For high-dimensional data features, use optimized settings
    clf = LogisticRegression(
        random_state=random_seed,
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=max_iter,
        # Use balanced class weights if data is imbalanced
        class_weight='balanced',
    )
    
    clf.fit(X_train, y_train)
    
    # Check if the model converged
    if hasattr(clf, 'n_iter_') and clf.n_iter_ >= max_iter:
        print(f"Warning: Classifier may not have converged (reached max_iter={max_iter})")
    
    return clf

def tune_regularization_strength(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
    penalty: str = 'l2',
    solver: str = 'liblinear',
    max_iter: int = 1000,
    cv_folds: int = 5
) -> float:
    """
    Automatically tune regularization strength using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_seed: Random seed for reproducibility
        penalty: Regularization penalty type
        solver: Solver algorithm
        max_iter: Maximum iterations
        cv_folds: Number of cross-validation folds
        
    Returns:
        Best regularization strength (C parameter)
    """
    # Test different regularization strengths
    # For high-dimensional data, we often need stronger regularization
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    best_score = 0
    best_C = 1.0
    
    print("Tuning regularization strength...")
    for C in C_values:
        clf = LogisticRegression(
            random_state=random_seed,
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight='balanced'
        )
        
        # Use cross-validation to get more robust estimate
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring='roc_auc')
        mean_score = cv_scores.mean()
        
        print(f"C={C}: CV AUC = {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
    
    print(f"Best regularization strength: C={best_C} (AUC={best_score:.4f})")
    return best_C

def evaluate_classifier(
    clf: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Evaluate a trained classifier on test data.
    
    Args:
        clf: Trained classifier
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        Dictionary of performance metrics
    """
    # Evaluate on test data
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Add confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics

def train_classifiers_for_all_layers(
    train_group1_activations: Dict,
    train_group2_activations: Dict,
    test_group1_activations: Dict = None,
    test_group2_activations: Dict = None,
    position: int = None,
    layer_range: Tuple[int, int] = None,
    random_seed: int = 42,
    C: float = 1.0,
    penalty: str = 'l2',
    solver: str = 'liblinear',
    max_iter: int = 100,
    tune_hyperparams: bool = False
) -> Tuple[Dict[int, LogisticRegression], Dict[int, Dict]]:
    """
    Train binary classifiers for all layers or a specified range of layers.
    
    Args:
        train_group1_activations: Dict of training group1 model activations
        train_group2_activations: Dict of training group2 model activations
        test_group1_activations: Dict of test group1 model activations (optional)
        test_group2_activations: Dict of test group2 model activations (optional)
        position: Position to analyze (if None, first available position is used)
        layer_range: Optional tuple of (start_layer, end_layer) to limit analysis
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dict of classifiers by layer, dict of metrics by layer)
    """
    # Get available layers and filter by range if specified
    available_layers = sorted([int(l) for l in train_group1_activations.keys()])
    if layer_range:
        start_layer, end_layer = layer_range
        available_layers = [l for l in available_layers if start_layer <= l <= end_layer]
    
    # If no position specified, use the first available one
    if position is None:
        first_layer = available_layers[0]
        available_positions = list(train_group1_activations[first_layer].keys())
        if not available_positions:
            raise ValueError(f"No positions found in activations for layer {first_layer}")
        position = available_positions[0]
        print(f"No position specified, using position {position}")
    
    print(f"Training classifiers at position {position} across {len(available_layers)} layers")
    
    # Check if we have test data
    has_test_data = test_group1_activations is not None and test_group2_activations is not None
    if has_test_data:
        print("Using provided test data for evaluation")
    else:
        print("No test data provided, evaluation will be on training data")
    
    # Train classifiers for each layer
    classifiers = {}
    metrics = {}
    
    for layer in tqdm(available_layers, desc="Training classifiers by layer"):
        try:
            # Prepare training data for this layer
            X_train, y_train = prepare_data_for_classifier(
                group1_activations=train_group1_activations,
                group2_activations=train_group2_activations,
                layer_idx=layer,
                position=position,
                random_seed=random_seed
            )
            
            # Tune hyperparameters if requested
            current_C = C
            if tune_hyperparams:
                print(f"Tuning hyperparameters for layer {layer}")
                current_C = tune_regularization_strength(
                    X_train=X_train,
                    y_train=y_train,
                    random_seed=random_seed,
                    penalty=penalty,
                    solver=solver,
                    max_iter=max_iter
                )
            
            # Train classifier
            clf = train_classifier(
                X_train=X_train,
                y_train=y_train,
                random_seed=random_seed,
                C=current_C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter
            )
            
            classifiers[layer] = clf
            
            # Prepare test data and evaluate if we have test data
            if has_test_data:
                try:
                    X_test, y_test = prepare_data_for_classifier(
                        group1_activations=test_group1_activations,
                        group2_activations=test_group2_activations,
                        layer_idx=layer,
                        position=position,
                        random_seed=random_seed
                    )
                    
                    # Evaluate on test data
                    layer_metrics = evaluate_classifier(
                        clf=clf,
                        X_test=X_test,
                        y_test=y_test
                    )
                    metrics[layer] = layer_metrics
                except Exception as e:
                    print(f"Error evaluating classifier for layer {layer}: {e}")
                    # Evaluate on training data as fallback
                    layer_metrics = evaluate_classifier(clf, X_train, y_train)
                    metrics[layer] = layer_metrics
                    metrics[layer]['note'] = "Evaluated on training data due to test data error"
            else:
                # Evaluate on training data
                layer_metrics = evaluate_classifier(clf, X_train, y_train)
                metrics[layer] = layer_metrics
                metrics[layer]['note'] = "Evaluated on training data (no test data provided)"
            
        except Exception as e:
            print(f"Error training classifier for layer {layer}: {e}")
            continue
    
    return classifiers, metrics

def visualize_classifier_performance(
    metrics: Dict[int, Dict],
    output_dir: str,
    title: str = "Classifier Performance Across Layers"
):
    """
    Visualize classifier performance metrics across layers.
    
    Args:
        metrics: Dictionary mapping layer indices to metric dictionaries
        output_dir: Directory to save visualizations
        title: Plot title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract layers and metrics
    layers = sorted(metrics.keys())
    accuracy = [metrics[l]['accuracy'] for l in layers]
    auc = [metrics[l]['auc'] for l in layers]
    precision = [metrics[l]['precision'] for l in layers]
    recall = [metrics[l]['recall'] for l in layers]
    f1 = [metrics[l]['f1'] for l in layers]
    
    # Create a line plot
    plt.figure(figsize=(12, 6))
    plt.plot(layers, accuracy, 'o-', label='Accuracy')
    plt.plot(layers, auc, 's-', label='AUC')
    plt.plot(layers, precision, '^-', label='Precision')
    plt.plot(layers, recall, 'd-', label='Recall')
    plt.plot(layers, f1, '*-', label='F1')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'classifier_performance.png'), dpi=300)
    plt.close()
    
    # Create a heatmap for more detailed visualization
    metrics_df = pd.DataFrame({
        'Layer': layers,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    # Set Layer as index and remove from columns
    metrics_df = metrics_df.set_index('Layer')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df.T, annot=True, fmt=".3f", cmap="Blues", vmin=0.5, vmax=1.0)
    plt.title('Classifier Performance by Layer')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'classifier_heatmap.png'), dpi=300)
    plt.close()
    
    # Save metrics as CSV for further analysis
    metrics_df.to_csv(os.path.join(output_dir, 'classifier_metrics.csv'))
    
    # Find and report top performing layers
    top_by_accuracy = metrics_df['Accuracy'].nlargest(5)
    top_by_auc = metrics_df['AUC'].nlargest(5)
    
    print("\nTop 5 layers by Accuracy:")
    for layer, acc in top_by_accuracy.items():
        print(f"Layer {layer}: {acc:.4f}")
    
    print("\nTop 5 layers by AUC:")
    for layer, auc_val in top_by_auc.items():
        print(f"Layer {layer}: {auc_val:.4f}")
    
    # Create summary file
    with open(os.path.join(output_dir, 'top_layers_summary.txt'), 'w') as f:
        f.write("Top 5 layers by Accuracy:\n")
        for layer, acc in top_by_accuracy.items():
            f.write(f"Layer {layer}: {acc:.4f}\n")
        
        f.write("\nTop 5 layers by AUC:\n")
        for layer, auc_val in top_by_auc.items():
            f.write(f"Layer {layer}: {auc_val:.4f}\n")

def save_classifiers_and_results(
    classifiers: Dict[int, LogisticRegression],
    metrics: Dict[int, Dict],
    output_dir: str,
    run_name: str = None
):
    """
    Save trained classifiers and evaluation metrics.
    
    Args:
        classifiers: Dictionary mapping layer indices to trained classifiers
        metrics: Dictionary mapping layer indices to metric dictionaries
        output_dir: Directory to save results
        run_name: Optional run name for file naming
    """
    # Create output directory
    if run_name:
        output_dir = os.path.join(output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_json = {}
    for layer, layer_metrics in metrics.items():
        metrics_json[str(layer)] = {
            k: float(v) if isinstance(v, (np.float32, np.float64)) else v.tolist() 
            if isinstance(v, np.ndarray) else v
            for k, v in layer_metrics.items()
        }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save classifiers
    os.makedirs(os.path.join(output_dir, 'classifiers'), exist_ok=True)
    for layer, clf in classifiers.items():
        torch.save(clf, os.path.join(output_dir, 'classifiers', f'layer_{layer}_classifier.pt'))
    
    print(f"Saved classifiers and metrics to {output_dir}")

def main():
    """Main function to run the binary classifier training pipeline."""
    args = parse_args()
    
    # Initialize wandb if enabled
    if args.use_wandb:
        # Define default run name if not provided
        group1_name = args.group1_name
        group2_name = args.group2_name
        run_name = f"binary_clf_{group1_name}__{group2_name}_seed_{args.random_seed}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )
        print(f"WandB logging enabled for run: {run_name}")
    else:
        group1_name = args.group1_name
        group2_name = args.group2_name
        run_name = f"binary_clf_{group1_name}__{group2_name}_seed_{args.random_seed}"
    
    # Create output directory based on group names
    args.output_dir = os.path.join(args.output_dir, f"{group1_name}__{group2_name}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training activations for both groups
    print(f"Loading Group 1 training activations from {args.load_activations_group1_train}")
    train_group1_activations = load_activations(args.load_activations_group1_train)
    
    print(f"Loading Group 2 training activations from {args.load_activations_group2_train}")
    train_group2_activations = load_activations(args.load_activations_group2_train)
    
    print(f"Successfully loaded training activations for {args.group1_name} and {args.group2_name}")
    
    # Handle test activations if provided
    test_group1_activations = None
    test_group2_activations = None
    
    if args.load_activations_group1_test and args.load_activations_group2_test:
        print(f"Loading Group 1 test activations from {args.load_activations_group1_test}")
        test_group1_activations = load_activations(args.load_activations_group1_test)
        
        print(f"Loading Group 2 test activations from {args.load_activations_group2_test}")
        test_group2_activations = load_activations(args.load_activations_group2_test)
                
        print(f"Successfully loaded test activations for {args.group1_name} and {args.group2_name}")
    
    # Determine layer range if specified
    layer_range = None
    if args.layer_range:
        try:
            start_layer, end_layer = map(int, args.layer_range.split(','))
            layer_range = (start_layer, end_layer)
            print(f"Analyzing layers {start_layer} to {end_layer}")
        except ValueError:
            print(f"Error: Invalid layer range '{args.layer_range}'. Using all available layers.")
    else:
        total_layers = len(train_group1_activations.keys())
        layer_range = (0, total_layers - 1)

    # Train classifiers for all layers
    classifiers, metrics = train_classifiers_for_all_layers(
        train_group1_activations=train_group1_activations,
        train_group2_activations=train_group2_activations,
        test_group1_activations=test_group1_activations,
        test_group2_activations=test_group2_activations,
        position=args.position,
        layer_range=layer_range,
        random_seed=args.random_seed,
        C=args.regularization_strength,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        tune_hyperparams=args.tune_hyperparams
    )
    
    # Visualize classifier performance
    visualize_classifier_performance(metrics, args.output_dir)
    
    # Save classifiers and results if requested
    if args.save_classifiers:
        save_classifiers_and_results(classifiers, metrics, args.output_dir, run_name)
    
    # Log to wandb if enabled
    if args.use_wandb:
        # Log images
        for img_file in os.listdir(args.output_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(args.output_dir, img_file)
                wandb.log({img_file: wandb.Image(img_path)})
        
        # Log metrics table
        metrics_table = wandb.Table(
            columns=['Layer', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1'],
            data=[[layer, m['accuracy'], m['auc'], m['precision'], m['recall'], m['f1']] 
                  for layer, m in metrics.items()]
        )
        wandb.log({"metrics": metrics_table})
        
        # Log best layers
        best_acc_layer = max(metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        best_auc_layer = max(metrics.items(), key=lambda x: x[1]['auc'])[0]
        
        wandb.log({
            "best_accuracy": metrics[best_acc_layer]['accuracy'],
            "best_accuracy_layer": best_acc_layer,
            "best_auc": metrics[best_auc_layer]['auc'],
            "best_auc_layer": best_auc_layer,
        })
        
        wandb.finish()

if __name__ == "__main__":
    main() 