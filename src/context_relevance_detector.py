"""
Context Relevance Detector

This module uses trained binary classifiers to detect context relevance
at the last prompt token before generation starts.
"""

import os
import torch
import numpy as np
import argparse
from typing import Dict, List, Optional, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import json
from tqdm import tqdm
import pickle

class ContextRelevanceDetector:
    """
    A detector that uses trained binary classifiers to determine context relevance
    based on model activations at the last prompt token.
    """
    
    def __init__(
        self, 
        classifiers_dir: str, 
        model: HookedTransformer,
        target_layers: Optional[List[int]] = None,
        hook_name: str = 'resid_post'
    ):
        """
        Initialize the context relevance detector.
        
        Args:
            classifiers_dir: Directory containing trained classifier files
            model: The transformer model to extract activations from
            target_layers: Specific layers to use for detection (if None, use all available)
            hook_name: Name of the hook point to extract activations from
        """
        self.classifiers_dir = classifiers_dir
        self.model = model
        self.hook_name = hook_name
        self.classifiers = {}
        self.target_layers = target_layers
        
        # Load trained classifiers
        self._load_classifiers()
    
    def _load_classifiers(self):
        """Load trained classifiers from the specified directory."""
        classifiers_path = os.path.join(self.classifiers_dir, 'classifiers')
        
        if not os.path.exists(classifiers_path):
            raise FileNotFoundError(f"Classifiers directory not found: {classifiers_path}")
        
        classifier_files = [f for f in os.listdir(classifiers_path) if f.endswith('_classifier.pt')]
        
        if not classifier_files:
            raise FileNotFoundError(f"No classifier files found in {classifiers_path}")
        
        for clf_file in classifier_files:
            # Extract layer number from filename (format: layer_{layer}_classifier.pt)
            try:
                layer_num = int(clf_file.split('_')[1])
                
                # Only load if it's in target layers (if specified)
                if self.target_layers is None or layer_num in self.target_layers:
                    clf_path = os.path.join(classifiers_path, clf_file)
                    classifier = torch.load(clf_path, map_location='cpu')
                    self.classifiers[layer_num] = classifier
                    print(f"Loaded classifier for layer {layer_num}")
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse layer number from {clf_file}: {e}")
                continue
        
        if not self.classifiers:
            raise ValueError("No classifiers were successfully loaded")
        
        print(f"Successfully loaded {len(self.classifiers)} classifiers for layers: {sorted(self.classifiers.keys())}")
    
    def extract_activations_at_position(
        self, 
        input_ids: torch.Tensor, 
        position: int = -1
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from specified layers at a given position.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position: Position to extract activations from (-1 for last token)
            
        Returns:
            Dictionary mapping layer numbers to activation tensors
        """
        activations = {}
        
        def create_hook(layer_num):
            def hook_fn(activations_tensor: torch.Tensor, hook: HookPoint):
                # Store activations for this layer at the specified position
                if position == -1:
                    # Last token
                    activations[layer_num] = activations_tensor[:, -1, :].clone()
                else:
                    # Specific position
                    if position < activations_tensor.shape[1]:
                        activations[layer_num] = activations_tensor[:, position, :].clone()
                    else:
                        print(f"Warning: Position {position} out of range for sequence length {activations_tensor.shape[1]}")
                return activations_tensor
            return hook_fn
        
        # Create hooks for target layers
        hooks = []
        for layer_num in self.classifiers.keys():
            hook_name = f'blocks.{layer_num}.hook_{self.hook_name}'
            hooks.append((hook_name, create_hook(layer_num)))
        
        # Run forward pass with hooks
        with self.model.hooks(fwd_hooks=hooks):
            _ = self.model(input_ids)
        
        return activations
    
    def predict_relevance(
        self, 
        input_ids: torch.Tensor, 
        position: int = -1,
        return_probabilities: bool = False,
        aggregate_method: str = 'majority_vote'
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Predict context relevance using trained classifiers.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position: Position to extract activations from (-1 for last token)
            return_probabilities: Whether to return prediction probabilities
            aggregate_method: How to aggregate predictions across layers ('majority_vote', 'average', 'max_confidence')
            
        Returns:
            Predictions [batch_size] and optionally probabilities/details
        """
        # Extract activations at the specified position
        activations = self.extract_activations_at_position(input_ids, position)
        
        batch_size = input_ids.shape[0]
        layer_predictions = {}
        layer_probabilities = {}
        
        # Get predictions from each layer's classifier
        for layer_num, layer_activations in activations.items():
            if layer_num in self.classifiers:
                classifier = self.classifiers[layer_num]
                
                # Reshape activations for classifier input
                layer_acts = layer_activations.cpu().numpy()
                if len(layer_acts.shape) > 2:
                    layer_acts = layer_acts.reshape(batch_size, -1)
                
                # Get predictions and probabilities
                predictions = classifier.predict(layer_acts)
                probabilities = classifier.predict_proba(layer_acts)
                
                layer_predictions[layer_num] = predictions
                layer_probabilities[layer_num] = probabilities
        
        # Aggregate predictions across layers
        if aggregate_method == 'majority_vote':
            # Take majority vote across layers
            all_predictions = np.array(list(layer_predictions.values()))  # [num_layers, batch_size]
            final_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
            
        elif aggregate_method == 'average':
            # Average probabilities and threshold at 0.5
            all_probabilities = np.array([probs[:, 1] for probs in layer_probabilities.values()])  # [num_layers, batch_size]
            avg_probabilities = np.mean(all_probabilities, axis=0)
            final_predictions = (avg_probabilities > 0.5).astype(int)
            
        elif aggregate_method == 'max_confidence':
            # Use prediction from the layer with highest confidence
            max_confidence_preds = []
            for batch_idx in range(batch_size):
                max_conf = 0
                best_pred = 0
                for layer_num, probs in layer_probabilities.items():
                    confidence = np.max(probs[batch_idx])
                    if confidence > max_conf:
                        max_conf = confidence
                        best_pred = layer_predictions[layer_num][batch_idx]
                max_confidence_preds.append(best_pred)
            final_predictions = np.array(max_confidence_preds)
        
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")
        
        if return_probabilities:
            details = {
                'layer_predictions': layer_predictions,
                'layer_probabilities': layer_probabilities,
                'aggregate_method': aggregate_method
            }
            return final_predictions, details
        else:
            return final_predictions
    
    def predict_relevance_with_confidence(
        self, 
        input_ids: torch.Tensor, 
        position: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict context relevance with confidence scores.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position: Position to extract activations from (-1 for last token)
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        predictions, details = self.predict_relevance(
            input_ids, position, return_probabilities=True, aggregate_method='average'
        )
        
        # Calculate confidence as the average max probability across layers
        batch_size = input_ids.shape[0]
        confidence_scores = []
        
        for batch_idx in range(batch_size):
            layer_confidences = []
            for layer_num, probs in details['layer_probabilities'].items():
                layer_confidences.append(np.max(probs[batch_idx]))
            confidence_scores.append(np.mean(layer_confidences))
        
        return predictions, np.array(confidence_scores)


def load_detector_from_config(config_path: str, model: HookedTransformer) -> ContextRelevanceDetector:
    """
    Load a context relevance detector from a configuration file.
    
    Args:
        config_path: Path to the detector configuration JSON file
        model: The transformer model to use
        
    Returns:
        Configured ContextRelevanceDetector instance
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return ContextRelevanceDetector(
        classifiers_dir=config['classifiers_dir'],
        model=model,
        target_layers=config.get('target_layers'),
        hook_name=config.get('hook_name', 'resid_post')
    )


def main():
    """Example usage of the context relevance detector."""
    parser = argparse.ArgumentParser(description="Test context relevance detector")
    parser.add_argument("--classifiers_dir", type=str, required=True,
                        help="Directory containing trained classifiers")
    parser.add_argument("--model_name", type=str, default="gpt2-small",
                        help="Model name to load")
    parser.add_argument("--test_prompts", type=str, nargs="+",
                        help="Test prompts to analyze")
    parser.add_argument("--target_layers", type=str, default=None,
                        help="Comma-separated list of layers to use (e.g., '10,11,12')")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Parse target layers
    target_layers = None
    if args.target_layers:
        target_layers = [int(l.strip()) for l in args.target_layers.split(',')]
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = HookedTransformer.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=torch.bfloat16
    )
    
    # Initialize detector
    print(f"Loading detectors from: {args.classifiers_dir}")
    detector = ContextRelevanceDetector(
        classifiers_dir=args.classifiers_dir,
        model=model,
        target_layers=target_layers
    )
    
    # Test prompts
    if args.test_prompts:
        test_prompts = args.test_prompts
    else:
        test_prompts = [
            "The capital of France is Paris. What is the capital of Germany?",
            "Random text about cooking recipes. What is the capital of Germany?",
            "Germany is a country in Europe. What is the capital of Germany?"
        ]
    
    print(f"\nTesting {len(test_prompts)} prompts:")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Tokenize
        input_ids = model.to_tokens(prompt)
        
        # Get prediction with confidence
        predictions, confidence = detector.predict_relevance_with_confidence(input_ids)
        
        # Get detailed results
        _, details = detector.predict_relevance(input_ids, return_probabilities=True)
        
        print(f"Relevance prediction: {'Relevant' if predictions[0] == 0 else 'Irrelevant'}")
        print(f"Confidence: {confidence[0]:.3f}")
        print(f"Layer predictions: {details['layer_predictions']}")


if __name__ == "__main__":
    main() 