import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, ActivationCache, utils
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
import argparse
import pickle

from utils import (
    format_prompt,
    LLAMA_3_1_INSTRUCT_PROMPT_FORMAT,
    QWEN_2_5_INSTRUCT_PROMPT_FORMAT,
    MODEL_ALIAS,
)
from tqdm import tqdm

class AACR:
    """
    Activation-Aware Context Routing implementation.
    
    Routes between external context and parametric knowledge based on:
    1. Context utility score (Ce) from trained linear probe
    2. Internal knowledge confidence (Ci) from Yes/No logit comparison
    """

    def __init__(
        self, 
        model_name: str,
        prober_path: str,
        optimal_layer: int,
        device: str = "cuda",
        context_threshold: float = 0.5,
        internal_threshold: float = 0.3,
        batch_size: int = 8
    ):
        """
        Initialize AACR system.
        
        Args:
            model_name: HuggingFace model name
            prober_path: Path to trained sklearn LogisticRegression classifier for context utility
            optimal_layer: Layer number where the prober was trained (l*)
            device: Device to run model on
            context_threshold: Threshold τ for context utility
            internal_threshold: Threshold θ for internal knowledge confidence
            batch_size: Default batch size for generation
        """
        self.device = device
        self.context_threshold = context_threshold
        self.optimal_layer = optimal_layer
        self.model_name = model_name
        self.internal_threshold = internal_threshold
        self.batch_size = batch_size

        # Load model using transformer_lens
        device = (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        self.device = device

        self.model = HookedTransformer.from_pretrained(
            model_name, 
            device=device, 
            dtype=torch.bfloat16,
            default_padding_side='left',
        )
        self.model.eval()

        # Get tokenizer from the model
        self.tokenizer = self.model.tokenizer

        # Load trained prober (sklearn LogisticRegression model)
        try:
            # Try loading with pickle first (for sklearn models)
            with open(prober_path, 'rb') as f:
                self.prober = pickle.load(f)
        except (pickle.UnpicklingError, UnicodeDecodeError):
            # Fallback to torch.load with weights_only=False for backward compatibility
            self.prober = torch.load(prober_path, map_location='cpu', weights_only=False)

        if not isinstance(self.prober, LogisticRegression):
            raise ValueError(f"Expected LogisticRegression model, but got {type(self.prober)}")

    def chat_format(self, prompt: str) -> str:
        if "llama" in self.model_name.lower():
            return LLAMA_3_1_INSTRUCT_PROMPT_FORMAT.format(prompt=prompt)
        elif "qwen" in self.model_name.lower():
            return QWEN_2_5_INSTRUCT_PROMPT_FORMAT.format(prompt=prompt)
        else:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

    def _get_context_utility_score(self, question: str, context: str) -> float:
        """
        Compute context utility score Ce using trained linear probe.
        
        Args:
            question: Input question
            context: Retrieved context/passages
            
        Returns:
            Context utility score between 0 and 1
        """
        # Format prompt for context utility assessment
        formatted_prompt = self.chat_format(format_prompt(question, [context], prompt_type="with_passage"))
        # Tokenize and get model activations using transformer_lens
        model_input_tokens = self.model.to_tokens(formatted_prompt, prepend_bos=False)

        with torch.no_grad():
            cache_act_name = utils.get_act_name("resid_post", self.optimal_layer)
            _, cache = self.model.run_with_cache(
                model_input_tokens,
                names_filter=cache_act_name,
                pos_slice=-1, # only get the last token
            )
            cache.to('cpu')

            # Get residual stream activations from the optimal layer
            # Use 'resid_post' to match get_model_activation.py
            hidden_states = cache[cache_act_name][0]  # Remove batch dim

            # Extract hidden state from last prompt token
            last_token_hidden = hidden_states[-1].float().numpy()  # [hidden_dim]

            # Use trained prober to get context utility score
            # Reshape to 2D array as expected by sklearn: [n_samples, n_features]
            features = last_token_hidden.reshape(1, -1)
            context_utility_score = self.prober.predict_proba(features)[0][1]  # Probability of class 1 (useful)

        return float(context_utility_score)

    def _get_internal_knowledge_confidence(self, question: str, internal_threshold: float) -> bool:
        """
        Assess internal knowledge confidence Ci using Yes/No logit comparison.
        
        Args:
            question: Input question
            
        Returns:
            True if model is confident in its internal knowledge, False otherwise
        """
        # Prompt for internal knowledge assessment
        formatted_prompt = self.chat_format(format_prompt(question, prompt_type="no_passage_knowledge_check"))

        # Tokenize using transformer_lens
        model_input_tokens = self.model.to_tokens(formatted_prompt, prepend_bos=False)

        with torch.no_grad():
            logits = self.model(model_input_tokens)[0, -1, :]  # Last token logits
            # logits.to('cpu')

            # Get logits for "Yes" and "No" tokens
            yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

            yes_logit = logits[yes_token_id].item()
            no_logit = logits[no_token_id].item()
            
            ratio = yes_logit / (yes_logit + no_logit)
            
            return ratio >= internal_threshold

    def batch_generate(self, formatted_prompts: list, batch_size: int = None) -> list:
        """
        Generate answers for all formatted prompts using efficient batch processing.
        
        Args:
            formatted_prompts: List of formatted prompts
            batch_size: Batch size for generation (uses batch_size if None)
            
        Returns:
            List of generated answers
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        answers = []
        
        # Process prompts in batches
        for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating answers"):
            batch_prompts = formatted_prompts[i:i+batch_size]
            
            input_ids = self.model.to_tokens(batch_prompts, prepend_bos=False)
            input_lengths = [len(tokens) for tokens in input_ids]
            
            with torch.no_grad():
                # Generate responses for the batch
                generated_tokens = self.model.generate(
                    input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    verbose=False,
                )
                
                # Process each generated sequence
                for j, (generated_seq, input_length) in enumerate(zip(generated_tokens, input_lengths)):
                    # Extract only the generated part (excluding input)
                    generated_part = generated_seq[input_length:]
                    
                    # Find EOS token to truncate if needed
                    eos_token_id = self.tokenizer.eos_token_id
                    if eos_token_id is not None:
                        eos_positions = (generated_part == eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos_positions) > 0:
                            generated_part = generated_part[:eos_positions[0]]
                    
                    # Decode the response
                    answer = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    answers.append(answer.strip())
        
        return answers

    def answer(self, question: str, context: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main AACR pipeline for answering a single question.
        Uses batch methods internally for consistency.
        
        Args:
            question: Input question
            context: Retrieved context/passages
            
        Returns:
            Tuple of (answer, decision_info) where decision_info contains:
            - context_utility_score: Ce score
            - used_context: Whether context was used
            - internal_confidence: Ci score (if context was insufficient)
            - decision: The routing decision made
        """
        # Use batch methods with single item for consistency
        results = self.batch_answer([question], [context], batch_size=self.batch_size)
        return results[0]

    def preprocess_decisions(self, questions: list, contexts: list) -> list:
        """
        Preprocess all question-context pairs to compute context utility scores 
        and internal knowledge confidence before batch generation.
        
        Args:
            questions: List of questions
            contexts: List of corresponding contexts
            
        Returns:
            List of decision dictionaries, each containing:
            - type: 'use_context', 'use_parametric_knowledge', or 'unable_to_answer'
            - context_utility_score: Computed Ce score
            - parametric_confidence: Computed Ci score (if applicable)
        """
        decisions = []
        for question, context in tqdm(zip(questions, contexts), desc="Getting decisions"):
            # Get context utility score
            context_utility_score = self._get_context_utility_score(question, context)

            if context_utility_score >= self.context_threshold:
                decision = {
                    'type': 'use_context',
                    'context_utility_score': context_utility_score,
                    'parametric_confidence': None
                }
            else:
                # Check parametric knowledge confidence
                parametric_confidence = self._get_internal_knowledge_confidence(question, internal_threshold=self.internal_threshold)
                if parametric_confidence:
                    decision = {
                        'type': 'use_parametric_knowledge',
                        'context_utility_score': context_utility_score,
                        'parametric_confidence': parametric_confidence
                    }
                else:
                    decision = {
                        'type': 'unable_to_answer',
                        'context_utility_score': context_utility_score,
                        'parametric_confidence': parametric_confidence
                    }
            decisions.append(decision)

        return decisions

    def batch_answer(self, questions: list, contexts: list, batch_size: int = None) -> list:
        """
        Efficiently process multiple question-context pairs.
        Uses preprocessing for decisions, then batch generation by decision groups.
        
        Args:
            questions: List of questions
            contexts: List of corresponding contexts
            batch_size: Batch size for generation (uses batch_size if None)
            
        Returns:
            List of (answer, decision_info) tuples
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Step 1: Preprocess decisions for all items
        decisions = self.preprocess_decisions(questions, contexts)

        # Step 2: Group items by decision type for efficient batch generation
        groups = {
            'use_context': [],
            'use_parametric_knowledge': [],
            'unable_to_answer': []
        }

        for i, decision in enumerate(decisions):
            groups[decision['type']].append(i)

        # Step 3: Generate answers in batches by decision type
        answers = [None] * len(questions)  # Placeholder for results

        # Generate for context-based items
        if groups['use_context']:
            context_prompts = []
            for i in groups['use_context']:
                prompt = self.chat_format(format_prompt(questions[i], [contexts[i]], prompt_type="with_passage_only"))
                context_prompts.append(prompt)

            context_answers = self.batch_generate(context_prompts, batch_size=batch_size)
            for idx, answer in zip(groups['use_context'], context_answers):
                answers[idx] = answer

        # Generate for parametric knowledge items
        if groups['use_parametric_knowledge']:
            parametric_prompts = []
            for i in groups['use_parametric_knowledge']:
                prompt = self.chat_format(format_prompt(questions[i], prompt_type="no_passage"))
                parametric_prompts.append(prompt)

            parametric_answers = self.batch_generate(parametric_prompts, batch_size=batch_size)
            for idx, answer in zip(groups['use_parametric_knowledge'], parametric_answers):
                answers[idx] = answer

        # Handle unable to answer items
        for i in groups['unable_to_answer']:
            answers[i] = "Unable to answer based on the provided context or current knowledge."

        # Step 4: Combine answers with decision info
        results = []
        for answer, decision in zip(answers, decisions):
            decision_info = {
                'context_utility_score': decision['context_utility_score'],
                'used_context': decision['type'] == 'use_context',
                'internal_confidence': decision.get('parametric_confidence'),
                'decision': decision['type']
            }
            results.append((answer, decision_info))

        return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prober_path", type=str, required=True)
    parser.add_argument("--optimal_layer", type=int, required=True)
    parser.add_argument("--context_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    return parser.parse_args()
