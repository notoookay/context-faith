import json
import argparse
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import numpy as np
from datasets import load_dataset
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from vllm import LLM, SamplingParams

from utils import load_jsonlines, save_file_jsonl


def evaluate_answer(predicted, labels, llm_evaluator=None, question=None):
    """
    Evaluate the predicted answer.
    
    Args:
        predicted: The model's predicted answer
        labels: The ground truth label(s)
        llm_evaluator: Optional LLM evaluator instance
        question: The question being evaluated (needed for LLM evaluation)
    
    Returns:
        Dictionary with evaluation results
    """
    if llm_evaluator is not None and question is not None:
        # Use LLM-based evaluation
        accuracy = llm_evaluator.evaluate(question, predicted, labels)
    else:
        # Use string matching (default approach)
        if isinstance(labels, str):
            labels = [labels]
        
        accuracy = max([1 if label.lower() in predicted.lower() else 0 for label in labels])
    
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
        for passage_result in item['ctxs']:
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
        # Check if we have both relevant and irrelevant passages
        if df['has_answer'].nunique() > 1:
            relevant_scores = df[df['has_answer']][metric].mean()
            irrelevant_scores = df[~df['has_answer']][metric].mean()
            
            plt.subplot(1, 2, i+1)
            plt.bar(['Relevant', 'Irrelevant'], [relevant_scores, irrelevant_scores], color=['green', 'red'])
            plt.title(f'Average {metric.replace("_", " ").title()}')
            plt.ylabel('Score')
            plt.ylim(0, 1)
        else:
            # Only one type of passage (e.g., all relevant)
            scores = df[metric].mean()
            plt.subplot(1, 2, i+1)
            plt.bar(['All Passages'], [scores], color=['blue'])
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
    
    # ROC curve only if we have both relevant and irrelevant passages
    if df['has_answer'].nunique() > 1:
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


class LLMEvaluator:
    """
    LLM-based evaluator that uses another language model to judge the correctness
    of answers, similar to compute_correctness_natual_questions.
    
    Uses VLLM for accelerated inference if available.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", batch_size=4):
        """
        Initialize the LLM evaluator.
        
        Args:
            model_name: The model to use for evaluation
            batch_size: Batch size for evaluation (used with VLLM)
        """
        self.batch_size = batch_size
        self.model_name = model_name
        
        # Determine if this is a chat model (contains 'instruct' in name)
        self.is_chat_model = 'instruct' in model_name.lower()
        print(f"Using chat model: {self.is_chat_model}")
        
        # Load tokenizer for chat formatting if needed
        if self.is_chat_model:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"Loaded tokenizer for chat formatting")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self.tokenizer = None
                self.is_chat_model = False  # Fall back to non-chat mode
        else:
            self.tokenizer = None
        
        print(f"Loading LLM evaluator model with VLLM: {model_name}")
        try:
            self.llm = LLM(model=model_name, dtype="float16")
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=100,
                stop=None
            )
            print(f"VLLM LLM evaluator model loaded successfully")
        except Exception as e:
            print(f"Error loading model with VLLM: {e}")
            raise e
        
    def evaluate(self, question, model_answer, ground_truth):
        """
        Evaluate a single question-answer pair using the LLM.
        
        Args:
            question: The question that was asked
            model_answer: The model's answer
            ground_truth: The ground truth answer(s)
            
        Returns:
            1 if correct, 0 if incorrect
        """
        # Convert ground_truth to string if it's a list
        if isinstance(ground_truth, list):
            ground_truth = ", ".join(ground_truth)
            
        # Format the evaluation prompt
        raw_prompt = f"""
Evaluate the following answer to a question. You will be given a question, a model answer, and the correct answer.
Determine if the model answer is correct or not according to the given correct answer. If the model answer is correct, write '1', otherwise write '0'.
If the model answer contains any incorrect information or contradicts the correct answer, write '0'. Even if parts of the answer are correct, the presence of any incorrect information means the entire answer should be marked as incorrect (0).

Examples:
Question: who is the young guitarist who played with buddy guy?
Ground Truth: Quinn Sullivan
Model Answer: Ronnie Earl Explanation: Ronnie Earl is an American blues guitarist and singer who has played with many famous blues musicians, including Buddy Guy.
Correctness: 0

Question: name of the first episode of stranger things 
Ground Truth: Chapter One : The Vanishing of Will Byers
Model Answer: The disappearance of Will Byers. Explanation: The first episode of the first season of Stranger Things is titled "The Vanishing of Will Byers".
Correctness: 1

Now evaluate this answer:
Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}
Correctness:
"""
        
        # Apply chat template if this is a chat model
        if self.is_chat_model and self.tokenizer is not None:
            try:
                messages = [{"role": "user", "content": raw_prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}. Using raw prompt.")
                prompt = raw_prompt
        else:
            prompt = raw_prompt
        
        # Generate response using VLLM
        outputs = self.llm.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Look for "1" or "0" in the response
        index_of_1 = generated_text.find('1')
        index_of_0 = generated_text.find('0')
        
        if index_of_1 != -1 and (index_of_0 == -1 or index_of_1 < index_of_0):
            return 1
        elif index_of_0 != -1 and (index_of_1 == -1 or index_of_0 < index_of_1):
            return 0
        
        # If no clear answer in first try, retry a couple times
        retries = 1
        while retries < 3:
            # Add a more direct prompt for retry
            retry_raw_prompt = raw_prompt + "\nPlease only respond with 0 or 1: "
            
            # Apply chat template if this is a chat model
            if self.is_chat_model and self.tokenizer is not None:
                try:
                    messages = [{"role": "user", "content": retry_raw_prompt}]
                    retry_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    retry_prompt = retry_raw_prompt
            else:
                retry_prompt = retry_raw_prompt
                
            outputs = self.llm.generate([retry_prompt], self.sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            
            index_of_1 = generated_text.find('1')
            index_of_0 = generated_text.find('0')
            
            if index_of_1 != -1 and (index_of_0 == -1 or index_of_1 < index_of_0):
                return 1
            elif index_of_0 != -1 and (index_of_1 == -1 or index_of_0 < index_of_1):
                return 0
                
            retries += 1
        
        # Default to 0 if we couldn't get a valid response
        return 0
    
    def batch_evaluate(self, questions, model_answers, ground_truths):
        """
        Evaluate a batch of question-answer pairs.
        
        Args:
            questions: List of questions
            model_answers: List of model answers
            ground_truths: List of ground truth answers
            
        Returns:
            List of 1s and 0s indicating correctness
        """
        # Prepare all prompts for batch processing
        prompts = []
        for q, a, g in zip(questions, model_answers, ground_truths):
            if isinstance(g, list):
                g = ", ".join(g)
                
            raw_prompt = f"""
Evaluate the following answer to a question. You will be given a question, a model answer, and the correct answer.
Determine if the model answer is correct or not according to the given correct answer. If the model answer is correct, write '1', otherwise write '0'.
If the model answer contains any incorrect information or contradicts the correct answer, write '0'. Even if parts of the answer are correct, the presence of any incorrect information means the entire answer should be marked as incorrect (0).

Examples:
Question: who is the young guitarist who played with buddy guy?
Ground Truth: Quinn Sullivan
Model Answer: Ronnie Earl Explanation: Ronnie Earl is an American blues guitarist and singer who has played with many famous blues musicians, including Buddy Guy.
Correctness: 0

Question: name of the first episode of stranger things 
Ground Truth: Chapter One : The Vanishing of Will Byers
Model Answer: The disappearance of Will Byers. Explanation: The first episode of the first season of Stranger Things is titled "The Vanishing of Will Byers".
Correctness: 1

Now evaluate this answer:
Question: {q}
Ground Truth: {g}
Model Answer: {a}
Correctness:
"""
            # Apply chat template if this is a chat model
            if self.is_chat_model and self.tokenizer is not None:
                try:
                    messages = [{"role": "user", "content": raw_prompt}]
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    prompt = raw_prompt
            else:
                prompt = raw_prompt
                
            prompts.append(prompt)
            
        # Process batch with VLLM
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Extract results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            
            index_of_1 = generated_text.find('1')
            index_of_0 = generated_text.find('0')
            
            if index_of_1 != -1 and (index_of_0 == -1 or index_of_1 < index_of_0):
                results.append(1)
            elif index_of_0 != -1 and (index_of_1 == -1 or index_of_0 < index_of_1):
                results.append(0)
            else:
                results.append(0)  # Default to 0 if no clear answer
                
        return results
    
    def evaluate_context_utility(self, question, context, model_answer):
        """
        Evaluate whether a given context is useful for answering a question.
        
        Args:
            question: The question that needs to be answered
            context: The context/passage to evaluate
            model_answer: The model's answer
            
        Returns:
            1 if context is useful, 0 if not useful
        """
        # Format the evaluation prompt for context utility
        raw_prompt = f"""
Evaluate whether the model thinks the provided context is useful for answering the question. You will be given a question, the context provided to the model, and the model's response. Determine if the model considers the context is useful based on its response language and behavior.

Write '1' if the model thinks the context is useful for answering (shows confidence in using the context), otherwise write '0' if the model thinks the context is not useful or insufficient. If the model gives the answer directly, we think it uses context information, so write '1'.

Key indicators:
- Context USEFUL (1): Model clearly states that the answer can be found in the context, gives confident answers citing the context
- Context NOT USEFUL (0): Model states context doesn't contain the answer, admits context is insufficient, or explicitly states uncertainty about context relevance

Examples:

Question: What is the capital of France?
Context: France is a beautiful country in Western Europe known for its cuisine and culture.
Model Answer: While the provided context mentions France, it doesn't explicitly state the capital. However, I know the capital of France is Paris.
Context Utility Assessment: 0

Question: Who invented the telephone?
Context: Alexander Graham Bell was a Scottish-American inventor who is credited with inventing and patenting the first practical telephone.
Model Answer: According to the context provided, Alexander Graham Bell invented the telephone. He was a Scottish-American inventor who is credited with inventing and patenting the first practical telephone.
Context Utility Assessment: 1

Question: What year was the iPhone first released?
Context: The smartphone market has grown significantly over the past decade.
Model Answer: The provided context doesn't contain information about when the iPhone was first released. Based on my knowledge, the iPhone was first released in 2007.
Context Utility Assessment: 0

Now evaluate this response:

Question: {question}
Context: {context}
Model Answer: {model_answer}
Context Utility Assessment:
"""
        
        # Apply chat template if this is a chat model
        if self.is_chat_model and self.tokenizer is not None:
            try:
                messages = [{"role": "user", "content": raw_prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}. Using raw prompt.")
                prompt = raw_prompt
        else:
            prompt = raw_prompt
        
        # Generate response using VLLM
        outputs = self.llm.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Look for "1" or "0" in the response
        index_of_1 = generated_text.find('1')
        index_of_0 = generated_text.find('0')
        
        if index_of_1 != -1 and (index_of_0 == -1 or index_of_1 < index_of_0):
            return 1
        elif index_of_0 != -1 and (index_of_1 == -1 or index_of_0 < index_of_1):
            return 0
        
        # If no clear answer in first try, retry a couple times
        retries = 1
        while retries < 3:
            # Add a more direct prompt for retry
            retry_raw_prompt = raw_prompt + "\nPlease only respond with 0 or 1: "
            
            # Apply chat template if this is a chat model
            if self.is_chat_model and self.tokenizer is not None:
                try:
                    messages = [{"role": "user", "content": retry_raw_prompt}]
                    retry_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    retry_prompt = retry_raw_prompt
            else:
                retry_prompt = retry_raw_prompt
                
            outputs = self.llm.generate([retry_prompt], self.sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            
            index_of_1 = generated_text.find('1')
            index_of_0 = generated_text.find('0')
            
            if index_of_1 != -1 and (index_of_0 == -1 or index_of_1 < index_of_0):
                return 1
            elif index_of_0 != -1 and (index_of_1 == -1 or index_of_0 < index_of_1):
                return 0
                
            retries += 1
        
        # Default to 0 if we couldn't get a valid response
        return 0
    
    def batch_evaluate_context_utility(self, questions, contexts, model_answers):
        """
        Evaluate a batch of context utility assessments.
        
        Args:
            questions: List of questions
            contexts: List of context passages to evaluate
            ground_truths: Optional list of ground truth answers for reference
            
        Returns:
            List of 1s and 0s indicating context utility
        """
        # Prepare all prompts for batch processing
        prompts = []
        for q, c, a in zip(questions, contexts, model_answers):
            # Include ground truth in prompt if provided
            raw_prompt = f"""
Evaluate whether the model thinks the provided context is useful for answering the question. You will be given a question, the context provided to the model, and the model's response. Determine if the model considers the context is useful based on its response language and behavior.

Write '1' if the model thinks the context is useful for answering (shows confidence in using the context), otherwise write '0' if the model thinks the context is not useful or insufficient. If the model gives the answer directly, we think it uses context information, so write '1'.

Key indicators:
- Context USEFUL (1): Model clearly states that the answer can be found in the context, gives confident answers citing the context
- Context NOT USEFUL (0): Model states context doesn't contain the answer, admits context is insufficient, or explicitly states uncertainty about context relevance

Examples:

Question: What is the capital of France?
Context: France is a beautiful country in Western Europe known for its cuisine and culture.
Model Answer: While the provided context mentions France, it doesn't explicitly state the capital. However, I know the capital of France is Paris.
Context Utility Assessment: 0

Question: Who invented the telephone?
Context: Alexander Graham Bell was a Scottish-American inventor who is credited with inventing and patenting the first practical telephone.
Model Answer: According to the context provided, Alexander Graham Bell invented the telephone. He was a Scottish-American inventor who is credited with inventing and patenting the first practical telephone.
Context Utility Assessment: 1

Question: What year was the iPhone first released?
Context: The smartphone market has grown significantly over the past decade.
Model Answer: The provided context doesn't contain information about when the iPhone was first released. Based on my knowledge, the iPhone was first released in 2007.
Context Utility Assessment: 0

Now evaluate this response:

Question: {q}
Context: {c}
Model Answer: {a}
Context Utility Assessment:
"""
            # Apply chat template if this is a chat model
            if self.is_chat_model and self.tokenizer is not None:
                try:
                    messages = [{"role": "user", "content": raw_prompt}]
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    prompt = raw_prompt
            else:
                prompt = raw_prompt
                
            prompts.append(prompt)
            
        # Process batch with VLLM
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Extract results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            
            index_of_1 = generated_text.find('1')
            index_of_0 = generated_text.find('0')
            
            if index_of_1 != -1 and (index_of_0 == -1 or index_of_1 < index_of_0):
                results.append(1)
            elif index_of_0 != -1 and (index_of_1 == -1 or index_of_0 < index_of_1):
                results.append(0)
            else:
                results.append(0)  # Default to 0 if no clear answer
                
        return results


def main(args):
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Load generated responses
    print(f"Loading generated responses from {args.input_file}...")
    data = load_dataset('json', data_files=args.input_file, split='train')
    print(f"Loaded {len(data)} items with generated responses.")
    
    # Convert to list
    results = [item for item in data]
    
    # Extract generation metadata
    generation_metadata = results[0].get('generation_metadata', {})
    
    # Initialize wandb if requested
    if args.run_name:
        run_name = args.run_name
    else:
        input_file_base = os.path.basename(args.input_file).split('.')[0]
        eval_method = "llm" if args.use_llm_eval else "string"
        run_name = f"{input_file_base}_{eval_method}_eval"
    
    if args.use_wandb:
        wandb.init(
            project="generation-evaluation", 
            name=run_name,
            config={
                "input_file": args.input_file,
                "evaluation_method": "llm" if args.use_llm_eval else "string_match",
                "llm_eval_model": args.llm_eval_model if args.use_llm_eval else None,
            }
        )
    
    # Initialize LLM evaluator if specified
    llm_evaluator = None
    if args.use_llm_eval:
        print(f"Using LLM-based evaluation with model: {args.llm_eval_model}")
        try:
            llm_evaluator = LLMEvaluator(
                model_name=args.llm_eval_model,
                batch_size=args.batch_size
            )
        except Exception as e:
            print(f"Error initializing LLM evaluator: {e}")
            print("Falling back to string matching")
            args.use_llm_eval = False
        
    # Evaluate responses
    print("Evaluating responses...")
    
    # Store evaluation results
    evaluated_results = []
    
    # Process in batches if using LLM evaluation
    if args.use_llm_eval and args.batch_eval:
        # Prepare mapping to track original items and passages
        passage_map = {}  # Maps passage ID to (item_idx, passage_idx)
        reverse_passage_map = {}  # Maps (item_idx, passage_idx) to passage_id
        passage_id_counter = 0
        
        # Collect all passages from all items for efficient batch processing
        # This cross-item batching strategy ensures we fully utilize GPU resources
        # regardless of how many passages each individual item has
        all_questions = []
        all_responses = []
        all_ground_truths = []
        all_passage_ids = []
        
        # Collect all passages across items
        for item_idx, item in enumerate(results):
            for passage_idx, passage in enumerate(item['ctxs']):
                all_questions.append(item['question'])
                all_responses.append(passage['response'])
                all_ground_truths.append(item['answers'])
                
                # Generate a unique ID for this passage
                passage_id = passage_id_counter
                passage_map[passage_id] = (item_idx, passage_idx)
                reverse_passage_map[(item_idx, passage_idx)] = passage_id
                all_passage_ids.append(passage_id)
                passage_id_counter += 1
        
        total_passages = len(all_passage_ids)
        print(f"Total passages to evaluate: {total_passages}")
        
        # Process in optimized batches across items
        evaluation_results = {}  # Map of passage_id -> accuracy score
        
        # Process in batches
        for i in tqdm(range(0, len(all_questions), args.batch_size), desc="Processing batches"):
            batch_questions = all_questions[i:i+args.batch_size]
            batch_responses = all_responses[i:i+args.batch_size]
            batch_ground_truths = all_ground_truths[i:i+args.batch_size]
            batch_passage_ids = all_passage_ids[i:i+args.batch_size]
            
            # Evaluate batch
            batch_accuracy_scores = llm_evaluator.batch_evaluate(
                batch_questions,
                batch_responses,
                batch_ground_truths
            )
            
            # Store results with their passage IDs
            for j, accuracy in enumerate(batch_accuracy_scores):
                passage_id = batch_passage_ids[j]
                evaluation_results[passage_id] = accuracy
        
        # Now reconstruct the original structure with evaluation results
        for item_idx, item in enumerate(results):
            item_result = {
                'id': item['id'],
                'question': item['question'],
                'answers': item['answers'],
                'ctxs': []
            }
            
            # Add results to each passage
            for passage_idx, passage in enumerate(item['ctxs']):
                # Get the passage ID directly from the reverse mapping
                passage_id = reverse_passage_map[(item_idx, passage_idx)]
                accuracy = evaluation_results[passage_id]
                
                # Create a copy of the passage with evaluation added
                passage_result = passage.copy()
                passage_result['evaluation'] = {'acc': accuracy}
                
                item_result['ctxs'].append(passage_result)
            
            evaluated_results.append(item_result)
    else:
        # Process each item individually
        for item_data in tqdm(results, desc="Evaluating items"):
            item_results = {
                'id': item_data['id'],
                'question': item_data['question'],
                'answers': item_data['answers'],
                'ctxs': []
            }
            
            # Evaluate each passage
            for passage in item_data['ctxs']:
                if args.use_llm_eval:
                    evaluation = evaluate_answer(
                        passage['response'], 
                        item_data['answers'],
                        llm_evaluator=llm_evaluator,
                        question=item_data['question']
                    )
                else:
                    evaluation = evaluate_answer(passage['response'], item_data['answers'])
                
                passage_result = passage.copy()
                passage_result['evaluation'] = evaluation
                
                item_results['ctxs'].append(passage_result)
                
            evaluated_results.append(item_results)
    
    # Calculate and print statistics
    print("\n=== Results Summary ===")
    print(f"Total items processed: {len(evaluated_results)}")
    
    if args.use_llm_eval:
        print(f"Evaluation method: LLM-based using {args.llm_eval_model}")
    else:
        print("Evaluation method: String matching")

    # Gather statistics
    total_passages = sum(len(item['ctxs']) for item in evaluated_results)

    # Check if we're using passage-based evaluation
    is_passage_based = False
    if evaluated_results and evaluated_results[0]['ctxs']:
        is_passage_based = 'has_answer' in evaluated_results[0]['ctxs'][0]
    
    # If we're using no_passage mode or don't have passage info
    if not is_passage_based:
        print(f"Total examples: {total_passages}")
        
        # Calculate overall accuracy
        total_correct = sum(sum(1 for p in item['ctxs'] if p['evaluation']['acc']) for item in evaluated_results)
        print(f"Overall accuracy: {total_correct} ({total_correct/total_passages*100:.2f}%)")
        
        # Log summary metrics to wandb
        if args.use_wandb:
            wandb.log({
                "total_examples": total_passages,
                "accuracy": total_correct/total_passages if total_passages > 0 else 0,
            })
            
        # Save a simple summary
        summary = {
            'total_items': len(evaluated_results),
            'total_examples': total_passages,
            'correct_answers': total_correct,
            'accuracy': total_correct/total_passages if total_passages > 0 else 0,
            'evaluation_method': 'llm' if args.use_llm_eval else 'string_match',
            'llm_eval_model': args.llm_eval_model if args.use_llm_eval else None,
        }
    else:
        # Statistics for passage-based evaluation
        passages_with_answer = sum(sum(1 for p in item['ctxs'] if p['has_answer']) for item in evaluated_results)
        passages_without_answer = total_passages - passages_with_answer

        print(f"Total passages: {total_passages}")
        print(f"Passages with answer: {passages_with_answer} ({passages_with_answer/total_passages*100:.2f}%)")
        print(f"Passages without answer: {passages_without_answer} ({passages_without_answer/total_passages*100:.2f}%)")

        # Calculate metrics
        acc_relevant = sum(sum(1 for p in item['ctxs'] if p['has_answer'] and p['evaluation']['acc']) for item in evaluated_results)
        
        if passages_without_answer > 0:
            acc_irrelevant = sum(sum(1 for p in item['ctxs'] if not p['has_answer'] and p['evaluation']['acc']) for item in evaluated_results)
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
            'total_items': len(evaluated_results),
            'total_passages': total_passages,
            'passages_with_answer': passages_with_answer,
            'passages_without_answer': passages_without_answer,
            'acc_relevant': acc_relevant,
            'acc_irrelevant': acc_irrelevant,
            'evaluation_method': 'llm' if args.use_llm_eval else 'string_match',
            'llm_eval_model': args.llm_eval_model if args.use_llm_eval else None,
        }
        
    # Save results
    if args.output_file:
        output_file = args.output_file
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{run_name}.jsonl")
    
    print(f"\nSaving evaluation results to {output_file}...")
    save_file_jsonl(evaluated_results, output_file)

    # Create visualizations
    if args.create_plots:
        print("Creating visualizations...")
        create_visualizations(evaluated_results, os.path.dirname(output_file), args, run_name=run_name)

    # Save summary
    summary_file = os.path.splitext(output_file)[0] + '_summary.jsonl'
    save_file_jsonl([summary], summary_file)

    # Log summary file to wandb if enabled
    if args.use_wandb:
        wandb.save(summary_file)

    print(f"Saved summary to {summary_file}")
    print("\nEvaluation complete!")

    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated responses")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Path to generated responses JSONL file from model_generate.py")
    parser.add_argument("--output_file", type=str, default=None, 
                       help="Path to output evaluated JSONL file (if not specified, will be saved in output_dir)")
    parser.add_argument("--output_dir", type=str, default="output/generation_evaluation", 
                       help="Directory to save output files")
    parser.add_argument("--run_name", type=str, default=None, 
                       help="Run name for output files and Weights & Biases")
    parser.add_argument("--create_plots", action="store_true", 
                       help="Create visualizations")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Enable Weights & Biases logging")
    
    # LLM evaluation arguments
    parser.add_argument("--use_llm_eval", action="store_true", 
                       help="Use LLM-based evaluation instead of string matching")
    parser.add_argument("--llm_eval_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Model to use for LLM-based evaluation")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for LLM evaluation across all items. Higher values improve throughput " 
                            "but require more GPU memory. Recommended values: 8-32 depending on model size and GPU memory.")
    parser.add_argument("--batch_eval", action="store_true", 
                       help="Perform LLM evaluation in batches across all items. This is more efficient when evaluating many passages.")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    main(args) 