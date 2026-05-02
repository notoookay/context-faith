import torch
from transformers import AutoTokenizer
import argparse
import os
import json
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import wandb
from vllm import LLM, SamplingParams

from utils import save_file_jsonl


class ExactAnswerExtractor:
    """
    Extracts exact answers from model responses using VLLM for accelerated inference.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", batch_size=8):
        """
        Initialize the exact answer extractor with VLLM.
        
        Args:
            model_name: The model to use for extraction
            batch_size: Batch size for inference
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
        
        print(f"Loading extraction model with VLLM: {model_name}")
        try:
            self.llm = LLM(model=model_name, dtype="bfloat16")
            self.sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for more deterministic outputs
                max_tokens=100,    # Only need a short answer
                stop=None
            )
            print(f"VLLM extraction model loaded successfully")
        except Exception as e:
            print(f"Error loading model with VLLM: {e}")
            raise e
    
    def extract_single(self, question, model_answer):
        """
        Extract the exact answer from a single model response.
        
        Args:
            question: The question that was asked
            model_answer: The model's response to extract from
            
        Returns:
            Tuple of (exact_answer, valid) where valid is 1 if extraction succeeded
        """
        prompt = self._format_extraction_prompt(question, model_answer)
        
        # Generate response using VLLM
        outputs = self.llm.generate([prompt], self.sampling_params)
        exact_answer = self._process_output(outputs[0].outputs[0].text.strip(), model_answer)
        
        # Determine if extraction is valid
        valid = 0
        if exact_answer.lower() in model_answer.lower() or exact_answer == "NO ANSWER":
            valid = 1
        
        return exact_answer, valid
    
    def extract_batch(self, questions, model_answers, answers, acc):
        """
        Extract exact answers from a batch of model responses.
        
        Args:
            questions: List of questions
            model_answers: List of model answers
            answers: List of ground truth answers
            acc: List of model answer accuracy
            
        Returns:
            List of tuples (exact_answer, valid)
        """
        # Prepare all prompts for batch processing
        prompts = [self._format_extraction_prompt(q, a) for q, a in zip(questions, model_answers)]
        
        # Process batch with VLLM
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Process results
        results = []
        for i, output in enumerate(outputs):
            exact_answer = self._process_output(output.outputs[0].text.strip(), model_answers[i])
            
            # Determine if extraction is valid
            valid = 0
            if exact_answer.lower() in model_answers[i].lower():
                valid = 1
            # If extraction is invalid, try regenerating up to 5 more times
            else:
                max_retries = 5
                print(f"Retrying {max_retries} times for question {questions[i]}")
                # Generate 5 replies at once for efficiency
                retry_prompt = self._format_extraction_prompt(questions[i], model_answers[i])
                retry_outputs = self.llm.generate([retry_prompt] * max_retries, self.sampling_params)
                # Try each output until we find a valid one
                for retry_output in retry_outputs:
                        candidate_answer = self._process_output(retry_output.outputs[0].text.strip(), model_answers[i])
                        if candidate_answer.lower() in model_answers[i].lower():
                            exact_answer = candidate_answer
                            valid = 1
                            break
                if exact_answer == "NO ANSWER":
                    if acc[i] == 1:
                        answer = None
                        for answer in answers[i]:
                            if answer.lower() in model_answers[i].lower():
                                exact_answer = answer
                                valid = 1
                                break
                    else:
                        exact_answer = "NO ANSWER"
                        valid = 1

            
            results.append((exact_answer, valid))
        
        return results
    
    def _format_extraction_prompt(self, question, model_answer):
        """Format the prompt for extracting the exact answer."""
        raw_prompt = f"""
Extract from the following long answer the short answer, only the tokens in the long answer. If the long answer does not answer the question, output NO ANSWER.

Q: Which musical featured the song The Street Where You Live?
A: The song "The Street Where You Live" is from the Lerner and Loewe musical "My Fair Lady." It is one of the most famous songs from the show, and it is sung by Professor Henry Higgins as he reflects on the transformation of Eliza Doolittle and the memories they have shared together.
Exact answer: My Fair Lady

Q: Which Swedish actress won the Best Supporting Actress Oscar for Murder on the Orient Express?
A: I'm glad you asked about a Swedish actress who won an Oscar for "Murder on the Orient Express," but I must clarify that there seems to be a misunderstanding here. No Swedish actress has won an Oscar for Best Supporting Actress for that film. The 1974 "Murder on the Orient Express" was an American production, and the cast was predominantly British and American. If you have any other questions or if there's another
Exact answer: NO ANSWER

Q: {question}
A: {model_answer}
Exact answer:
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
                return prompt
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}. Using raw prompt.")
                return raw_prompt
        else:
            return raw_prompt
    
    def _process_output(self, output_text, model_answer):
        """Process the output text to extract the exact answer."""
        # Check if model answer contains "unable"
        if "unable" in model_answer.lower():
            return "unable"
            
        # Handle different model output formats
        if 'mistral' in self.model_name.lower():
            exact_answer = output_text.replace(".</s>", "").replace("</s>", "").split('\n')[0].split("(")[0].strip().strip(".")
        elif 'llama' in self.model_name.lower():
            exact_answer = output_text.replace(".<|eot_id|>", "").replace("<|eot_id|>", "").replace("Exact answer:", "").split('\n')[-1].split("(")[0].strip().strip(".")
        else:
            # Generic processing for other models
            exact_answer = output_text.strip().split('\n')[0].strip().split("(")[0].strip().strip(".")
        
        # Handle empty or too long answers
        if not exact_answer or len(exact_answer) > len(model_answer):
            exact_answer = "NO ANSWER"
        
        return exact_answer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract exact answers from generated responses")
    
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the model generation JSONL file")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to output JSONL file (if not specified, will be generated based on input file)")
    parser.add_argument("--output_dir", type=str, default="output/exact_answers", 
                        help="Directory to save output files")
    parser.add_argument("--extract_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                        help="Model to use for extraction (VLLM compatible)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for output files and Weights & Biases")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="extract_exact_answer", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                        help="Weights & Biases entity (username or team name)")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing passages across all items. Higher values improve throughput " 
                             "but require more GPU memory. Recommended values: 8-32 depending on model size and GPU memory.")
    parser.add_argument("--max_items", type=int, default=-1, 
                        help="Maximum number of items to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def main(args):
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set run name
    if args.run_name:
        run_name = args.run_name
    else:
        input_file_base = os.path.basename(args.input_file).split('.')[0]
        run_name = f"extract_{input_file_base}_{args.extract_model.split('/')[-1].replace('.', '-')}"
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )
        
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_dataset('json', data_files=args.input_file, split='train')
    print(f"Loaded {len(data)} items.")
    
    # Limit the number of items if specified
    if args.max_items > 0:
        data = data.select(range(min(args.max_items, len(data))))
        print(f"Limited to {len(data)} items.")
    
    # Convert to list
    results = [item for item in data]
    
    # Initialize extractor
    print(f"Initializing extractor with model {args.extract_model}...")
    extractor = ExactAnswerExtractor(
        model_name=args.extract_model,
        batch_size=args.batch_size
    )
    
    # Process and extract exact answers
    print("Extracting exact answers...")
    
    # Prepare mapping to track original items and passages
    passage_map = {}  # Maps passage ID to (item_idx, passage_idx)
    reverse_passage_map = {}  # Maps (item_idx, passage_idx) to passage_id
    passage_id_counter = 0
    
    all_questions = []
    all_responses = []
    all_passage_ids = []
    all_acc = []
    all_answers = []

    for item_idx, item in enumerate(results):
        for passage_idx, passage in enumerate(item['ctxs']):
            all_questions.append(item['question'])
            all_responses.append(passage['response'])
            all_answers.append(item['answers'])
            all_acc.append(passage['accuracy'])
            # Generate a unique ID for this passage
            passage_id = passage_id_counter
            passage_map[passage_id] = (item_idx, passage_idx)
            reverse_passage_map[(item_idx, passage_idx)] = passage_id
            all_passage_ids.append(passage_id)
            passage_id_counter += 1
    
    total_passages = len(all_passage_ids)
    print(f"Total passages to process: {total_passages}")
    
    # Process in optimized batches across items
    extraction_results = {}  # Map of passage_id -> (exact_answer, valid)
    valid_extractions = 0
    no_answer_count = 0
    
    # Process in batches
    for i in tqdm(range(0, len(all_questions), args.batch_size), desc="Processing batches"):
        batch_questions = all_questions[i:i+args.batch_size]
        batch_responses = all_responses[i:i+args.batch_size]
        batch_answers = all_answers[i:i+args.batch_size]
        batch_passage_ids = all_passage_ids[i:i+args.batch_size]
        batch_acc = all_acc[i:i+args.batch_size]
        # Process the batch
        batch_results = extractor.extract_batch(
            questions=batch_questions,
            model_answers=batch_responses,
            answers=batch_answers,
            acc=batch_acc
        )
        
        # Store results with their passage IDs
        for j, (exact_answer, valid) in enumerate(batch_results):
            passage_id = batch_passage_ids[j]
            extraction_results[passage_id] = (exact_answer, valid)
            
            if exact_answer == "NO ANSWER":
                no_answer_count += 1
            if valid == 1:
                valid_extractions += 1
    
    # Now reconstruct the original structure with extracted answers
    processed_results = []
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
            exact_answer, valid = extraction_results[passage_id]
            
            # Create a copy of the passage with exact answer added
            passage_result = passage.copy()
            passage_result['exact_answer'] = exact_answer
            passage_result['extraction_valid'] = valid
            
            item_result['ctxs'].append(passage_result)
        
        processed_results.append(item_result)
    
    # Save results
    if args.output_file:
        output_file = args.output_file
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{run_name}.jsonl")
    
    print(f"\nSaving extracted answers to {output_file}...")
    save_file_jsonl(processed_results, output_file)
    
    # Log summary statistics
    if args.use_wandb:
        wandb.log({
            "valid_extractions_percentage": valid_extractions / total_passages if total_passages > 0 else 0,
            "no_answer_percentage": no_answer_count / total_passages if total_passages > 0 else 0,
            "total_passages": total_passages,
            "valid_extractions": valid_extractions,
            "no_answer_count": no_answer_count,
        })
    
    # Print summary
    print("\n=== Extraction Summary ===")
    print(f"Total items processed: {len(processed_results)}")
    print(f"Total passages/examples: {total_passages}")
    print(f"Valid extractions: {valid_extractions} ({valid_extractions/total_passages*100:.2f}% of total)")
    print(f"NO ANSWER responses: {no_answer_count} ({no_answer_count/total_passages*100:.2f}% of total)")
    
    print("\nExtraction complete!")
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    return output_file


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
