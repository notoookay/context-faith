import torch
from torch.utils.data import Dataset
from utils import format_prompt, LLAMA_3_1_INSTRUCT_PROMPT_FORMAT, QWEN_2_5_INSTRUCT_PROMPT_FORMAT


class RAGDataset(Dataset):
    def __init__(self, data, max_passages_per_item=-1, tokenizer=None, prompt_type="with_passage", n_samples=1, dataset_with_modified_passages=False, model_name=None):
        self.tasks = []
        self.prompt_type = prompt_type
        self.n_samples = n_samples
        self.dataset_with_modified_passages = dataset_with_modified_passages
        self.model_name = model_name
        self.use_llama_format = "llama-3.1" in model_name.lower() if model_name else False
        self.use_qwen_format = "qwen" in model_name.lower() if model_name else False
        self.is_chat_model = "it" in model_name.lower() or "instruct" in model_name.lower() if model_name else False
        
        # Process all items and create tasks
        for item_idx, item in enumerate(data):
            question = item['question']
            answers = item['answers']
            item_id = item.get('id') or str(item_idx)
            
            # For no passage variants, create just one task per question
            if prompt_type.startswith("no_passage"):
                
                if self.is_chat_model:
                    # Create prompt
                    native_prompt = format_prompt(question, prompt_type=prompt_type)
                    if self.use_llama_format:
                        prompt = LLAMA_3_1_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
                    elif self.use_qwen_format:
                        prompt = QWEN_2_5_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
                    else:
                        prompt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": native_prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                else:
                    native_prompt = format_prompt(question, prompt_type=prompt_type, base_model_prompt=True)
                    prompt = native_prompt
                
                # Add task
                for _ in range(n_samples):
                    self.tasks.append({
                        'item_id': item_id,
                        'question': question,
                        'answers': answers,
                        'passage': None,  # No passage for this mode
                        'native_prompt': native_prompt,
                        'prompt': prompt,
                        'passage_idx': -1  # No passage index
                })
            
            # For with passage variant, create task for each passage
            else:
                if 'ctxs' not in item or not item['ctxs']:
                    continue
                
                # Get passages
                passages = item['ctxs']
                if max_passages_per_item > 0:
                    passages = passages[:max_passages_per_item]
                
                # Create a task for each passage
                for passage_idx, passage in enumerate(passages):
                    
                    if self.is_chat_model:
                        # Create prompt
                        native_prompt = format_prompt(question, [passage], prompt_type=prompt_type, format_passages=(not self.dataset_with_modified_passages))
                        if self.use_llama_format:
                            prompt = LLAMA_3_1_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
                        elif self.use_qwen_format:
                            prompt = QWEN_2_5_INSTRUCT_PROMPT_FORMAT.format(prompt=native_prompt)
                        else:
                            prompt = tokenizer.apply_chat_template(
                                [{"role": "user", "content": native_prompt}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                            )
                    else:
                        native_prompt = format_prompt(question, [passage], prompt_type=prompt_type, format_passages=(not self.dataset_with_modified_passages), base_model_prompt=True)
                        prompt = native_prompt
                    
                    # Add task
                    self.tasks.append({
                        'item_id': item_id,
                        'question': question,
                        'answers': answers,
                        'passage': passage,
                        'native_prompt': native_prompt,
                        'prompt': prompt,
                        'passage_idx': passage_idx
                    })
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]


def collate_fn(batch, tokenizer, model_name):
    """
    Custom collate function to process and batch items together.
    
    Args:
        batch: List of items from the dataset
        tokenizer: Tokenizer to use for encoding
        model_name: Name of the model (used to determine prompt formatting)
    
    Returns:
        Dictionary with batched tensors and metadata
    """
    # Format prompts for the model
    prompts = [item['prompt'] for item in batch]
    
    # Tokenize all prompts in the batch
    encoded_inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,  # Don't add BOS/EOS tokens automatically
    )
    
    # Preserve metadata
    metadata = [{
        'item_id': item['item_id'],
        'question': item['question'],
        'answers': item['answers'],
        'passage': item['passage'],
        'prompt_length': len(encoded_inputs['input_ids'][i])
    } for i, item in enumerate(batch)]
    
    return {
        'input_ids': encoded_inputs['input_ids'],
        'attention_mask': encoded_inputs['attention_mask'],
        'metadata': metadata
    } 