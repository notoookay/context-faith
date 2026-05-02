import jsonlines
import json


MODEL_ALIAS = {
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}

SAE_MAP = {
    "gemma-2-2b": "gemma-scope-2b-pt-res-canonical",
    "gemma-2-2b-it": "gemma-scope-2b-pt-res-canonical",
    "gemma-2-9b": "gemma-scope-9b-pt-res-canonical",
    "gemma-2-9b-it": "gemma-scope-9b-pt-res-canonical",
    "llama-3.1-8b-instruct": "llama_scope_lxr_8x",
    "llama-3.1-8b": "llama_scope_lxr_8x",
}

NEURONPEDIA_MODEL_ID = {
    "gemma-2-2b": "gemma-2-2b",
    "gemma-2-2b-it": "gemma-2-2b",
    "gemma-2-9b": "gemma-2-9b",
    "gemma-2-9b-it": "gemma-2-9b",
    "llama-3.1-8b": "llama3.1-8b",
    "llama-3.1-8b-instruct": "llama3.1-8b"
}

NEURONPEDIA_DATASET_NAME = {
    "gemma-2-2b": "gemmascope-res-16k",
    "gemma-2-2b-it": "gemmascope-res-16k",
    "gemma-2-9b": "gemmascope-res-16k",
    "gemma-2-9b-it": "gemmascope-res-16k",
    "llama-3.1-8b": "llamascope-res-8k",
    "llama-3.1-8b-instruct": "llamascope-res-8k",
}

PROMPT_TEMPLATE_RELEVANCE_CHECKING = """
Given the following question and context, determine if the context is relevant for answering the question.

Question:
{question}

Context:
{passages}

Is the context relevant? Give your reasoning and respond with "Yes" or "No".
"""

PROMPT_TEMPLATE = """
Answer the following question:

{question}

The following context may be relevant or irrelevant to the question:

{passages}
"""

PROMPT_TEMPLATE_KNOWLEDGE_CHOOSING = """
Answer the following question based on the given context or your own knowledge:

{question}

You may use the following context:

{passages}
"""

PROMPT_PARAMETRIC_CHECK_TEMPLATE = """
Answer the following question:

{question}

Please provide a complete and accurate response based on your knowledge. If you don't have sufficient information to provide a reliable answer, please reply with: "Unable to answer based on my current knowledge."
"""

PROMPT_NO_PASSAGE_TEMPLATE = """
Answer the following question:

{question}
"""

BASE_MODEL_PROMPT_TEMPLATE = """
Answer the following question:

Context:
{passages}

Question:
{question}

Answer:
"""

# Using HuggingFace tokenizer would add system prompt with date and addtional information
# So we strip the system prompt manually
LLAMA_3_1_INSTRUCT_PROMPT_FORMAT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
QWEN_2_5_INSTRUCT_PROMPT_FORMAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

_refusal_prefixes = [
    "there are several",
    "there are multiple",
    "there isn't a single ",
    "it's difficult to ",
    "it is difficult to",
    "unfortunately",
]

_refusal_substrings = [
    "i'm unable",
    "i am unable",
    "sorry",
    "not familiar",
    "couldn't find",
    "could not find",
    "any information",
    "not aware"
    "not familiar",
    "not sure",
    "it's impossible to give",
    "there doesn't seem to be any",
    "this is a complex ",
    "i cannnot ",
    "i do not have access to",
    "there's no single definitive answer",
    "be more specific",
    "provide more context",
    "provide me with more"
    "there doesn't seem to be any",
    "this is a complex ",
    "i cannnot ",
    "i do not have access to",
    "there's no single definitive answer",
    "i cannot answer this question",
    "the premise of the question is incorrect",
    "unable to answer",
]


_unknown_strings = [
    "there is no widely known",
    "is not a known",
    "doesn't seem to be a widely known",
    "there's no widely known",
    "There doesn't seem to be a",
    "There is no ",
    "fictional",
    "unfortunately,",
    "doesn't exist",
    "there is no ",
    "doesn't seem to be a known",
    "is a bit of a tricky one!",
    "a bit of a trick",
    "not yet available",
    "there doesn't appear to be a ",
    "please provide me with",
    "i cannot give you an",
    "there's no widely known",
    "it's not a widely known",
    "no publicly available information",
    "not a very well-known one",
    "please clarify",
    "i do not have access to real-time",
    "i'm unable to access real-time",
    "i am unable to access real-time",
    "unable to answer",
    "i believe you",
    "i need a bit more information",
    "i couldn't find any information",
    "i need a little more",
    "i can't give you a specific",
    "i can't find any information",
    "i'm unable ",
    "i am unable",
    "i'm not able",
    "i am not able",
    "i cannot find",
    "there is no evidence",
    "the context does not specify",
    "the context does not provide",
    "is not explicitly stated",
    "not mentioned in the context",
    "not provided",
    "unknown",
    "not publicly available information",
    "please note",
    "does not exist",
    "it's important to note that",
    "not readily available",
    "not a known",
    "do not have information",
    "not aware of",
    "i need more information"
]

def is_generation_refusal(generation: str):
    generation = generation.lower()
    return any(substring in generation for substring in _unknown_strings) or any(substring in generation for substring in _refusal_substrings) or any(generation.startswith(prefix.lower()) for prefix in _test_prefixes_llm_attacks) or any(generation.startswith(prefix.lower()) for prefix in _refusal_prefixes)

def format_passage(title, text):
    # Passages should contain the title and text for each passage
    return f"(Title: {title}) {text}"

def format_prompt(question, passage_list=None, prompt_type="with_passage", format_passages=True, base_model_prompt=False):
    """
    Format a prompt based on the specified type.
    
    Args:
        question: The question to answer
        passage_list: List of passages (only used if prompt_type is "with_passage")
        prompt_type: Type of prompt - "with_passage", "with_passage_no_refuse", "no_passage", or "no_passage_no_refuse"
        
    Returns:
        Formatted prompt string
    """
    if prompt_type.startswith("with_passage") and passage_list:
        # Original behavior with passages
        l = len(passage_list)
        passages = ""
        for i, passage in enumerate(passage_list):
            if format_passages:
                passages += f"Passage {i+1}: {format_passage(passage['title'], passage['text'])}"
            else:
                # For passages that are already formatted, basically just add the modified_ctx to the prompt
                passages += f"Passage {i+1}: {passage['modified_passage']}"
            if i != l-1:
                passages += "\n"
        if base_model_prompt:
            return BASE_MODEL_PROMPT_TEMPLATE.format(question=question, passages=passages)
        elif prompt_type == "with_passage_check_relevance":
            return PROMPT_TEMPLATE_RELEVANCE_CHECKING.format(question=question, passages=passages)
        else:
            return PROMPT_TEMPLATE_KNOWLEDGE_CHOOSING.format(question=question, passages=passages)
    
    elif prompt_type == "no_passage":
        # Version that does prompt model to refuse to answer
        return PROMPT_NO_PASSAGE_TEMPLATE.format(question=question)
    elif prompt_type == "no_passage_knowledge_check":
        return PROMPT_PARAMETRIC_CHECK_TEMPLATE.format(question=question)
    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}")


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)