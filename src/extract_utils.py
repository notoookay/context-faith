from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from typing import List

def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    if 'instruct' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {}))
    else: # non instruct model
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {}))
        if "input_ids" in model_input:
            model_input = model_input["input_ids"]
    return model_input

def generate(model_input, model, model_name, do_sample=False, output_scores=False, temperature=1.0, top_k=50, top_p=1.0,
             max_new_tokens=100, stop_token_id=None, tokenizer=None, output_hidden_states=False, additional_kwargs=None):

    if stop_token_id is not None:
        eos_token_id = stop_token_id
    else:
        eos_token_id = None

    model_output = model.generate(model_input,
                                  max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states,
                                  output_scores=output_scores,
                                  return_dict_in_generate=True, do_sample=do_sample,
                                  temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                                  **(additional_kwargs or {}))

    return model_output

def load_model_and_validate_gpu(model_path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Started loading model")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                 torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    assert ('cpu' not in model.hf_device_map.values())
    return model, tokenizer

def compute_correctness_with_llm(all_questions, model_answers, labels, model=None, tokenizer=None):
    print("Computing correctness with LLM")

    if model is None:
        model, tokenizer = load_model_and_validate_gpu('mistralai/Mistral-7B-Instruct-v0.3')
    correctness = []
    for question, model_answer, label in tqdm(zip(all_questions, model_answers, labels)):
        if str(label).lower() in str(model_answer).lower():
            correctness.append(1)
        else:
            prompt = f"""
                Evaluate the following answers to questions. For each question you would be given a model answer and the correct answer.
                You would have to determine if the model answer is correct or not. If the model answer is correct, write '1' and if it is not correct, write '0'.
                For example:
                
                Question: who is the young guitarist who played with buddy guy?
                Ground Truth: Quinn Sullivan
                Model Answer: Ronnie Earl Explanation: Ronnie Earl is an American blues guitarist and singer who has played with many famous blues musicians, including Buddy Guy. He is known for his soulful and melodic playing style, and has released many albums that blend blues, jazz, and rock music. Earl has also been a member of the Buddy Guy Blues Band and has played with other notable blues musicians such as B.B. King, Eric Clapton, and Stevie Ray Vaughan. He is considered one of the most
                Correctness: 0
                
                Question: name of the first episode of stranger things 
                Ground Truth: Chapter One : The Vanishing of Will Byers
                Model Answer:  The disappearance of Will Byers. Explanation: The first episode of the first season of Stranger Things is titled "The Vanishing of Will Byers". The episode introduces the main characters and sets the tone for the rest of the series. It follows the story of Will Byers, a young boy who goes missing in the fictional town of Hawkins, Indiana, and the subsequent search for him by his mother Joyce and his friends Mike, Dustin, and Lucas. The episode sets the stage for the supernatural
                Correctness: 1
                
                Question: {question}
                Ground Truth: {label}
                Model Answer: {model_answer}
                Correctness:
                """

            model_input = tokenize(prompt, tokenizer, 'mistralai/Mistral-7B-Instruct-v0.3').to(model.device)
            valid = 0
            retries = 0
            sample = True
            while valid == 0 and retries < 5:
                with torch.no_grad():
                    model_output = generate(model_input, model, 'mistralai/Mistral-7B-Instruct-v0.3', sample, False)
                    current_correctness = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])

                current_correctness = (
                    current_correctness.replace(".</s>", "").replace("</s>", "").split('\n')[0].strip().strip("."))
                index_of_1 = current_correctness.find('1')
                index_of_0 = current_correctness.find('0')
                if index_of_1 != -1 and (index_of_0 == -1 or index_of_1 < index_of_0):
                    valid = 1
                    correctness.append(1)
                    break
                elif index_of_0 != -1 and (index_of_1 == -1 or index_of_0 < index_of_1):
                    valid = 1
                    correctness.append(0)
                    break
                else:
                    print(f"Invalid input: {current_correctness}")
                    retries += 1

                sample = True
                retries += 1

            if valid == 0:
                print("Invalid input")
                correctness.append(0)

    return correctness

def compute_correctness_with_triviaqa(model_answers, labels):
    """
    Compute correctness of model answers via accuracy
    """

    correctness = []
    for model_answer in model_answers:
        correctness.append(max([1 if label in model_answer else 0 for label in labels]))

    return correctness

LIST_CORRECTNESS_FN = {
    "llm": compute_correctness_with_llm,
    "triviaqa": compute_correctness_with_triviaqa,
}

def compute_correctness(data, data_name, method='llm', correctness_model=None):
    if method == 'llm':
        pass # TODO: Implement this
    else:
        if 'triviaqa' in data_name:
            for item in data:
                item["correctness"] = compute_correctness_with_triviaqa(
                    item["predicts"], item["answer"]
                )
                item["avg_correctness"] = np.mean(item["correctness"])
        else:
            pass

def plot_accuracy_distribution(
    data: List[dict],
    output_dir: str = "plots",
    file_name: str = "accuracy_distribution",
):
    """Plot the distribution of average accuracy scores"""
    avg_c = [item['avg_correctness'] for item in data]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot the distribution of average accuracies
    plt.figure(figsize=(10, 6))
    sns.histplot(avg_c, bins=11, kde=True)  # 11 bins for 0.0, 0.1, ..., 1.0
    plt.title("Distribution of Average Accuracy Scores")
    plt.xlabel("Average Accuracy")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)

    # Add a vertical line for the mean
    mean_acc = np.mean(avg_c)
    plt.axvline(x=mean_acc, color='r', linestyle='--', label=f'Mean: {mean_acc:.4f}')
    plt.legend()

    # Save the figure
    output_file = os.path.join(output_dir, f"{file_name}__accuracy_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy distribution plot saved to {output_file}")

    # Create additional visualization: Pie chart of zero vs non-zero accuracy
    plt.figure(figsize=(8, 8))
    zero_acc = sum(acc == 0 for acc in avg_c)
    non_zero_acc = len(avg_c) - zero_acc
    plt.pie(
        [zero_acc, non_zero_acc], 
        labels=['Zero Accuracy', 'Non-Zero Accuracy'],
        autopct='%1.1f%%',
        colors=['#FF9999', '#66B2FF'],
        explode=(0.1, 0)
    )
    plt.title("Proportion of Questions with Zero vs Non-Zero Accuracy")

    # Save the pie chart
    pie_output_file = os.path.join(output_dir, f"{file_name}__accuracy_pie_chart.png")
    plt.savefig(pie_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy pie chart saved to {pie_output_file}")