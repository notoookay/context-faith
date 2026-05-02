import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"'", u"'", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def evaluate_faithfulness(predictions, task_type, strict_match=False):
    """
    Evaluate the faithfulness of predictions based on task type
    
    Args:
        predictions: List of prediction dictionaries
        task_type: Type of task (unanswerable, inconsistent, counterfactual)
        strict_match: Whether to use strict phrase matching
    
    Returns:
        accuracy, correct_count, total_count
    """
    correct_count = 0
    total_count = len(predictions)
    
    if task_type == "unanswerable":
        if strict_match:
            valid_phrases = ['unknown']
        else:
            valid_phrases = ['unknown', 'no answer', 'no information', 'not', 'unclear', 'unable to answer']
    
    elif task_type == "inconsistent":
        if strict_match:
            valid_phrases = ['conflict']
        else:
            valid_phrases = ['conflict', 'multiple answers', 'disagreement', 'inconsistent', 'contradictory', 'contradiction', 'inconsistency', 'two answers', '2 answers', 'conflicting']
    
    elif task_type == "counterfactual":
        # For counterfactual, we should check if the model has been fooled by the counterfactual info
        # This requires comparison with the true answer, so the logic is different
        for pred in predictions:
            model_answer = pred['response']
            true_answer = pred['answers'] # string
            # true_answer_key = pred['answer_key'] # string
            if true_answer.lower() in model_answer.lower():
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        return accuracy, correct_count, total_count
    
    # For unanswerable and inconsistent tasks
    for pred in predictions:
        model_answer = normalize_answer(pred['response'])
        is_correct = any(phrase in model_answer for phrase in valid_phrases)
        if is_correct:
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy, correct_count, total_count

def acc(response, answers):
    """
    Evaluate if a response contains any of the provided answers.
    
    Args:
        response: The generated response
        answers: List of possible correct answers
    
    Returns:
        Dict with accuracy score (0 or 1)
    """
    if not response or not answers:
        return {'acc': 0}
    
    # Normalize response
    response = response.lower().strip()
    
    # Check if any answer is in the response
    for answer in answers:
        if answer.lower().strip() in response:
            return {'acc': 1}
    
    return {'acc': 0}