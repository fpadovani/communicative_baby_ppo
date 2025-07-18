import re
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
import torch
from datasets import Dataset


embedder = SentenceTransformer('all-MiniLM-L6-v2')
bleu_metric = evaluate.load("bleu")


def custom_collate_fn(batch):
    # batch is a list of dicts, each with 'prompt' and 'teacher_responses'
    prompts = [item["original_prompt"] for item in batch]
    teacher_responses = [item["best_response"] for item in batch]
    return {
        "original_prompt": prompts,
        "best_response": teacher_responses
    }

def custom_collate_fn_ppo_prompt(batch):
    # batch is a list of dicts, each with 'query'
    prompts = [item["query"] for item in batch]
    return {
        "query": prompts
    }

# Filter and restructure dataset
def make_dataset(shuffled_data):
    return Dataset.from_dict({
        "original_prompt": [item["original_prompt"] for item in shuffled_data],
        "best_response": [item["best_response"] for item in shuffled_data]
    })

def calculate_bleu_evaluate(reference, hypothesis):
    """
    reference and hypothesis should be strings
    Returns BLEU score between 0 and 1.
    """
    result = bleu_metric.compute(predictions=[hypothesis], references=[[reference]])
    return result["bleu"]


def calculate_bleu1_nltk(reference, hypothesis):
    reference = re.sub(r'[^\w\s]', '', reference.lower())
    hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower())
    
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method1  # Optional smoothing
    return sentence_bleu(
        [reference_tokens],
        hypothesis_tokens,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smoothie
    )

def generate_teacher_responses(teacher_model, mother_sent, num_responses=10):
    
    system_msg = "Please respond in a way that mimics the style of a good child in response to mother's speech. You should respond in a nice way! The response should be only a meaningful very short sentence. Don't include any additional text or mother answers"
    user_msg = f"Mother: {mother_sent}\n Child: "

    prompt = f"{system_msg}\n{user_msg}"

    outputs = teacher_model(prompt, max_new_tokens=20, do_sample=True, num_return_sequences=num_responses, temperature= 0.5)

    responses = [out['generated_text'].replace(prompt, "").strip() for out in outputs]
    return responses


def generate_teacher_score(teacher_model, mother_sent, baby_response):
    """
    Generate a score from the teacher model for the baby response.
    The score should be between 1 and 5.
    """
    system_msg = "You are presented with a dialogue between a mother (MOT) and a child (CHI). Please rate how fluent the child's response is, on a scale from 1 (completely unfitting) to 5 (perfectly fine answer):"
    user_msg = f"MOT:{mother_sent} CHI:{baby_response}"

    prompt = f"{system_msg}\n{user_msg}"

    outputs = teacher_model(prompt, max_new_tokens=10, do_sample=False)
    
    # Extract the score from the output
    score_text = outputs[0]['generated_text'].replace(prompt, "").strip()
    
    # Try to convert to an integer
    try:
        score = int(score_text)
        return max(1, min(score, 5))  # Ensure score is between 1 and 5
    except ValueError:
        return None  # If conversion fails, return None
    


def extract_first_chi_utterance(text):
    """
    Extract the first utterance after *CHI: from a multi-utterance baby response.
    Keeps ending punctuation like '.', '!', or '?'.

    Example: "*CHI: yes! *MOT: okay. *CHI: mummy." → "yes!"
    """
    pattern = r"\*CHI:\s*(.+?)(?=\s*\*[A-Z]+:|$)"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip().lower()  # Keep punctuation
    return ""

def extract_first_answer_with_punct(text):
    """
    Extract the first sentence from the teacher response string,
    starting from the beginning of the answer and including
    the first punctuation mark (.!?).
    """
    
    #text = re.sub(r"\b\d+[\.)]\s*|\b\d+\b", "", text)
    punc_match = re.search(r"[.!?]", text)
    
    if punc_match:
        pos = punc_match.start()
        text = text[:pos + 1]
        text = text.split("\n")[0]
        return text
    else:
        text = text.split("\n")[0]
        return text


def extract_first_utterance(text):
    # Look for the earliest occurrence of any of these tokens (with or without leading space)
    split_tokens = [r"\nMother", r"\n Mother", r"\nChild", r"\n Child"]
    
    earliest_pos = None
    for token in split_tokens:
        match = re.search(token, text)
        if match:
            pos = match.start()
            if earliest_pos is None or pos < earliest_pos:
                earliest_pos = pos
                
    if earliest_pos is not None:
        # Return everything before the earliest token
        return text[:earliest_pos].strip()
    else:
        # No tokens found, extract first sentence by punctuation
        first_line = text.split("\n")[0].strip()
        punc_match = re.search(r"[.!?]", first_line)
        if punc_match:
            return first_line[:punc_match.start()+1].strip()
        else:
            return first_line
    


def reward_fn_balanced(prompts, baby_responses, teacher_lists, log_file=None):
    rewards = []

    with open(log_file, "a", encoding="utf-8") as f:
        for prompt, baby_out, teacher_texts in zip(prompts, baby_responses, teacher_lists):

            teacher_texts_cleaned = [extract_first_utterance(t) for t in teacher_texts]

            reward_scores = []

            for t_text in teacher_texts:
                emb_b, emb_t = embedder.encode([baby_out, t_text], convert_to_tensor=True)
                sem_sim = util.cos_sim(emb_b, emb_t).item()

                bleu_score = calculate_bleu1_nltk(t_text.lower(), baby_out)
                reward = sem_sim * (1 - bleu_score)
                reward_scores.append(reward)

            avg_reward = sum(reward_scores) / len(reward_scores)
            rewards.append(torch.tensor(avg_reward))
        

            f.write(f"Prompt: {prompt}\n")
            f.write(f"Baby said: {baby_out}\n")
            f.write(f"Teacher refs: {teacher_texts_cleaned}\n")
            f.write(f"Rewards scores: {rewards}\n")
            f.write("—" * 40 + "\n")
            

    return rewards





######### PROMPTING OLMO #########
def custom_collate_fn_olmo(batch):
    return {"query": [item["query"] for item in batch]}

def build_teacher_prompt(mot, chi):
    return (
        "<|system|>\nYou are presented with a dialogue between a mother (MOT) and a child (CHI). "
        "Please rate how fluent the child's response is, on a scale from 0 (completely unfitting) to 5 (perfectly fine answer). "
        "Respond with only the number.\n<|end|>\n"
        f"<|user|>\nMOT: {mot}\nCHI: {chi}\n<|end|>\n<|assistant|>\n"
    )

def build_teacher_prompt_new_model(mother_utt, child_utt):
    return (
        "<|im_start|>system\n"
        "You are presented with a dialogue between a mother (MOT) and a child (CHI). "
        "Rate how contextually plausible and fluent the child's utterance is in response to the mother's utterance on a scale from 1 to 5. Respond with only the number.<|im_end|>\n"
        "<|im_start|>user\n"
        f"MOT: {mother_utt}\CHI: {child_utt}\nScore (1-5):<|im_end|>\n"
        "<|im_start|>assistant\n"
    )



def extract_score(text):
        """Extract numerical score from text"""
        match = re.search(r"[1-5](?:\.\d+)?", text)
        return float(match.group(0)) if match else 0.0