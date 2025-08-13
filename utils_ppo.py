from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
import torch
from datasets import Dataset
import unicodedata, re

DEBUG_ONCE = True          # flips to False after 


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.strip())
    s = s.strip('"\''"“”‘’")              # strip any straight/curly quotes
    s = s.replace("*MOT:", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s



def safe_tensor(x, device):
    return torch.tensor(float(x), dtype=torch.float32, device=device)


embedder = SentenceTransformer('all-MiniLM-L6-v2')


#######
# PROMPTING with vLLM
#######


def generate_teacher_responses_vllm_batch(llm, mother_sents, sampling_params, scorer_model, scorer_tokenizer):
    """Generate and rank responses using vLLM and HuggingFace for scoring."""
    system_msg = (
        "You are a young child having a conversation with your mother. "
        "When your mother says something, you should answer as a typical and natural-sounding child. "
        "Do NOT repeat her words. Instead, give a new, relevant answer that shows understanding. "
        "Keep it short and child-like."
    )

    # Step 1: Create prompts for vLLM
    full_prompts = []
    for mother_sent in mother_sents:
        user_msg = f"Mother says:{mother_sent}\n Child answers:"
        full_prompt = f"{system_msg}\n{user_msg}"
        full_prompts.append(full_prompt)

    # Step 2: Generate responses (batch)
    outputs = llm.generate(full_prompts, sampling_params)

    # Step 3: Collect raw responses
    all_responses = []
    for output in outputs:
        responses = [completion.text.strip() for completion in output.outputs]
        all_responses.append(responses)

    return all_responses



def custom_collate_fn(batch):
    # batch is a list of dicts, each with 'prompt' and 'teacher_responses'
    prompts = [item["original_prompt"] for item in batch]
    teacher_responses = [item["best_response"] for item in batch]
    return {
        "original_prompt": prompts,
        "best_response": teacher_responses
    }


# Filter and restructure dataset
def make_dataset(shuffled_data):
    return Dataset.from_dict({
        "original_prompt": [item["original_prompt"] for item in shuffled_data],
        "best_response": [item["best_response"] for item in shuffled_data]
    })



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
        "Please rate how contextually appropriate and fluent the child's response is, on a scale from 0 (completely unfitting) "
        "to 5 (perfectly fine answer). If CHI answer is too short rate it low.\n<|end|>\n"
        f"<|user|>\nMOT: {mot}\nCHI: {chi}\n<|end|>\n<|assistant|>\n"
    )


def extract_score(text):
        """Extract numerical score from text"""
        match = re.search(r"[1-5](?:\.\d+)?", text)
        return float(match.group(0)) if match else 0.0


def lnP_mom(prompt, answer, tokenizer, model):
    """
    Return natural‑log probability of `answer` under MOM given `prompt`.
    `prompt` should be the *MOT sentence only* (no *MOT tag).
    """
    ctx = prompt + " "    
    ids_ctx = tokenizer(ctx, add_special_tokens=False).input_ids
    ids_ans = tokenizer(" " + answer, add_special_tokens=False).input_ids
    ids = torch.tensor([ids_ctx + ids_ans], device=model.device)
    labels = ids.clone()
    labels[0, :len(ids_ctx)] = -100       # mask prompt tokens
    with torch.no_grad():
        loss = model(ids, labels=labels).loss    # mean NLL (nats)
    return -loss.item() * len(ids_ans)   
