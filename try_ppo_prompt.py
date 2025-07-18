import torch
from transformers import GPT2Tokenizer, pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from torch.utils.data import DataLoader
from transformers import default_data_collator
import re
import csv
from utils_ppo import *
from tqdm import tqdm
import gc
import os 

log_rewards_file = "./txt_files/reward_tracking_log_score.csv"
CSV_LOG_FILE = "./csv_logs/responses_with_scores.csv"
if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mother Prompt", "Child Response", "Raw Score", "Extracted Score"])


torch.cuda.empty_cache()

# Baby model (generates child utterances)
BABY = "bbunzeck/another-llama"
tokenizer_baby = AutoTokenizer.from_pretrained(BABY, use_fast=True)
tokenizer_baby.pad_token = tokenizer_baby.eos_token
baby_model = AutoModelForCausalLMWithValueHead.from_pretrained(BABY)
ref_baby = AutoModelForCausalLMWithValueHead.from_pretrained(BABY)

# OLMo teacher model (scores responses)
TEACHER = "allenai/OLMo-2-1124-7B-Instruct"
tokenizer_teacher = AutoTokenizer.from_pretrained(TEACHER)
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

config = PPOConfig(batch_size=16, mini_batch_size=16)



# === 2. Load mother prompts ===
with open("/home/p318482/babyLM_controlled/bielefeld/txt_files/mother_prompts_1.txt", "r", encoding="utf-8") as f:
    mother_prompts = [line.strip() for line in f if line.strip()]


def build_teacher_prompt(mot, chi):
    return (
        "<|system|>\nYou are presented with a dialogue between a mother (MOT) and a child (CHI). "
        "Please rate how contextually appropriate and fluent the child's response is, on a scale from 0 (completely unfitting) "
        "to 5 (perfectly fine answer). If CHI answer is too short rate it low.\n<|end|>\n"
        f"<|user|>\nMOT: {mot}\nCHI: {chi}\n<|end|>\n<|assistant|>\n"
    )


def score_with_olmo(mot, chi, max_new_tokens=10):
    prompt = build_teacher_prompt(mot, chi)
    inputs = tokenizer_teacher(prompt, return_tensors="pt").to(teacher_model.device)
    output = teacher_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer_teacher.eos_token_id
    )
    full_output = tokenizer_teacher.decode(output[0], skip_special_tokens=True)
    score_response = full_output[len(prompt):].strip()
    return score_response



def extract_score(text):
    match = re.search(r"[1-5](?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def reward_fn(samples, responses, log_file=None, **kwargs):
    rewards = []
    prompts = samples["query"]

    with open(log_file, "a", encoding="utf-8") as f_txt, \
         open(CSV_LOG_FILE, "a", encoding="utf-8", newline="") as f_csv:

        writer = csv.writer(f_csv)

        for prompt, baby_out in zip(prompts, responses):
            baby_text = tokenizer_baby.decode(baby_out, skip_special_tokens=True).strip()
            baby_utt = extract_first_chi_utterance(baby_text)
            mot_utt = prompt.replace("*MOT:", "").strip()

            if not baby_utt:
                rewards.append(torch.tensor(reward_val))
                
                f_txt.write(f"Prompt: {prompt}\nNo usable baby response.\n{'—' * 40}\n")
                writer.writerow([mot_utt, "", "N/A", 0.0])
                continue

            score_raw = score_with_olmo(mot_utt, baby_utt)
            reward_val = extract_score(score_raw)

            f_txt.write(f"Prompt: {prompt}\nCHI: {baby_utt}\nScore: {score_raw} => {reward_val}\n{'—' * 40}\n")
            writer.writerow([mot_utt, baby_utt, score_raw, reward_val])
            rewards.append(torch.tensor(reward_val))

    return rewards



def custom_collate_fn(batch):
    return {"query": [item["query"] for item in batch]}


train_dataset = Dataset.from_dict({"query": mother_prompts})

ppo = PPOTrainer(
    config,
    model=baby_model,
    ref_model=ref_baby,
    tokenizer=tokenizer_baby,
    dataset=train_dataset,

)


log_file = "ppo_training_log_score.txt"

with open(log_file, "w") as f:
    f.write("PPO Training Log\n")
    f.write("=" * 40 + "\n")

    for epoch in range(10):
        ppo.dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

        
        for batch_idx, batch in enumerate(tqdm(ppo.dataloader, desc=f"Epoch {epoch + 1}", leave=True)):
            prompts_batch = batch["query"]
            if len(prompts_batch) < config.batch_size:
                continue  # skip small final batch
            tokenized_prompts = [tokenizer_baby.encode('*MOT: ' + p, return_tensors="pt").squeeze(0) for p in prompts_batch]
            baby_responses = ppo.generate(tokenized_prompts, max_new_tokens=20, return_prompt=False, do_sample=True, top_p=0.9)  
            rewards = reward_fn({"query": prompts_batch}, baby_responses, log_file=log_file)
            stats = ppo.step(tokenized_prompts, baby_responses, rewards)
            avg_reward = sum(r.item() for r in rewards) / len(rewards)

            with open(log_rewards_file, "a") as f:
                f.write(f"{epoch+1},{batch_idx+1},{avg_reward:.4f}\n")

        
        epoch_output_dir = f"./fine_tuned_models/rfscore-baby-epoch-{epoch+1}"
        ppo.model.save_pretrained(epoch_output_dir)
        tokenizer_baby.save_pretrained(epoch_output_dir)
        print(f"Saved checkpoint for epoch {epoch + 1} at {epoch_output_dir}")

# Save locally
output_dir = "./fine_tuned_models/rfscore-baby"
ppo.model.save_pretrained(output_dir)
tokenizer_baby.save_pretrained(output_dir)

# Push to Hugging Face Hub
repo_name = "fpadovani/rfscore-baby"
create_repo(repo_name, exist_ok=True)

ppo.model.push_to_hub(repo_name)
tokenizer_baby.push_to_hub(repo_name)