import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import GPT2Tokenizer, pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from transformers import LogitsProcessor, LogitsProcessorList
from huggingface_hub import HfApi, create_repo, upload_folder
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import json
import shutil
import argparse
from utils_ppo import *

global_step = 0
SAVE_EVERY = 1000

class ClampLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        return torch.clamp(scores, -1e4, 1e4)

# Reward function — uses baby output + teacher references
def reward_fn_blue(prompts, baby_responses, teacher_lists, log_file=None):
    rewards = []

    with open(log_file, "a", encoding="utf-8") as f:
        for prompt, baby_out, teacher_text in zip(prompts, baby_responses, teacher_lists):

            if not baby_out:
                rewards.append(torch.tensor(0.0))
                f.write(f"Prompt: {prompt}\n")
                f.write("No usable baby response.\n")
                f.write("—" * 40 + "\n")
                continue

            
            bleu_score = calculate_bleu1_nltk(teacher_text.lower(), baby_out)
            rewards.append(torch.tensor(bleu_score, dtype=torch.float))

            f.write(f"Prompt: {prompt}\n")
            f.write(f"Baby said: {baby_out}\n")
            f.write(f"Teacher refs: {teacher_text}\n")
            f.write(f"BLEU scores: {bleu_score}\n")
            f.write("—" * 40 + "\n")

    return rewards

def reward_fn_sem_similarity(prompts, baby_responses, teacher_lists, log_file=None):
    rewards = []
    with open(log_file, "a", encoding="utf-8") as f:
        for prompt, baby_out, teacher_text in zip(prompts, baby_responses, teacher_lists):

            emb_b, emb_t = embedder.encode([baby_out, teacher_text], convert_to_tensor=True)
            sem_sim = util.cos_sim(emb_b, emb_t).item()
            rewards.append(torch.tensor(float(sem_sim), dtype=torch.float))

            f.write(f"Prompt: {prompt}\n")
            f.write(f"Baby said: {baby_out}\n")
            f.write(f"Teacher refs: {teacher_text}\n")
            f.write(f"Rewards scores: {rewards}\n")
            f.write("—" * 40 + "\n")
            

    return rewards

def parse_arguments():
    parser = argparse.ArgumentParser(description="Choose reward function for PPO training.")
    parser.add_argument(
        "--reward_fn", 
        choices=["semsim", "bleu"], 
        default="semsim", 
        help="Reward function to use: 'semsim' (semantic similarity) or 'bleu' (BLEU score)"
    )
    return parser.parse_args()

def main():
    global global_step
    args = parse_arguments()


    if args.reward_fn == "bleu":
        reward_fn = reward_fn_blue
        repo_name = "rfblue-abs-002"
        log_rewards_file = "./txt_files/reward_tracking_blue.csv"
        log_file = "ppo_training_blue.txt"
        
    else:
        reward_fn = reward_fn_sem_similarity
        repo_name = "rfsem-abs-002"
        log_rewards_file = "./txt_files/reward_tracking_semsim.csv"
        log_file = "ppo_training_semsim.txt"
    

    create_repo(f"fpadovani/{repo_name}", exist_ok=True)

    with open(log_rewards_file, "w") as f:
        f.write("epoch,batch,avg_reward\n")

    BABY = "bbunzeck/another-llama"
    tokenizer_baby = AutoTokenizer.from_pretrained(BABY, use_fast=True)
    baby = AutoModelForCausalLMWithValueHead.from_pretrained(BABY)
    ref_baby = AutoModelForCausalLMWithValueHead.from_pretrained(BABY)
    tokenizer_baby.pad_token = tokenizer_baby.eos_token    

    global embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # load the mother prompts and teacher model responses
    raw_data = []
    with open("./txt_files/complete_teacher_answers_best.json", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  
                raw_data.append(json.loads(line))

    raw_data = raw_data[:220000]  
    create_repo(f"fpadovani/{repo_name}", exist_ok=True)

    config = PPOConfig(
        batch_size       = 16,
        mini_batch_size  = 4)
    '''learning_rate    = 5e-6,    # lower LR
        log_with         = None,
        kl_penalty       = "abs",
        #kl_target        = 0.05,   # tighter KL (adaptive)
        init_kl_coef     = 0.02)'''

    ppo = PPOTrainer(
        config,
        model=baby,
        ref_model=ref_baby,
        tokenizer=tokenizer_baby,
        dataset=None
    )

    clamp_lp = LogitsProcessorList([ClampLogits()])

    # === Training Loop ===
    for epoch in range(4):
        random.shuffle(raw_data)
        dataset_epoch = make_dataset(raw_data)

        dataloader = DataLoader(
            dataset_epoch,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            prompts_batch = batch["original_prompt"]
            teacher_refs_batch = batch["best_response"]

            prompts_batch = [
                prompt.strip() + " *CHI:" if not prompt.strip().endswith("*CHI:") else prompt.strip()
                for prompt in prompts_batch
            ]

            if len(prompts_batch) < config.batch_size:
                continue

            tokenized_prompts = [tokenizer_baby.encode(p, return_tensors="pt").squeeze(0) for p in prompts_batch]
            generations = ppo.generate(tokenized_prompts, max_new_tokens=20, do_sample=True, top_p=0.9, return_prompt=True, logits_processor=clamp_lp)

            response_tensors = []
            decoded_responses = []

            for prompt, result in zip(prompts_batch, generations):
                full_text = tokenizer_baby.decode(result, skip_special_tokens=True).strip()
                gen_text = extract_first_chi_utterance(full_text[len(prompt)-5:].strip())
                decoded_responses.append(gen_text)

                response_tensor = tokenizer_baby.encode(gen_text, return_tensors="pt").squeeze(0)
                response_tensors.append(response_tensor)

            rewards = reward_fn(prompts_batch, decoded_responses, teacher_refs_batch, log_file=log_file)

            stats = ppo.step(tokenized_prompts, response_tensors, rewards)
            avg_reward = sum(r.item() for r in rewards) / len(rewards)

            with open(log_rewards_file, "a") as f:
                f.write(f"{epoch+1},{batch_idx+1},{avg_reward:.4f}\n")

            global_step += 1  
            if global_step % SAVE_EVERY == 0:
                output_dir = f"./fine_tuned_models/{repo_name.split('/')[-1]}/checkpoint-{global_step}"
                ppo.model.save_pretrained(output_dir)
                tokenizer_baby.save_pretrained(output_dir)

                upload_folder(
                    repo_id=f"fpadovani/{repo_name}",
                    folder_path=output_dir,
                    path_in_repo=f"checkpoint-{global_step}", 
                    commit_message=f"Add checkpoint at step {global_step}",
                )

                shutil.rmtree(output_dir)
        
        # Save locally and push to hub at the end
        output_dir = f"./fine_tuned_models/{repo_name.split('/')[-1]}/epoch-{epoch+1}"
        ppo.model.save_pretrained(output_dir)
        tokenizer_baby.save_pretrained(output_dir)

        upload_folder(
            repo_id=f"fpadovani/{repo_name}",
            folder_path=output_dir,
            path_in_repo=f"epoch-{epoch+1}", 
            commit_message=f"Add checkpoint at step {global_step}",
                )

    # Save locally and push to hub at the end
    output_dir = f"./fine_tuned_models/{repo_name.split('/')[-1]}"
    ppo.model.save_pretrained(output_dir)
    tokenizer_baby.save_pretrained(output_dir)

    ppo.model.push_to_hub(f"fpadovani/{repo_name}")
    tokenizer_baby.push_to_hub(f"fpadovani/{repo_name}")

if __name__ == "__main__":
    main()