import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import GPT2Tokenizer, pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from torch.utils.data import DataLoader
from transformers import default_data_collator
import re
import csv
import shutil
from utils_ppo import *
from tqdm import tqdm
import gc
import os 


# === CONFIG ===
SAVE_EVERY = 1000
log_rewards_file = "./txt_files/reward_tracking_log_score.csv"
CSV_LOG_FILE = "./csv_logs/responses_with_scores.csv"
repo_name = "fpadovani/rfscore-baby"
global_step = 0

class ClampLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        return torch.clamp(scores, -1e4, 1e4)


if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mother Prompt", "Child Response", "Raw Score", "Extracted Score"])

with open(log_rewards_file, "w", encoding="utf-8") as f:
    f.write("epoch,batch,avg_reward\n")

torch.cuda.empty_cache()
create_repo(repo_name, exist_ok=True)



def main():
    global global_step

    # === Models ===
    BABY = "bbunzeck/another-llama"
    tokenizer_baby = AutoTokenizer.from_pretrained(BABY, use_fast=True)
    tokenizer_baby.pad_token = tokenizer_baby.eos_token
    baby_model = AutoModelForCausalLMWithValueHead.from_pretrained(BABY)
    ref_baby = AutoModelForCausalLMWithValueHead.from_pretrained(BABY)

    TEACHER = "allenai/OLMo-2-1124-7B-Instruct"
    tokenizer_teacher = AutoTokenizer.from_pretrained(TEACHER)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    config = PPOConfig(
    batch_size       = 16,
    mini_batch_size  = 4)
    '''learning_rate    = 5e-6,    # lower LR
    log_with         = None,
    kl_penalty       = "abs",
    #kl_target        = 0.05,   # tighter KL (adaptive)
    init_kl_coef     = 0.02)'''



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


    def reward_fn(samples, responses, log_file, **kwargs):
        rewards = []
        prompts = samples["query"]


        with open(log_file, "a", encoding="utf-8", newline="") as f_csv:

            writer = csv.writer(f_csv)

            for prompt, baby_out in zip(prompts, responses):
                baby_text = tokenizer_baby.decode(baby_out, skip_special_tokens=True).strip()
                baby_utt = extract_first_chi_utterance(baby_text)
                mot_utt = prompt.replace("*MOT:", "").strip()

                if not baby_utt:
                    rewards.append(torch.tensor(0.0))
                    continue

                score_raw = score_with_olmo(mot_utt, baby_utt)
                reward_val = extract_score(score_raw)

                writer.writerow([mot_utt, baby_utt, score_raw, reward_val])
                rewards.append(torch.tensor(reward_val))

        return rewards

    # === 2. Load mother prompts ===
    with open("./txt_files/mother_prompts_1.txt", "r", encoding="utf-8") as f:
        mother_prompts = [line.strip() for line in f if line.strip()][:220000]


    train_dataset = Dataset.from_dict({"query": mother_prompts})

    # === PPO Setup ===
    ppo = PPOTrainer(
        config,
        model=baby_model,
        ref_model=ref_baby,
        tokenizer=tokenizer_baby,
        dataset=train_dataset,
    )
    clamp_lp = LogitsProcessorList([ClampLogits()])


    for epoch in range(4):
        ppo.dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn_olmo
        )

        for batch_idx, batch in enumerate(tqdm(ppo.dataloader, desc=f"Epoch {epoch + 1}", leave=True)):
            prompts_batch = batch["query"]
            if len(prompts_batch) < config.batch_size:
                continue

            tokenized_prompts = [
                tokenizer_baby.encode('*MOT: ' + p, return_tensors="pt").squeeze(0)
                for p in prompts_batch
            ]

            baby_responses = ppo.generate(
                tokenized_prompts,
                max_new_tokens=20,
                return_prompt=False,
                do_sample=True,
                top_p=0.9,
                logits_processor=clamp_lp
                
            )

            rewards = reward_fn({"query": prompts_batch}, baby_responses, log_file=CSV_LOG_FILE)
            stats = ppo.step(tokenized_prompts, baby_responses, rewards)
            avg_reward = sum(r.item() for r in rewards) / len(rewards)

            with open(log_rewards_file, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{batch_idx+1},{avg_reward:.4f}\n")

            global_step += 1
            if global_step % SAVE_EVERY == 0:
                output_dir = f"./fine_tuned_models/{repo_name.split('/')[-1]}/checkpoint-{global_step}"
                ppo.model.save_pretrained(output_dir)
                tokenizer_baby.save_pretrained(output_dir)

                upload_folder(
                    repo_id=repo_name,
                    folder_path=output_dir,
                    path_in_repo=f"checkpoint-{global_step}",
                    commit_message=f"Add checkpoint at step {global_step}",
                )
                shutil.rmtree(output_dir)
            
            

        # Optional: Save end-of-epoch checkpoint
        epoch_output_dir = f"./fine_tuned_models/{repo_name.split('/')[-1]}/epoch-{epoch+1}"
        ppo.model.save_pretrained(epoch_output_dir)
        tokenizer_baby.save_pretrained(epoch_output_dir)
        upload_folder(
            repo_id=repo_name,
            folder_path=epoch_output_dir,
            path_in_repo=f"epoch-{epoch+1}",
            commit_message=f"Add checkpoint at epoch {epoch+1}",
        )
        shutil.rmtree(epoch_output_dir)

    # Final save
    final_dir = f"./fine_tuned_models/{repo_name.split('/')[-1]}"
    ppo.model.save_pretrained(final_dir)
    tokenizer_baby.save_pretrained(final_dir)

    upload_folder(
        repo_id=repo_name,
        folder_path=final_dir,
        path_in_repo="final",
        commit_message="Final model upload",
    )

if __name__ == "__main__":
    main()