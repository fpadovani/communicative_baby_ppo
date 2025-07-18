import torch
from transformers import GPT2Tokenizer, pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
SAVE_EVERY = 10
log_rewards_file = "./txt_files/reward_tracking_log_score.csv"
CSV_LOG_FILE = "./csv_logs/responses_with_scores.csv"
repo_name = "fpadovani/rfscore-baby"
global_step = 0

# Performance optimization configs
BATCH_TEACHER_INFERENCE = True  # Enable batched teacher inference
MAX_TEACHER_BATCH_SIZE = 8      # Adjust based on your GPU memory
USE_TEACHER_CACHE = True        # Cache teacher responses
CACHE_SIZE = 1000               # LRU cache size
ASYNC_SCORING = True            # Use async scoring (experimental)

if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mother Prompt", "Child Response", "Raw Score", "Extracted Score"])

torch.cuda.empty_cache()
create_repo(repo_name, exist_ok=True)

# Simple LRU cache for teacher responses
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key: str):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)

def main():
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

    config = PPOConfig(batch_size=16, mini_batch_size=2)
    
    # Initialize cache
    teacher_cache = LRUCache(CACHE_SIZE) if USE_TEACHER_CACHE else None

    def score_with_olmo_single(mot, chi, max_new_tokens=10):
        """Original single scoring function"""
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

    def score_with_olmo_batch(mot_chi_pairs, max_new_tokens=10):
        """Batched scoring function - much faster!"""
        prompts = [build_teacher_prompt(mot, chi) for mot, chi in mot_chi_pairs]
        
        # Tokenize all prompts
        inputs = tokenizer_teacher(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(teacher_model.device)
        
        # Generate all at once
        with torch.no_grad():
            outputs = teacher_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer_teacher.eos_token_id
            )
        
        # Decode responses
        score_responses = []
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            full_output = tokenizer_teacher.decode(output, skip_special_tokens=True)
            score_response = full_output[len(prompt):].strip()
            score_responses.append(score_response)
        
        return score_responses

    def get_cached_or_compute_score(mot, chi):
        """Get score from cache or compute it"""
        if not USE_TEACHER_CACHE:
            return score_with_olmo_single(mot, chi)
        
        cache_key = f"{mot}|||{chi}"
        cached_score = teacher_cache.get(cache_key)
        if cached_score is not None:
            return cached_score
        
        score = score_with_olmo_single(mot, chi)
        teacher_cache.put(cache_key, score)
        return score

    def reward_fn_optimized(samples, responses, log_file=None, **kwargs):
        """Optimized reward function with batching and caching"""
        rewards = []
        prompts = samples["query"]
        
        # Prepare data for processing
        valid_pairs = []
        valid_indices = []
        
        for i, (prompt, baby_out) in enumerate(zip(prompts, responses)):
            baby_text = tokenizer_baby.decode(baby_out, skip_special_tokens=True).strip()
            baby_utt = extract_first_chi_utterance(baby_text)
            mot_utt = prompt.replace("*MOT:", "").strip()
            
            if not baby_utt:
                rewards.append(torch.tensor(0.0))
                continue
                
            valid_pairs.append((mot_utt, baby_utt))
            valid_indices.append(i)
        
        # Process in batches
        if BATCH_TEACHER_INFERENCE and valid_pairs:
            all_scores = []
            for i in range(0, len(valid_pairs), MAX_TEACHER_BATCH_SIZE):
                batch_pairs = valid_pairs[i:i + MAX_TEACHER_BATCH_SIZE]
                batch_scores = score_with_olmo_batch(batch_pairs)
                all_scores.extend(batch_scores)
        else:
            # Fallback to individual scoring (with caching)
            all_scores = []
            for mot_utt, baby_utt in valid_pairs:
                score_raw = get_cached_or_compute_score(mot_utt, baby_utt)
                all_scores.append(score_raw)
        
        # Process scores and log
        with open(CSV_LOG_FILE, "a", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            
            score_idx = 0
            for i, (prompt, baby_out) in enumerate(zip(prompts, responses)):
                if i in valid_indices:
                    baby_text = tokenizer_baby.decode(baby_out, skip_special_tokens=True).strip()
                    baby_utt = extract_first_chi_utterance(baby_text)
                    mot_utt = prompt.replace("*MOT:", "").strip()
                    
                    score_raw = all_scores[score_idx]
                    reward_val = extract_score(score_raw)
                    
                    writer.writerow([mot_utt, baby_utt, score_raw, reward_val])
                    rewards.append(torch.tensor(reward_val))
                    score_idx += 1
        
        return rewards

    def reward_fn_async(samples, responses, log_file=None, **kwargs):
        """Async version using ThreadPoolExecutor"""
        rewards = []
        prompts = samples["query"]
        
        def process_single_response(prompt, baby_out):
            baby_text = tokenizer_baby.decode(baby_out, skip_special_tokens=True).strip()
            baby_utt = extract_first_chi_utterance(baby_text)
            mot_utt = prompt.replace("*MOT:", "").strip()
            
            if not baby_utt:
                return 0.0, None, None, None
                
            score_raw = get_cached_or_compute_score(mot_utt, baby_utt)
            reward_val = extract_score(score_raw)
            return reward_val, mot_utt, baby_utt, score_raw
        
        # Process responses concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_single_response, prompt, baby_out)
                for prompt, baby_out in zip(prompts, responses)
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        # Log results
        with open(CSV_LOG_FILE, "a", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            for reward_val, mot_utt, baby_utt, score_raw in results:
                if mot_utt is not None:
                    writer.writerow([mot_utt, baby_utt, score_raw, reward_val])
                rewards.append(torch.tensor(reward_val))
        
        return rewards

    # Choose reward function based on config
    if ASYNC_SCORING:
        reward_fn = reward_fn_async
    else:
        reward_fn = reward_fn_optimized

    # === 2. Load mother prompts ===
    with open("./txt_files/mother_prompts_1.txt", "r", encoding="utf-8") as f:
        mother_prompts = [line.strip() for line in f if line.strip()]

    train_dataset = Dataset.from_dict({"query": mother_prompts})

    # === PPO Setup ===
    ppo = PPOTrainer(
        config,
        model=baby_model,
        ref_model=ref_baby,
        tokenizer=tokenizer_baby,
        dataset=train_dataset,
    )

    # Training loop with timing
    with open(log_rewards_file, "w") as f:
        f.write("epoch,batch,avg_reward,batch_time\n")
        
        for epoch in range(10):
            ppo.dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn_olmo
            )

            for batch_idx, batch in enumerate(tqdm(ppo.dataloader, desc=f"Epoch {epoch + 1}", leave=True)):
                batch_start_time = time.time()
                
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
                    top_p=0.9
                )

                rewards = reward_fn({"query": prompts_batch}, baby_responses)
                stats = ppo.step(tokenized_prompts, baby_responses, rewards)
                avg_reward = sum(r.item() for r in rewards) / len(rewards)

                batch_time = time.time() - batch_start_time
                f.write(f"{epoch+1},{batch_idx+1},{avg_reward:.4f},{batch_time:.2f}\n")

                print(f"Batch {batch_idx+1} completed in {batch_time:.2f}s, avg_reward: {avg_reward:.4f}")

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

                # Memory cleanup
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

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