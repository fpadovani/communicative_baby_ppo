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
import json
from transformers import LogitsProcessor, LogitsProcessorList

import sys, os, tqdm
TQDM_OFF = os.environ.get("TQDM_DISABLE", "0") == "1" or not sys.stdout.isatty()
tqdm.tqdm.monitor_interval = 0 

class ClampLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        return torch.clamp(scores, -1e4, 1e4)



# === CONFIG ===
SAVE_EVERY = 10
log_rewards_file = "./txt_files/reward_tracking_conf.csv"
CSV_LOG_FILE = "./csv_logs/responses_with_conf_scores.csv"
repo_name = "fpadovani/rfscore-baby"
global global_step
NEG_REWARD = -0.5          # instead of -1 for fallback



if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mother Prompt", "Child Response", "Raw Score", "Extracted Score"])

torch.cuda.empty_cache()
#create_repo(repo_name, exist_ok=True)



def main():
    global_step = 0


    # === Models ===
    #BABY = "bbunzeck/another-llama"
    BABY = "bbunzeck/another-llama"#"./fine_tuned_models/rfscore-baby/epoch-1"
    tokenizer_baby = AutoTokenizer.from_pretrained(BABY, use_fast=True)
    tokenizer_baby.pad_token = tokenizer_baby.eos_token
    baby_model = AutoModelForCausalLMWithValueHead.from_pretrained(BABY)
    ref_baby = AutoModelForCausalLMWithValueHead.from_pretrained(BABY)

    # fozen Mom (same model you used for the 10 alternatives) 
    MOM_MODEL_ID = "meta-llama/Llama-3.2-3B"
    tok_mom = AutoTokenizer.from_pretrained(MOM_MODEL_ID)
    mom_model = AutoModelForCausalLM.from_pretrained(
                    MOM_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    mom_model.eval()
    # -----------------------------------------------------------------------------

    mom_logps_dict = {}   # key = MOT sentence, value = list[float] len=10
    with open("./txt_files/mom_scores.jsonl") as f_ms:
        for line in f_ms:
            row = json.loads(line)
            #key = row["prompt"].strip() 
            key = norm_key(row["prompt"])
            mom_logps_dict[key] = row["mom_logps"] # MOT only
    # --- after building mom_logps_dict ---
    print("Loaded mom_logps_dict:", len(mom_logps_dict))
    k0 = next(iter(mom_logps_dict))
    print("Example key:", repr(k0))
    print("Len(logps):", len(mom_logps_dict[k0]))
    assert len(mom_logps_dict[k0]) == 10

    
    print("Loaded mom_logps_dict:", len(mom_logps_dict), "prompts")
    # Peek one key/value
    _first_key = next(iter(mom_logps_dict))
    print("Example key:", repr(_first_key))
    print("Example mom_logps len:", len(mom_logps_dict[_first_key]))
    assert len(mom_logps_dict[_first_key]) == 10, "Mom list must be 10"



    #config = PPOConfig(batch_size=16, mini_batch_size=2)
    # config = PPOConfig(
    # batch_size       = 16,
    # mini_batch_size  = 2,
    # learning_rate    = 2.5e-6,    # lower LR
    # log_with         = None,
    # kl_penalty       = "kl",
    # #kl_target        = 0.015,   # tighter KL (adaptive)
    # init_kl_coef     = 0.2)
    config = PPOConfig(
    batch_size       = 16,
    mini_batch_size  = 2,
    learning_rate    = 5e-6,    # lower LR
    log_with         = None,
    kl_penalty       = "abs",
    #kl_target        = 0.05,   # tighter KL (adaptive)
    init_kl_coef     = 0.02)

    def reward_fn_rank(samples, responses, **kw):
        global DEBUG_ONCE
        rewards = []
        MOT_lines = samples["query"]

        for mot_line, baby_ids in zip(MOT_lines, responses):
            # decode
            baby_text = tokenizer_baby.decode(baby_ids, skip_special_tokens=True).strip()
            baby_utt  = extract_first_chi_utterance(baby_text) or baby_text

            key = norm_key(mot_line)
            device = baby_ids.device

            # FALLBACK check
            if not baby_utt or key not in mom_logps_dict:
                if DEBUG_ONCE:
                    print("FALLBACK:", {
                        "reason": "empty" if not baby_utt else "key_miss",
                        "key": repr(key)[:120]
                    })
                    DEBUG_ONCE = False

                rewards.append(safe_tensor(NEG_REWARD, device))
                continue

            mom_logps = mom_logps_dict[key]
            ell_baby  = lnP_mom(key, baby_utt, tok_mom, mom_model)

            rank   = sum(lp <= ell_baby for lp in mom_logps) / len(mom_logps)
            reward = 2 * rank - 1

            # sanity guards
            if DEBUG_ONCE:
                print("DEBUG SAMPLE:")
                print(" prompt key:", repr(key))
                print(" baby_utt  :", repr(baby_utt))
                print(" ell_baby  :", ell_baby)
                print(" mom_logps :", "min", min(mom_logps),
                    "max", max(mom_logps),
                    "mean", sum(mom_logps)/len(mom_logps))
                print(" rank/reward:", rank, reward)
                DEBUG_ONCE = False

            assert -1.01 <= reward <= 1.01, f"Reward out of range: {reward}"
            rewards.append(safe_tensor(reward, device))

        # final check
        stack = torch.stack(rewards)
        assert torch.isfinite(stack).all(), "NaN/inf in rewards!"
        return rewards


    # === 2. Load mother prompts ===
    with open("./txt_files/mother_prompts_1.txt", "r", encoding="utf-8") as f:
       raw_prompts = [line.strip() for line in f if line.strip()]

    mother_prompts_all = [norm_key(p) for p in raw_prompts]
    mother_prompts     = [p for p in mother_prompts_all if p in mom_logps_dict]

    print(f"Total prompts in file : {len(mother_prompts_all):,}")
    print(f"Prompts with Mom logps: {len(mother_prompts):,}")
    missing = [p for p in mother_prompts_all if p not in mom_logps_dict]

    training_prompts = []
    with open("./txt_files/mom_scores.jsonl") as jl:
        for line in jl:
            row = json.loads(line)
            training_prompts.append(norm_key(row["prompt"]))

    print("Prompts with Mom log‑ps:", len(training_prompts))          # ≈ 115 k
    train_dataset = Dataset.from_dict({"query": training_prompts})

    print(f"Total prompts in file : {len(mother_prompts_all):,}")
    print(f"Prompts with Mom logps: {len(mother_prompts):,}")
    print("Missing prompts:", len(missing))
    if missing:
        for m in missing[:5]:
            print("MISSING:", repr(m))
    print("First raw prompt:", repr(mother_prompts[0]))
    print("First normed key:", repr(norm_key(mother_prompts[0])))
    print("In dict?", norm_key(mother_prompts[0]) in mom_logps_dict)




    # === PPO Setup ===
    
    ppo = PPOTrainer(
        config,
        model=baby_model,
        ref_model=ref_baby,
        tokenizer=tokenizer_baby,
        dataset=train_dataset,
    )
    clamp_lp = LogitsProcessorList([ClampLogits()])

    with open(log_rewards_file, "w") as f:
        f.write("epoch,batch,avg_reward\n")
        for epoch in range(3):
            ppo.dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn_olmo
            )

            for batch_idx, batch in enumerate(tqdm.tqdm(ppo.dataloader, desc=f"Epoch {epoch + 1}", leave=True)):
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
                if DEBUG_ONCE:
                    ex = tokenizer_baby.decode(baby_responses[0], skip_special_tokens=True)
                    print("Sample BABY raw gen:", repr(ex))

                # new confidence reward
                rewards = reward_fn_rank({"query": prompts_batch}, baby_responses)
                stats = ppo.step(tokenized_prompts, baby_responses, rewards)
                avg_reward = sum(r.item() for r in rewards) / len(rewards)
                if (batch_idx % 100) == 0:
                  print(f"[ep {epoch+1} batch {batch_idx+1}] avg_reward={avg_reward:.3f}")
                fallback_rate = (torch.stack(rewards) == NEG_REWARD).float().mean().item()
                if (batch_idx % 200) == 0:   # every 200 batches ≈ every 3 k prompts
                    print(f"[ep {epoch+1}  b {batch_idx+1:>4}] "
                        f"avg={avg_reward:+.3f}  fallback={fallback_rate:.2%}")



                f.write(f"{epoch+1},{batch_idx+1},{avg_reward:.4f}\n")
                f.flush()

                global_step += 1
                if global_step % SAVE_EVERY == 0:
                    output_dir = f"./fine_tuned_models/{repo_name.split('/')[-1]}/checkpoint-{global_step}"
                    ppo.model.save_pretrained(output_dir)
                    tokenizer_baby.save_pretrained(output_dir)
                    shutil.rmtree(output_dir)

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