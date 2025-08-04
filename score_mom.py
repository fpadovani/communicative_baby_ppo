"""Scores each of the 10 teacher replies with the *same* teacher model and
writes mom_scores.jsonl  (adds field  "mom_logps": [float, …]).
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_ppo import lnP_mom, extract_first_utterance   # lnP_mom shown below

TEACHER = "meta-llama/Llama-3.2-3B"
#TEACHER="mistralai/Mistral-7B-Instruct-v0.2"
tok_mom = AutoTokenizer.from_pretrained(TEACHER)
mom     = AutoModelForCausalLM.from_pretrained(
            TEACHER, torch_dtype=torch.float16, device_map="cuda")
mom.eval()

with open("./txt_files/responses_dataset_vllm_fast.json") as fin, \
     open("./txt_files/mom_scores.jsonl", "w") as fout:
    for line in fin:
        row = json.loads(line)
        prompt  = row["prompt"].replace("*MOT:", "").strip()
        logps = [lnP_mom(prompt, ans, tok_mom, mom) for ans in row["teacher_responses"]]
        row["mom_logps"] = logps
        fout.write(json.dumps(row) + "\n")