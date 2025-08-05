import pandas as pd
from datasets import load_dataset
from minicons import scorer
import torch
from tqdm import tqdm

# === Model paths ===
BASELINE_PATH = "bbunzeck/another-llama"
BLUE = "/Users/frapadovani/Desktop/communicative_baby_dpo/dpo_outputs_complete_synthetic/checkpoints/checkpoint-5630" 
SEMSIM = "/Users/frapadovani/Desktop/communicative_baby_dpo/dpo_outputs_complete/checkpoints/checkpoint-5630"
SCORE = "/Users/frapadovani/Desktop/communicative_baby_dpo/dpo_outputs_complete_synthetic/checkpoints/checkpoint-5630" 
UNCERTAINTY = "/Users/frapadovani/Desktop/communicative_baby_ppo/fine_tuned_models/babylm-conf/epoch-1"  

# === Load the dataset ===
print("Loading lexical-decision dataset from HuggingFace...")
dataset = load_dataset("bbunzeck/lexical-decision", split="train")
print(f"Loaded {len(dataset)} samples.")

# Convert to list of dicts for easier iteration
data = dataset.to_list()

# === Load MiniCONS models ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

baseline_model = scorer.IncrementalLMScorer(BASELINE_PATH, device=device)
finetuned_model_1 = scorer.IncrementalLMScorer(BLUE, device=device)
finetuned_model_2 = scorer.IncrementalLMScorer(SEMSIM, device=device)
finetuned_model_3 = scorer.IncrementalLMScorer(BLUE, device=device)
finetuned_model_4 = scorer.IncrementalLMScorer(SEMSIM, device=device)

# === Evaluation function ===
def evaluate_lexical_decision_model(model, data):
    correct = 0
    total = len(data)

    for row in tqdm(data):
        lexeme = row["lexeme"]   # real word
        wug = row["wug"]         # nonword

        # Skip pairs with missing or empty inputs
        if not lexeme or not wug:
            continue

        try:
            real_score = model.sequence_score(lexeme, reduction=lambda x: x.sum(0).item(), bow_correction=True)
            wug_score = model.sequence_score(wug, reduction=lambda x: x.sum(0).item(), bow_correction=True)
        except Exception as e:
            continue

        # Correct if real word is scored higher (more probable)
        if real_score > wug_score:
            correct += 1

    return correct / total

# === Run evaluations ===
print("\ns Evaluating models on lexical decision task...\n")

acc_baseline = evaluate_lexical_decision_model(baseline_model, data)
print(f" Baseline model accuracy: {acc_baseline:.3f}")

acc_ft1 = evaluate_lexical_decision_model(finetuned_model_1, data)
print(f"Fine-tuned model 1 accuracy: {acc_ft1:.3f}")

acc_ft2 = evaluate_lexical_decision_model(finetuned_model_2, data)
print(f"Fine-tuned model 2 accuracy: {acc_ft2:.3f}")