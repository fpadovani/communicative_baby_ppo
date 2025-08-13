import pandas as pd
from datasets import load_dataset
from minicons import scorer
import torch
from tqdm import tqdm
from huggingface_hub import list_repo_refs

# === Model paths ===
BASELINE_PATH = "bbunzeck/another-llama"
BLUE = "./models/rfbleu/epoch-1/epoch-1"
SEMSIM = "./models/rfsem-kl/epoch-1/epoch-1"
SCORE = "./models/rfscore-kl/checkpoint-5000/checkpoint-5000"
UNCERTAINTY = "./models/rfconfig-baby/epoch-1/epoch-1"
LLAMALOGUE = "bbunzeck/llamalogue"

# === Load the dataset ===
print("🔄 Loading lexical-decision dataset from HuggingFace...")
dataset = load_dataset("bbunzeck/lexical-decision", split="train")
print(f"✅ Loaded {len(dataset)} samples.")

# Convert to list of dicts for easier iteration
data = dataset.to_list()

# === Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Evaluation function ===
def evaluate_lexical_decision_model(model, data):
    correct = 0
    total = len(data)
    for row in tqdm(data, leave=False):
        lexeme = row["lexeme"]   # real word
        wug = row["wug"]         # nonword

        if not lexeme or not wug:
            continue

        try:
            real_score = model.sequence_score(lexeme, reduction=lambda x: x.sum(0).item(), bow_correction=True)
            wug_score = model.sequence_score(wug, reduction=lambda x: x.sum(0).item(), bow_correction=True)
        except Exception:
            continue

        if real_score > wug_score:
            correct += 1

    return correct / total

# === Results storage ===
results = []

# === Evaluate local & baseline models ===
models_to_eval = [
    ("Baseline", BASELINE_PATH),
    ("Blue", BLUE),
    ("SemSim", SEMSIM),
    ("Score", SCORE),
    ("Uncertainty", UNCERTAINTY),
]

for model_name, path in models_to_eval:
    print(f"📌 Evaluating {model_name}")
    model = scorer.IncrementalLMScorer(path, device=device)
    acc = evaluate_lexical_decision_model(model, data)
    results.append({"Model": model_name, "Accuracy": acc})

# === Evaluate all checkpoints of llamalogue ===
print(f"\n🔍 Listing checkpoints for {LLAMALOGUE}...")
refs = list_repo_refs(LLAMALOGUE)
branches = [b.name for b in refs.branches]  # includes main + all checkpoints

for branch in branches:
    print(f"📌 Evaluating llamalogue ({branch})")
    model = scorer.IncrementalLMScorer(LLAMALOGUE, device=device, revision=branch)
    acc = evaluate_lexical_decision_model(model, data)
    results.append({"Model": f"llamalogue-{branch}", "Accuracy": acc})

# === Save results to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv("lexical_decision_results.csv", index=False)
print("\n💾 Results saved to lexical_decision_results.csv")
