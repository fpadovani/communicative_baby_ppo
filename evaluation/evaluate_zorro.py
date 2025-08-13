from minicons import scorer
from pathlib import Path
from huggingface_hub import list_repo_refs
import pandas as pd
import torch
import os

# === CONFIG ===
BLUE = "./models/rfbleu/epoch-1/epoch-1"
SEMSIM = "./models/rfsem-kl/epoch-1/epoch-1"
SCORE = "./models/rfscore-kl/checkpoint-5000/checkpoint-5000"
UNCERTAINTY = "./models/rfconfig-baby/epoch-1/epoch-1"
LLAMALOGUE = "bbunzeck/llamalogue"
ZORRO_FOLDER = './evaluation/test_suites/zorro'
RESULTS_CSV = "zorro_results.csv"

# === Evaluation function ===
def evaluate_zorro(lm, test_suite_folder, lower_case=False):
    paradigm_accuracies = {}
    test_suite_folder = Path(test_suite_folder)

    for path_paradigm in test_suite_folder.glob('*.txt'):
        print(f"Processing: {path_paradigm.name}")

        sentences_ = path_paradigm.read_text().strip().split('\n')
        assert len(sentences_) % 2 == 0, f"File {path_paradigm.name} must have an even number of lines!"
        sentences = [s.lower() for s in sentences_] if lower_case else sentences_

        correct_count = 0
        total_count = 0

        for i in range(0, len(sentences), 2):
            grammatical = sentences[i]
            ungrammatical = sentences[i + 1]

            stimuli = [grammatical, ungrammatical]
            scores = lm.sequence_score(
                stimuli,
                reduction=lambda x: x.sum(0).item(),
                bow_correction=True
            )

            if scores[0] > scores[1]:
                correct_count += 1
            total_count += 1

        accuracy = correct_count / total_count
        paradigm_accuracies[path_paradigm.name] = accuracy
        print(f"Accuracy for {path_paradigm.name}: {accuracy:.2%}")
    
    all_accuracies = list(paradigm_accuracies.values())
    overall_accuracy = sum(all_accuracies) / len(all_accuracies)
    print(f"\n✅ Overall BLiMP Accuracy: {overall_accuracy:.2%}")

    return paradigm_accuracies, overall_accuracy

# === Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Helpers for CSV incremental save and load ===
def load_existing_results(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame()

def append_result_to_csv(filename, row_dict):
    df = pd.DataFrame([row_dict])
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=header, index=False)

# Load existing results to avoid duplicates
existing_results = load_existing_results(RESULTS_CSV)

# === Evaluate local models ===
models_to_eval = [
    ("Blue", BLUE),
    ("SemSim", SEMSIM),
    ("Score", SCORE),
    ("Uncertainty", UNCERTAINTY),
]

for model_name, path in models_to_eval:
    if not existing_results.empty and (existing_results['Model'] == model_name).any():
        print(f"Skipping {model_name}, already evaluated.")
        continue

    print(f"\n📌 Evaluating {model_name}")
    model = scorer.IncrementalLMScorer(path, device=device)
    paradigm_acc, overall_acc = evaluate_zorro(model, ZORRO_FOLDER, lower_case=True)

    row = {
        "Model": model_name,
        "OverallAccuracy": overall_acc,
        **{f"Paradigm_{k}": v for k, v in paradigm_acc.items()}
    }

    append_result_to_csv(RESULTS_CSV, row)
    # Fix here: replace append() with pd.concat
    new_row_df = pd.DataFrame([row])
    existing_results = pd.concat([existing_results, new_row_df], ignore_index=True)

# === Evaluate HuggingFace llamalogue checkpoints ===
print(f"\n🔍 Listing checkpoints for {LLAMALOGUE}...")
refs = list_repo_refs(LLAMALOGUE)
branches = [b.name for b in refs.branches]

for branch in branches:
    model_name = f"llamalogue-{branch}"

    if not existing_results.empty and (existing_results['Model'] == model_name).any():
        print(f"Skipping {model_name}, already evaluated.")
        continue

    print(f"\n📌 Evaluating {model_name}")
    model = scorer.IncrementalLMScorer(LLAMALOGUE, device=device, revision=branch)
    paradigm_acc, overall_acc = evaluate_zorro(model, ZORRO_FOLDER, lower_case=True)

    row = {
        "Model": model_name,
        "OverallAccuracy": overall_acc,
        **{f"Paradigm_{k}": v for k, v in paradigm_acc.items()}
    }

    append_result_to_csv(RESULTS_CSV, row)
    # Fix here as well
    new_row_df = pd.DataFrame([row])
    existing_results = pd.concat([existing_results, new_row_df], ignore_index=True)

print(f"\n💾 Results saved incrementally to {RESULTS_CSV}")
