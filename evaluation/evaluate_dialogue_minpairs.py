import os
import pandas as pd
from datasets import load_dataset
from minicons import scorer
import torch
from huggingface_hub import list_repo_refs
from tqdm import tqdm

# CONFIG
DIALOGUE_WORDS= "fpadovani/dialogue_eval_words" 
DIALOGUE_TOKENS = "fpadovani/dialogue_eval_tokens" 
BLUE = "./models/rfbleu/epoch-1/epoch-1"
SEMSIM = "./models/rfsem-kl/epoch-1/epoch-1"
SCORE = "./models/rfscore-kl/checkpoint-5000/checkpoint-5000"
LLAMALOGUE = "bbunzeck/llamalogue"
UNCERTAINTY = "./models/rfconfig-baby/epoch-1/epoch-1"
SPLIT = "train"

RESULTS_CSV = "evaluation_results.csv"
CHECKPOINTS_CSV = "evaluation_results_checkpoints.csv"

# Load datasets
print(f"🔄 Loading datasets: {DIALOGUE_WORDS} and {DIALOGUE_TOKENS}")
dataset_words = load_dataset(DIALOGUE_WORDS, split=SPLIT)
dataset_tokens = load_dataset(DIALOGUE_TOKENS, split=SPLIT)
print(f"✅ Loaded {len(dataset_words)} words samples and {len(dataset_tokens)} tokens samples.")

data_words = dataset_words.to_list()
data_tokens = dataset_tokens.to_list()

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
blue = scorer.IncrementalLMScorer(BLUE, device=device)
semsim = scorer.IncrementalLMScorer(SEMSIM, device=device)
score = scorer.IncrementalLMScorer(SCORE, device=device)
uncertainty = scorer.IncrementalLMScorer(UNCERTAINTY, device=device)

def evaluate_model(model, data):
    correct = 0
    total = len(data)
    for row in tqdm(data):
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        pos_input = prompt + " " + chosen
        neg_input = prompt + " " + rejected

        pos_score = model.sequence_score(pos_input, reduction = lambda x: x.sum(0).item(), bow_correction=True)
        neg_score = model.sequence_score(neg_input, reduction = lambda x: x.sum(0).item(), bow_correction=True)

        if pos_score > neg_score:
            correct += 1
    return correct / total

def append_result_to_csv(filename, new_row):
    # Append a single row dict to csv, creating file if needed
    df = pd.DataFrame([new_row])
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=header, index=False)

def load_existing_results(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=["Model", "Dataset", "Accuracy"])

# === Evaluate predefined models and write inline ===
existing_results = load_existing_results(RESULTS_CSV)

models_to_eval = [
    ("Blue", blue),
    ("SemSim", semsim),
    ("Score", score),
    ("Uncertainty", uncertainty),
]

datasets = [
    ("Words", data_words),
    ("Tokens", data_tokens),
]

for model_name, model_obj in models_to_eval:
    for dataset_name, dataset_data in datasets:
        # Check if result already exists
        already_done = ((existing_results["Model"] == model_name) & (existing_results["Dataset"] == dataset_name)).any()
        if already_done:
            print(f"Skipping {model_name} on {dataset_name}, already evaluated.")
            continue

        print(f"Evaluating {model_name} on {dataset_name}...")
        accuracy = evaluate_model(model_obj, dataset_data)
        print(f"Result: {accuracy:.3f}")

        # Append result immediately
        append_result_to_csv(RESULTS_CSV, {"Model": model_name, "Dataset": dataset_name, "Accuracy": accuracy})
        # Update in-memory results using pd.concat instead of append()
        new_row_df = pd.DataFrame([{"Model": model_name, "Dataset": dataset_name, "Accuracy": accuracy}])
        existing_results = pd.concat([existing_results, new_row_df], ignore_index=True)

print(f"\n💾 Results saved incrementally to {RESULTS_CSV}")

# === Evaluate HuggingFace repo checkpoints and write inline ===
existing_checkpoints_results = load_existing_results(CHECKPOINTS_CSV)

print(f"🔍 Listing checkpoints for {LLAMALOGUE}")
refs = list_repo_refs(LLAMALOGUE)
branches = [b.name for b in refs.branches]

for branch in branches:
    for dataset_name, dataset_data in datasets:
        model_name = f"llamalogue-{branch}"

        # Check if already evaluated
        already_done = ((existing_checkpoints_results["Model"] == model_name) & (existing_checkpoints_results["Dataset"] == dataset_name)).any()
        if already_done:
            print(f"Skipping {model_name} on {dataset_name}, already evaluated.")
            continue

        print(f"\n📌 Evaluating {model_name} on {dataset_name}...")
        hf_model = scorer.IncrementalLMScorer(LLAMALOGUE, device=device, revision=branch)
        accuracy = evaluate_model(hf_model, dataset_data)
        print(f"Result: {accuracy:.3f}")

        # Append result immediately
        append_result_to_csv(CHECKPOINTS_CSV, {"Model": model_name, "Dataset": dataset_name, "Accuracy": accuracy})
        # Update in-memory results using pd.concat instead of append()
        new_row_df = pd.DataFrame([{"Model": model_name, "Dataset": dataset_name, "Accuracy": accuracy}])
        existing_checkpoints_results = pd.concat([existing_checkpoints_results, new_row_df], ignore_index=True)

print(f"\n💾 Checkpoint results saved incrementally to {CHECKPOINTS_CSV}")
