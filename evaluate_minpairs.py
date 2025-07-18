import pandas as pd
from datasets import load_dataset
from minicons import scorer
import torch
from tqdm import tqdm

# CONFIG: Choose your model and dataset
DATASET_PATH = "fpadovani/child-dpo-preferences-eval"  # HF dataset repo
BASELINE_PATH = "bbunzeck/another-llama"
FINETUNED_PATH = "" #choose among the fine-tuned models [fpadovani/rfblue1-baby, fpadovani/rfsem1-baby, fpadovani/rfscore1-baby]
SPLIT = 'train'



dataset = load_dataset(DATASET_PATH, split=SPLIT)


# Convert to list of dicts for iteration
data = dataset.to_list()

# Load MiniCONS models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
baseline_model = scorer.IncrementalLMScorer(BASELINE_PATH, device=device)
finetuned_model = scorer.IncrementalLMScorer(FINETUNED_PATH, device=device)

# Evaluation function
def evaluate_model(model, data):
    correct = 0
    total = len(data)

    for row in tqdm(data):
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        pos_input = prompt + " " + chosen
        neg_input = prompt + " " + rejected

        pos_score = model.sequence_score(pos_input)
        neg_score = model.sequence_score(neg_input)

        if pos_score > neg_score:
            correct += 1

    return correct / total




baseline_acc = evaluate_model(baseline_model, data)
print(f"Baseline model accuracy: {baseline_acc:.3f}")


finetuned_acc = evaluate_model(finetuned_model, data)
print(f"Fine-tuned model accuracy: {finetuned_acc:.3f}")
