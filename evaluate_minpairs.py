import pandas as pd
from minicons import scorer
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

# Load minimal pairs from CSV
data = pd.read_csv("./dpo_dataset/len_pairs_no_overlap_1.csv")
print(f"Loaded {len(data)} pairs.")

# Initialize your models
baseline_path = "bbunzeck/another-llama"

# Load MiniCONS models
baseline_model = scorer.IncrementalLMScorer(baseline_path, device='cuda' if torch.cuda.is_available() else 'cpu')

# Evaluate a model on the dataset
def evaluate_model(model, data, pos_col="pospair", neg_col="negpair"):
    correct = 0
    total = len(data)
    
    for _, row in tqdm(data.iterrows(), total=total):
        pos_sent = row[pos_col]
        neg_sent = row[neg_col]

        pos_score = model.sequence_score(pos_sent)
        neg_score = model.sequence_score(neg_sent)

        # Higher score (logprob) = better
        if pos_score > neg_score:
            correct += 1

    accuracy = correct / total
    return accuracy

# Run evaluations
print("🔍 Evaluating baseline model...")
baseline_acc = evaluate_model(baseline_model, data)
print(f"✅ Baseline model accuracy: {baseline_acc:.3f}")

'''print("🔍 Evaluating fine-tuned model...")
finetuned_acc = evaluate_model(finetuned_model, data)
print(f"✅ Fine-tuned model accuracy: {finetuned_acc:.3f}")'''
