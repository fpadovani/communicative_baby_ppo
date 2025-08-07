import pandas as pd
from datasets import load_dataset
from minicons import scorer
import torch
from tqdm import tqdm

# CONFIG: Choose your model and dataset
DIALOGUE_WORDS= "fpadovani/dialogue_eval_words" 
DIALOGUE_TOKENS = "fpadovani/dialogue_eval_tokens" 
BASELINE_PATH = "bbunzeck/another-llama"
BLUE = "./models/rfblue-kl/epoch-1/epoch-1"
SEMSIM = "./models/rfsem-kl/epoch-1/epoch-1"
SCORE = "./models/rfscore-kl/checkpoint-5000/checkpoint-5000"
UNCERTAINTY = "./models/rfconfig-baby/epoch-1/epoch-1"
SPLIT = "train"

# Load dataset from HuggingFace
print(f"🔄 Loading HuggingFace evaluation datasets words: {DIALOGUE_WORDS}")
dataset_words = load_dataset(DIALOGUE_WORDS, split=SPLIT)
print(f"✅ Loaded {len(dataset_words)} samples.")

# Load dataset 2 from HuggingFace
print(f"🔄 Loading HuggingFace evaluation datasets tokens: {DIALOGUE_TOKENS}")
dataset_tokens = load_dataset(DIALOGUE_TOKENS, split=SPLIT)
print(f"✅ Loaded {len(dataset_tokens)} samples.")



# Convert to list of dicts for iteration
data_words = dataset_words.to_list()
data_tokens = dataset_tokens.to_list()

# Load MiniCONS models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
baseline_model = scorer.IncrementalLMScorer(BASELINE_PATH, device=device)
blue = scorer.IncrementalLMScorer(BLUE, device=device)
semsim = scorer.IncrementalLMScorer(SEMSIM, device=device)
score = scorer.IncrementalLMScorer(SCORE, device=device)
uncertainty = scorer.IncrementalLMScorer(UNCERTAINTY, device=device)

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

        pos_score = model.sequence_score(pos_input, reduction = lambda x: x.sum(0).item(), bow_correction=True)
        neg_score = model.sequence_score(neg_input, reduction = lambda x: x.sum(0).item(), bow_correction=True)

        if pos_score > neg_score:
            correct += 1

    return correct / total


#### WORDS MATCHED EVALUATION

finetuned_1_words = evaluate_model(blue, data_words)
print(f"Fine-tuned model accuracy: {finetuned_1_words:.3f}")

finetuned_2_words = evaluate_model(semsim, data_words)
print(f"Fine-tuned model accuracy: {finetuned_2_words:.3f}")

finetuned_3_words = evaluate_model(score, data_words)
print(f"Fine-tuned model accuracy: {finetuned_3_words:.3f}")

finetuned_4_words = evaluate_model(uncertainty, data_words)
print(f"Fine-tuned model accuracy: {finetuned_4_words:.3f}")

#### TOKENS MATCHED EVALUATION

finetuned_1_tokens = evaluate_model(blue, data_tokens)
print(f"Fine-tuned model accuracy: {finetuned_1_tokens:.3f}")

finetuned_2_tokens = evaluate_model(semsim, data_tokens)
print(f"Fine-tuned model accuracy: {finetuned_2_tokens:.3f}")

finetuned_3_tokens = evaluate_model(score, data_tokens)
print(f"Fine-tuned model accuracy: {finetuned_3_tokens:.3f}")

finetuned_4_tokens = evaluate_model(uncertainty, data_tokens)
print(f"Fine-tuned model accuracy: {finetuned_4_tokens:.3f}")