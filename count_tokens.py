import json
from pathlib import Path

PATH = Path("./txt_files/prompts/10_teacher_answers.json")

raw_data = []
with PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        raw_data.append(json.loads(line))

# optionally limit to first N
raw_data = raw_data[:150_000]

# Extract prompts (key name in your sample is "prompt")
prompts = [item.get("prompt", "") for item in raw_data]

# Flatten teacher_responses: support both list-of-strings and single string cases
teacher_refs = []
for item in raw_data:
    tr = item.get("teacher_responses", [])
    if isinstance(tr, list):
        # keep only strings, skip None/non-strings
        teacher_refs.extend([r for r in tr if isinstance(r, str)])
    elif isinstance(tr, str):
        teacher_refs.append(tr)
    # else: ignore missing/invalid formats

def count_whitespace_tokens(text):
    if not text:
        return 0
    # split on any whitespace (default .split())
    return len(text.split())

# Totals
total_prompt_words = sum(count_whitespace_tokens(p) for p in prompts)
total_teacher_words = sum(count_whitespace_tokens(r) for r in teacher_refs)
total_words = total_prompt_words + total_teacher_words

# Counts / averages
n_prompts = len(prompts)
n_teacher_responses = len(teacher_refs)
avg_words_per_prompt = total_prompt_words / n_prompts if n_prompts else 0
avg_words_per_teacher_response = total_teacher_words / n_teacher_responses if n_teacher_responses else 0

print(f"Prompts: {n_prompts}, Teacher responses (flattened): {n_teacher_responses}")
print(f"Total words in prompts: {total_prompt_words:,} (avg {avg_words_per_prompt:.2f} per prompt)")
print(f"Total words in teacher responses: {total_teacher_words:,} (avg {avg_words_per_teacher_response:.2f} per response)")
print(f"Combined total words: {total_words:,}")

##############################################

import json
from pathlib import Path

# Path to your JSONL file
PATH = Path("./txt_files/prompts/best_teacher_answer.json")

total_prompt_words = 0
total_best_response_words = 0
n_lines = 0
limit = 220_000  # only first 220k rows

with PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if n_lines >= limit:
            break
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        prompt = obj.get("original_prompt", "")
        best_resp = obj.get("best_response", "")

        total_prompt_words += len(prompt.split())
        total_best_response_words += len(best_resp.split())
        n_lines += 1

print(f"Number of entries counted: {n_lines}")
print(f"Total words in original_prompt: {total_prompt_words:,}")
print(f"Total words in best_response: {total_best_response_words:,}")
print(f"Combined total words: {total_prompt_words + total_best_response_words:,}")
