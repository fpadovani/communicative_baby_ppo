from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

BABY_baseline = "bbunzeck/another-llama"
BABY_fine_tuned = "./fine_tuned_models/rfblue1-baby-step-7000"
blimp_folder = './test_suites/blimp'


# Load tokenizer (they are likely the same for both models)
tokenizer = AutoTokenizer.from_pretrained(BABY_baseline)
tokenizer.pad_token = tokenizer.eos_token  # In case it's not set

# Load baseline model
model_baseline = AutoModelForCausalLM.from_pretrained(BABY_baseline)
pipeline_baseline = pipeline("text-generation", model=model_baseline, tokenizer=tokenizer)

# Load fine-tuned model
model_finetuned = AutoModelForCausalLM.from_pretrained(BABY_fine_tuned)
pipeline_finetuned = pipeline("text-generation", model=model_finetuned, tokenizer=tokenizer)

# Prompt example
prompt = "*MOT: did you get enough milk? *CHI:"

# Generate responses
output_baseline = pipeline_baseline(prompt, max_new_tokens=20, do_sample=True, top_p=0.9)[0]["generated_text"]
output_finetuned = pipeline_finetuned(prompt, max_new_tokens=20, do_sample=True, top_p=0.9)[0]["generated_text"]

# Print results
print("=== Baseline ===")
print(output_baseline)

print("\n=== Fine-tuned ===")
print(output_finetuned)






