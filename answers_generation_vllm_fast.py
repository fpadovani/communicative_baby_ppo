from vllm import LLM, SamplingParams
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
from utils_ppo import extract_first_utterance, generate_teacher_responses_vllm_batch

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def check_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem = torch.cuda.mem_get_info(i)[0] / 1024**3
            total_mem = torch.cuda.mem_get_info(i)[1] / 1024**3
            used_mem = total_mem - free_mem
            print(f"GPU {i}: Used: {used_mem:.2f} GB, Free: {free_mem:.2f} GB, Total: {total_mem:.2f} GB")

# CONFI
TEACHER = "meta-llama/Llama-3.2-3B"
PROMPT_FILE = "./txt_files/mother_prompts_1.txt"
    

def main():
    
    print("Initial GPU memory:")
    check_gpu_memory()
    
    print("\nLoading vLLM model...")
    llm = LLM(
        model=TEACHER,
        dtype="float16",
        gpu_memory_utilization=0.8, 
        max_model_len=512,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        tensor_parallel_size=1,
    )
    
    print("After vLLM loading:")
    check_gpu_memory()
    
    print("\nLoading scorer model...")
    scorer_model = AutoModelForCausalLM.from_pretrained(
        TEACHER, 
        torch_dtype=torch.float16,
        device_map="auto", 
    )
    scorer_tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    
    print("After scorer loading:")
    check_gpu_memory()

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.80,
        max_tokens=20,
        n=10,
        stop=["\n", "Mother:", "Child:"],
        skip_special_tokens=True,
    )

    BATCH_SIZE = 16  # Reduced batch size to be safe

    # Load prompts
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = ['*MOT: ' + line.strip() for line in f if line.strip()]

    print(f"\nProcessing {len(prompts)} prompts in batches of {BATCH_SIZE}")

    with open('./txt_files/10_responses_1.json', "a", encoding="utf-8") as f:
        
        # Process in batches
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            batch_teacher_sents = [prompt.replace('*MOT:', "") for prompt in batch_prompts]
            
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(prompts) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            # Generate responses for entire batch
            batch_responses = generate_teacher_responses_vllm_batch(
                llm, batch_teacher_sents, sampling_params, scorer_model, scorer_tokenizer
            )
            
            # Process and write results
            for j, (prompt, teacher_texts) in enumerate(zip(batch_prompts, batch_responses)):
                teacher_utts = [extract_first_utterance(t) for t in teacher_texts]
                
                entry = {
                    "prompt": prompt,
                    "teacher_responses": teacher_utts
                }
                
                f.write(json.dumps(entry) + "\n")
            
            f.flush()
            print(f"Completed batch {i//BATCH_SIZE + 1}")

    print("Processing complete!")

if __name__ == '__main__':
    main()