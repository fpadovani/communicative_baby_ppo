import json
import argparse
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
import torch


with open('./txt_files/10_answers_2.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]
        
    results = []
        
    output_file = './txt_files/first_best_options.json'
    output_f = open(output_file, 'a', encoding='utf-8')
    
    for i, item in enumerate(data):
        
        mother_prompt = item['prompt']
        child_responses = item['teacher_responses']
        
        
        result = {
            'original_prompt': mother_prompt,
            'best_response': child_responses[1]
        }
        
        results.append(result)
        
        # Write result immediately if output file given
        if output_f:
            json.dump(result, output_f, ensure_ascii=False)
            output_f.write('\n')
            output_f.flush()
    
    if output_f:
        output_f.close()
        print(f"\nResults appended to {output_file}")