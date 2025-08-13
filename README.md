# BabyLM Challenge 2025 - PPO fine-tuning for a Baby Model 
To run correctly the code in this repository you need this version of trl==0.8.6 (as specified in the requirements.txt file). 

## Model
As baseline, we use the a model pre-trained on dialogue turns between a child and a caregiver  -> [Llamalogue](https://huggingface.co/CLAUSE-Bielefeld/llamalogue/tree/main)

## PPO Datasets

Our PPO pipeline requires to have some real mother prompts to be provided either to our pre-trained Baby model or to our teacher LLM model.
We used the portion of dialogue data that was generated during CHILDES data pre-processing, but not used for pre-training of llamalogue and we extracted *MOT: utterances and questions that are longer than 3 tokens (in order to have meaningful sentences). 

We used the `answers_generation_vllm_fast.py` script to prompt a teacher model -> Llama-3.2-3B asking to simulate real and natural sounding child answers to a mother utterance:
<pre><code> "You are a young child having a conversation with your mother. "
"When your mother says something, you should answer as a typical and natural-sounding child. "
"Do NOT repeat her words. Instead, give a new, relevant answer that shows understanding. "
"Keep it short and child-like."
</code></pre>

We collected 10 different answers that are used to calculate the Confidence reward (refer to our papers for more details) here -> [fpadovani/10_teacher_ground_truth](https://huggingface.co/datasets/fpadovani/10_teacher_ground_truth)
We collected only one LLM answer, used as ground truth for all the other reward functions (Bleu, Semantic Similarity and LLM-score) here -> [fpadovani/single_teacher_ground_truth](https://huggingface.co/datasets/fpadovani/single_teacher_ground_truth)

## PPO Training

- Use `ppo_training_sem.py` 

### Reward Functions 

1. UNIGRAM BLEU (based on the unigram token overlap between child and teacher utterances)
2. SEMANTIC SIMILARITY (cosine similarity calculated using this model `embedder = SentenceTransformer('all-MiniLM-L6-v2')`


You can fine-tune with PPO the baseline model using this command and providing semsim or bleu as reward_fn function:

<pre><code> python ppo_training_blue_semsim --reward_fn semsim </code></pre>

3. The third type of reward is directly assigned by the LLM (allenai/OLMo-2-1124-7B-Instruct) model, that is asked to score the answer of the baby model to a mother prompt. This is the prompt I used:

<pre><code> 
"<|system|>\nYou are presented with a dialogue between a mother (MOT) and a child (CHI). "
"Please rate how contextually appropriate and fluent the child's response is, on a scale from 0 (completely unfitting) "
"to 5 (perfectly fine answer). If CHI answer is too short rate it low.\n<|end|>\n"
f"<|user|>\nMOT: {mot}\nCHI: {chi}\n<|end|>\n<|assistant|>\n"
</code></pre>

I have two script to run this PPO Training, one is slower than the other (that uses VLLM).

- `ppo_training_score.py`
- `ppo_training_score_faster.py`


The fine-tuned models can be found here:
- [fpadovani/rfblue-baby](https://huggingface.co/fpadovani/rfblue-baby)
- [fpadovani/rfsem1-baby](https://huggingface.co/fpadovani/rfsem1-baby)
- [fpadovani/rfscore-baby](https://huggingface.co/fpadovani/rfscore-baby)

## Evaluation with DPO
We should familiarize with the BabyLM Challenge evaluation pipeline of this year -> [2025](https://github.com/babylm/evaluation-pipeline-2025)

In the meantime I have scripts that evaluate our baseline and finetuned models on Zorro, on our own minimal dialogue pair dataset (with words matched length and token matched length) and on single lexical items (taken from Bastian lexical decision task paper -> [bbunzeck/lexical-decision](https://huggingface.co/datasets/bbunzeck/lexical-decision):

- *`./evaluation/evaluate_zorro.py`* 
- *`./evaluation/evaluate_dialogue_minpairs.py`*
- *`./evaluation/evaluate_lexicon.py`*


**As soon as I have the final results I will post them here!**

**BASELINE**: our *bbunzeck/another-llama* baseline model scores **65.5%%** (accuracy) on Zorro and **64.3%** on the minimal pairs evaluation set based on words match, and **63.8%** on dialogue minimal pairs based on tokens match. It scores **40.3%** on the lexical decision task. \

**BLUE first EPOCH**: the last checkpoint of our fine-tuned model on real dpo pairs scores **62.48%** on Zorro and **62%** on the minimal pairs evaluation set based on words match, and **61%** on dialogue minimal pairs based on tokens match. It scores **40.7%** on the lexical decision task. \

**SEMSIM first EPOCH**: the last checkpoint of our fine-tuned model on real dpo pairs scores **64.68%** on Zorro and **61.1%** on the minimal pairs evaluation set based on words match, and **63.6%** on dialogue minimal pairs based on tokens match. It scores **39.7%** on the lexical decision task. \

**SCORE first EPOCH**: the last checkpoint of our fine-tuned model on real dpo pairs scores **65.18%%** on Zorro and **60.6%%** on the minimal pairs evaluation set based on words match, and **62.5%** on dialogue minimal pairs based on tokens match. It scores **40.2%** on the lexical decision task.\

**UNCERTAINTY first EPOCH**: the last checkpoint of our fine-tuned model on real dpo pairs scores **...%** on Zorro and **63.7%** on the minimal pairs evaluation set based on words match, and **...%** on dialogue minimal pairs based on tokens match. \ It scores **40.8%** on the lexical decision task. \



## Plots of reward 
In the `./plots` folder you can find the reward trend. The plots that are present do not refer to a complete training.
I will upload the final ones as soon as I have the fully fine-tuned models. The important thing is the growing reward ;)






