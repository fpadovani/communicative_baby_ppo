# BabyLM Challenge 2025 
### *PPO fine-tuning for a Communicative Baby Model* 
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

- Use `ppo_training_blue_semsim.py --reward_fn bleu/semsim` to fine-tune with Bleu or Semantic Similarity Reward functions
- Use `ppo_training_score.py` to fine-tune with OLMo-generated rewards
- Use `ppo_training_conf.py` to fine-tune with Confidence scores as rewards


The fine-tuned models can be found here:
- [CLAUSE-Bielefeld/communicative-baby-rfbleu](https://huggingface.co/CLAUSE-Bielefeld/communicative-baby-rfbleu)
- [CLAUSE-Bielefeld/communicative-baby-rfsemsim](https://huggingface.co/CLAUSE-Bielefeld/communicative-baby-rfsemsim)
- [CLAUSE-Bielefeld/communicative-baby-rfolmo_score](https://huggingface.co/CLAUSE-Bielefeld/communicative-baby-rfolmo_score)
- [CLAUSE-Bielefeld/communicative-baby-rfconfidence](https://huggingface.co/CLAUSE-Bielefeld/communicative-baby-rfconfidence)

## Datasets for Evaluation 

- this is the dataset split for evaluation with appropriate and random sentence matched in terms of word length -> [**dpo_dataset/huggingface_dpo_format_eval.json**](https://huggingface.co/datasets/fpadovani/dialogue_eval_words) \
- this is the dataset split for evaluation with appropriate and random sentence matched in terms of token length -> [**dpo_dataset/huggingface_dpo_format_eval_tokens.json**](https://huggingface.co/datasets/fpadovani/dialogue_eval_tokens) \

## Evaluation

Scripts that evaluate our baseline and finetuned models on Zorro, on our own minimal dialogue pair dataset (with words matched length and token matched length) and on single lexical items:

- *`./evaluation/evaluate_zorro.py`* 
- *`./evaluation/evaluate_dialogue_minpairs.py`*
- *`./evaluation/evaluate_lexicon.py`*


## Plots of reward 
In the `./plots` folder you can find the reward trend plots. 









