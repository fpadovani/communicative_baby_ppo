from minicons import scorer
from pathlib import Path

def evaluate_zorro(lm, test_suite_folder, lower_case=False):

    paradigm_accuracies = {}
    test_suite_folder = Path(test_suite_folder)

    for path_paradigm in test_suite_folder.glob('*.txt'):
        print(f"Processing: {path_paradigm.name}")

        # Load and preprocess sentences
        sentences_ = path_paradigm.read_text().strip().split('\n')
        assert len(sentences_) % 2 == 0, f"File {path_paradigm.name} must have an even number of lines!"
        sentences = [s.lower() for s in sentences_] if lower_case else sentences_

        # Initialize counters
        correct_count = 0
        total_count = 0

        # Evaluate each minimal pair
        for i in range(0, len(sentences), 2):
            grammatical = sentences[i]
            ungrammatical = sentences[i + 1]

            # Score both sentences
            stimuli = [grammatical, ungrammatical]
            scores = lm.sequence_score(
                stimuli,
                reduction=lambda x: x.sum(0).item(),
                bow_correction=True
            )

            # Lower perplexity / surprisal means higher probability
            if scores[0] > scores[1]:
                correct_count += 1
            total_count += 1

        # Store accuracy per paradigm
        accuracy = correct_count / total_count
        paradigm_accuracies[path_paradigm.name] = accuracy
        print(f"Accuracy for {path_paradigm.name}: {accuracy:.2%}")
    
    # Compute and print overall average accuracy
    all_accuracies = list(paradigm_accuracies.values())
    overall_accuracy = sum(all_accuracies) / len(all_accuracies)
    print(f"\n✅ Overall BLiMP Accuracy: {overall_accuracy:.2%}")

    return paradigm_accuracies, overall_accuracy


BASELINE_PATH = "bbunzeck/another-llama"
BLUE = "./models/rfblue-kl/epoch-1/epoch-1"
SEMSIM = "./models/rfsem-kl/epoch-1/epoch-1"
SCORE = "./models/rfscore-kl/checkpoint-5000/checkpoint-5000"
UNCERTAINTY = "./models/rfconfig-baby/epoch-1/epoch-1"
zorro_folder = './evaluation/test_suites/zorro'

baseline_model = scorer.IncrementalLMScorer(BASELINE_PATH, device='cpu')
blue = scorer.IncrementalLMScorer(BLUE, device='cpu')
semsim = scorer.IncrementalLMScorer(SEMSIM, device='cpu')
score = scorer.IncrementalLMScorer(SCORE, device='cpu')
uncertainty = scorer.IncrementalLMScorer(UNCERTAINTY, device='cpu')

paradigm_acc_baseline, overall_acc_baseline = evaluate_zorro(
    lm=baseline_model,
    test_suite_folder=zorro_folder,
    lower_case=True
)
print(paradigm_acc_baseline, overall_acc_baseline)

paradigm_acc_finetuned1, overall_acc_finetuned1 = evaluate_zorro(
    lm=blue,
    test_suite_folder=zorro_folder,
    lower_case=True
)

print(paradigm_acc_finetuned1, overall_acc_finetuned1)

paradigm_acc_finetuned2, overall_acc_finetuned2 = evaluate_zorro(
    lm=semsim,
    test_suite_folder=zorro_folder,
    lower_case=True
)
print(paradigm_acc_finetuned2, overall_acc_finetuned2)

paradigm_acc_finetuned3, overall_acc_finetuned3 = evaluate_zorro(
    lm=score,
    test_suite_folder=zorro_folder,
    lower_case=True
)
print(paradigm_acc_finetuned3, overall_acc_finetuned3)

paradigm_acc_finetuned4, overall_acc_finetuned4 = evaluate_zorro(
    lm=uncertainty,
    test_suite_folder=zorro_folder,
    lower_case=True
)
print(paradigm_acc_finetuned4, overall_acc_finetuned4)



