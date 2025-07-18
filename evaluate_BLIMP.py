from minicons import scorer
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def evaluate_blimp(lm, test_suite_folder, lower_case=False):

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


BABY_baseline = "bbunzeck/another-llama"
BABY_fine_tuned = "./fine_tuned_models/rfblue1-baby-step-11000"
blimp_folder = './evaluation_zorro/test_suites/blimp'


lm = scorer.IncrementalLMScorer(BABY_baseline, device='cuda')

paradigm_acc, overall_acc = evaluate_blimp(
    lm=lm,
    test_suite_folder=blimp_folder,
    lower_case=True
)

print(paradigm_acc, overall_acc)

