from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pickle

with open('output.pkl', 'rb') as f:
    model_output = pickle.load(f)

with open('gold.pkl', 'rb') as f:
    reference_output = pickle.load(f)

reference = [ [ref] for ref in reference_output]
# Calculate BLEU score

print(model_output[4], "\n", reference_output[4])

scores = []
with open('testbleu.txt', 'w') as f:
    for sentence, target in zip(model_output, reference_output):
        score = sentence_bleu([target], sentence)
        scores.append(score)
        f.write(f"{sentence}\t{score}")

print(f'BLEU score: {sum(scores)/len(scores)}')
