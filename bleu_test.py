from translate import *
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings('ignore')

reference_text_file = "data/test/example.europarl.en.test"
to_be_translated_file = "data/test/example.europarl.de.test"
model_path = "models/model-04-0.9535.keras"

if __name__ == '__main__':
    with open(reference_text_file, "r", encoding="utf8") as file:
        sentences_en = file.readlines()
    references = [[[words for words in sentence.split()]] for sentence in sentences_en]
    predictions = translate(to_be_translated_file, model_path)
    if len(predictions) != len(references):
        raise Exception("Size of predictions is different from size of references")

    total_score = 0
    for idx in range(len(predictions)):
        prediction = predictions[idx].replace("<PAD>", "").strip().split()
        total_score += round(sentence_bleu(references[idx], prediction, weights=(1, 0, 0, 0)), 4)

    mean_score = round(total_score, 4) / len(predictions)

    print("Sum BLEU-Score : {}".format(total_score))
    print("Mean BLEU-Score: {}".format(mean_score))
