from translate import *
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings('ignore')

reference_text_file = "data/test/example.europarl.en.test"
to_be_translated_file = "data/test/example.europarl.de.test"
model_path = "models/model-30-0.9031.keras"

if __name__ == '__main__':
    with open(reference_text_file, "r", encoding="utf8") as file:
        sentences_en = file.readlines()
    references = [[[words for words in sentence.split()]] for sentence in sentences_en]
    predictions = translate(to_be_translated_file, model_path)
    #predictions = [[words for words in sentence.split()] for sentence in sentences_en]
    if len(predictions) != len(references):
        raise Exception("Size of predictions is different from size of references")

    total_score = 0
    for idx in range(len(predictions)):
        total_score += sentence_bleu(references[idx], predictions[idx])

    mean_score = round(total_score, 4) / len(predictions)

    print("BLEU-Score: {}".format(mean_score))