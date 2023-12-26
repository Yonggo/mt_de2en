from translate import *
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    with open("data/test/example.europarl.en.test", "r", encoding="utf8") as file:
        sentences_en = file.readlines()
    references = [[[words for words in sentence.split()]] for sentence in sentences_en]
    predictions = translate("data/test/example.europarl.de.test", "models/model-30-0.9031.keras")
    #predictions = [[words for words in sentence.split()] for sentence in sentences_en]
    if len(predictions) != len(references):
        raise Exception("Size of predictions is different from size of references")

    total_score = 0
    for idx in range(len(predictions)):
        total_score += sentence_bleu(references[idx], predictions[idx])

    mean_score = round(total_score, 4) / len(predictions)

    print("BLEU-Score: {}".format(mean_score))