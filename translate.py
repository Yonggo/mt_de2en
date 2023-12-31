import keras
import numpy as np
from tqdm import tqdm

from helper import *


def convert_to_text(pred_in_vec, dictionary):
    words = [dictionary[np.argmax(word_candidates)] for word_candidates in pred_in_vec[0]]
    return " ".join(words)

def translate(path_de, path_model):
    model = keras.models.load_model(path_model)
    print("==> Used Model: {}".format(path_model))

    with open(path_de, "r", encoding="utf8") as file:
        sentences = file.readlines()

    voca_de = load_voca("data/de_voca.pickle")
    voca_en = load_voca("data/en_voca.pickle")

    vec2word_en = {id: word for word, id in voca_en.word_index.items()}
    vec2word_en[0] = '<PAD>'

    word_embedding_inputs = []
    for sentence in sentences:
        sent = []
        for word in sentence.split():
            try:
                sent.append(voca_de.word_index[word])
            except KeyError:
                try:
                    sent.append(voca_de.word_index[word.lower()])
                except KeyError:
                    pass
        word_embedding_inputs.append(sent)

    word_embedding_inputs = pad(word_embedding_inputs, model.input_shape[1])
    predictions = []
    for input in tqdm(word_embedding_inputs, desc="Translating " + path_de):
        predicted = model.predict(input.reshape(1, -1), verbose=False)
        predictions.append(convert_to_text(predicted, vec2word_en))

    return predictions


if __name__ == '__main__':
    outputs = translate("data/test/example.europarl.de.test", "models/model-04-0.9535.keras")
    translations = [" ".join(words) for words in outputs]
    results = []
    for translation in translations:
        results.append(translation)
    print("================= Result =====================")
    print(results)
    save_data("data/output.txt", results)
