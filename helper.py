import os
import pickle
import keras
from keras.utils import pad_sequences


def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post')


def save_data(path, data_list):
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(data_list))


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding="utf8") as f:
        data = f.read()

    return data.split('\n')


def save_voca(path, voca):
    with open(path, 'wb') as f:
        pickle.dump(voca, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


def load_voca(path):
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def save_model(model, path):
    model.save(path)


def load_model(path):
    return keras.models.load_model(path)


def reduce_data(sentences, remain_rate):
    if remain_rate >= 1:
        return sentences

    slice_amount = 10000
    size = int(len(sentences)/slice_amount) + 2
    remain_amount = int(slice_amount * remain_rate)
    result = []
    start = 0
    for idx in range(1, size):
        end = start + remain_amount
        result += sentences[start:end]
        start = idx * slice_amount

    return result


def summary_model(path):
    model = keras.models.load_model(path)
    print(model.summary())


def load_model(path):
    return keras.models.load_model(path)


if __name__ == '__main__':
    summary_model("models.back/model.keras.h5")