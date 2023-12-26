import collections
import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler

from helper import *
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Embedding, Bidirectional, RepeatVector, CuDNNGRU, \
    CuDNNLSTM, LSTM, Dropout
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def tokenize(x):
    x_tk = Tokenizer(lower=False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk


def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[np.argmax(word_candidates)] for word_candidates in logits])


def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return round(lr*0.95**epoch, 4)


def embed_model(input_shape, output_sequence_length, german_vocab_size, english_vocab_size, batch_size, validation_split):
    embedding = Embedding(input_dim=german_vocab_size, output_dim=256, input_length=input_shape[1])
    logits = TimeDistributed(Dense(english_vocab_size, activation="softmax"))
    steps_for_decay = int(input_shape[0] * (1 - validation_split) / batch_size) + 1
    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=steps_for_decay*2,
        decay_rate=0.95,
        staircase=False)
    model = Sequential()
    model.add(embedding)
    model.add(Bidirectional(GRU(512, return_sequences=False, dropout=0.2)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(512, return_sequences=True, dropout=0.2)))
    model.add(logits)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr_schedule),
                  metrics=['accuracy'])

    return model


gpus = tf.config.list_physical_devices('GPU')
print('GPU Devices: ', gpus)
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# Loading training data
de_sentences = reduce_data(load_data('data/train/europarl-v7.de-en.de'), 0.2)
en_sentences = reduce_data(load_data('data/train/europarl-v7.de-en.en'), 0.2)
if len(de_sentences) != len(en_sentences):
    raise Exception("amount of training sentences is different: {} to {}".format(len(de_sentences), len(en_sentences)))
print('Dataset Loaded')
print("Training Sentences number: {}".format(len(de_sentences)))
print()
german_words_counter = collections.Counter([word for sentence in de_sentences for word in sentence.split()])
english_words_counter = collections.Counter([word for sentence in en_sentences for word in sentence.split()])
print('{} German words.'.format(len([word for sentence in de_sentences for word in sentence.split()])))
print('{} unique German words.'.format(len(german_words_counter)))
print('10 Most common words in the German dataset:')
print('"' + '" "'.join(list(zip(*german_words_counter.most_common(10)))[0]) + '"')
print()
print('{} English words.'.format(len([word for sentence in en_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()

# Tokenizing
print("Tokenizing...")
de_tokenized, de_tokenizer = tokenize(de_sentences)
en_tokenized, en_tokenizer = tokenize(en_sentences)
print("Saving vocabularies...")
save_voca("data/de_voca.pickle", de_tokenizer)
save_voca("data/en_voca.pickle", en_tokenizer)

# Pad Tokenized output
print("Padding...")
de_pad = pad(de_tokenized)
en_pad = pad(en_tokenized)

max_de_sequence_length = de_pad.shape[1]
max_en_sequence_length = en_pad.shape[1]
de_vocab_size = len(de_tokenizer.word_index)
en_vocab_size = len(en_tokenizer.word_index)
print('Data Preprocessed')
print("Max German sentence length:", max_de_sequence_length)
print("Max English sentence length:", max_en_sequence_length)
print("German vocabulary size:", de_vocab_size)
print("English vocabulary size:", en_vocab_size)


if __name__ == '__main__':
    # Callbacks
    stopping = EarlyStopping(monitor="val_accuracy", patience=5)
    filepath = "models/model-{epoch:02d}-{val_accuracy:.4f}.keras"
    #filepath = "models/model.keras"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_accuracy", verbose=1, mode='max')
    #lr_schedule_callback = LearningRateScheduler(lr_scheduler)

    de_pad = pad(de_pad, max_en_sequence_length)
    en_pad = en_pad.reshape(*en_pad.shape, 1)

    batch_size = 20
    validation_split = 0.05
    epochs = 20
    model = embed_model(de_pad.shape, max_en_sequence_length, de_vocab_size, en_vocab_size, batch_size, validation_split)
    #model = load_model("models/model.keras")
    model.fit(de_pad,
              en_pad,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split,
              callbacks=[checkpoint])

    print("==================== Simple TEST =====================")
    predicted = model.predict(de_pad[:1])[0]
    print("Prediction:")
    print(logits_to_text(predicted, en_tokenizer).replace("<PAD>", "").strip())
    print("Original:")
    print(en_sentences[0])
    print("==================== Model Summary ====================")
    model.summary()