#!/usr/bin/env python3

from typing import List, Tuple
import argparse
import collections
import datetime
import numpy as np
import os
import pickle
import re

# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--data_dir", default="data/", type=str, help="Path to directory with datasets.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")


SEQ_LENGTH = 20


class PredictionModel(tf.keras.Model):
    """RNN-based model which predicts next word in sentence."""

    # According to:
    # https://www.tensorflow.org/text/tutorials/text_generation

    def __init__(self, word_mapping: tf.keras.layers.StringLookup) -> None:
        inputs = tf.keras.Input(shape=[SEQ_LENGTH - 1], dtype=tf.string)
        features = word_mapping(inputs)
        hidden = tf.keras.layers.Embedding(word_mapping.vocabulary_size(), args.we_dim)(features)
        hidden = tf.keras.layers.GRU(args.rnn_dim, return_sequences=True)(hidden)
        outputs = tf.keras.layers.Dense(word_mapping.vocabulary_size())(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        # Compile the model
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

        self.word_mapping = word_mapping


def get_vocabulary(filename: str) -> List[str]:
    """Get most common words in a text file.

    The result is cached and reused on sequent runs.
    """

    CACHE_FILE = "vocab.pickle"
    VOCAB_SIZE = 10000

    # use cached data
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    # count frequency of each word in file
    counter = collections.Counter()
    with open(filename) as f:
        for line in f:
            counter.update(line.split(" "))

    vocab = counter.most_common(VOCAB_SIZE)
    vocab = list(map(lambda x: x[0], vocab))

    # cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(vocab, f)
    return vocab


def prepare_dataset(
    filename: str,
    word_mapping: tf.keras.layers.StringLookup,
    train: bool = False
) -> tf.data.Dataset:
    """Create dataset from a text file.

    The text file is expected to be preprocessed, one line per sentence, with tokens separated by spaces.
    """
    def examples_from_line(line: tf.string) -> tf.Tensor:
        """Create examples with a fixed number of tokens."""
        tokens = tf.strings.split(line, sep=" ")
        num_tokens = tf.shape(tokens)[0]
        # drop remainder
        num_tokens -= num_tokens % SEQ_LENGTH
        num_examples = num_tokens // SEQ_LENGTH
        tokens = tokens[:num_tokens]
        # split into examples
        examples = tf.reshape(tokens, [num_examples, SEQ_LENGTH])
        return examples


    def add_targets(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # targets are inputs shifted by one
        inputs = x[:-1]
        targets = x[1:]
        targets = word_mapping(targets)
        return inputs, targets

    ds = tf.data.TextLineDataset(filename)
    if train:
        ds = ds.shuffle(buffer_size=10000, seed=args.seed)

    ds = ds.map(examples_from_line)
    ds = ds.unbatch()  # unroll examples into one tensor
    ds = ds.map(add_targets)
    ds = ds.batch(args.batch_size)
    return ds


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data

    TRAIN_FILENAME = os.path.join(args.data_dir, "train.txt")
    DEV_FILENAME = os.path.join(args.data_dir, "dev.txt")

    vocab = get_vocabulary(TRAIN_FILENAME)
    word_mapping = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)

    # Train

    model = PredictionModel(word_mapping)

    train = prepare_dataset(TRAIN_FILENAME, word_mapping, train=True)
    dev = prepare_dataset(DEV_FILENAME, word_mapping)

    try:
        model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])
    except KeyboardInterrupt:  # if patience runs out
        print()

    model.save(os.path.join(args.logdir, "prediktor.model"))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
