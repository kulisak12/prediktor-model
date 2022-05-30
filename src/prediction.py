#!/usr/bin/env python3

from typing import List, Tuple
import argparse
import numpy as np
import os

# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--k", default=3, type=int, help="Number of suggestions.")
parser.add_argument("--model", default="prediktor.model", type=str, help="Path to trained model.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def tokens_from_sentence(sentence: str) -> tf.Tensor:
    sentence_tensor = tf.constant(sentence, dtype=tf.string)
    tokens = tf.strings.split(sentence_tensor, sep=" ")
    return tokens


def get_prediction_probs(model: tf.keras.models.Model, x: tf.Tensor) -> tf.Tensor:
    """Run prediction and return probabilities.

    Args:
    x: Input tensor for one example.
    """
    x = tf.expand_dims(x, axis=0)  # batch
    probs = model(x)
    probs = tf.squeeze(probs, axis=0)  # unbatch

    # don't predict [UNK]
    vocab_size = tf.shape(probs)[-1]
    unk_mask = tf.one_hot(0, vocab_size, on_value=np.inf)
    probs -= unk_mask

    probs = tf.nn.softmax(probs, axis=-1)  # model returns logits
    return probs


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    model = tf.keras.models.load_model(args.model)
    word_mapping = model.layers[1]
    reverse_word_mapping = tf.keras.layers.StringLookup(
        vocabulary=word_mapping.get_vocabulary(),
        invert=True,
        mask_token=None
    )

    while sentence := input():
        tokens = tokens_from_sentence(sentence)
        probs = get_prediction_probs(model, tokens)
        probs = probs[-1]
        top = tf.math.top_k(probs, k=args.k)
        words = reverse_word_mapping(top.indices)
        words = map(lambda x: x.decode(), words.numpy())
        print(" ".join(words))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
