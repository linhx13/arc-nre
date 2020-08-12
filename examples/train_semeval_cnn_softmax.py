import os
import argparse

import arcnre.tf


def run_train(args):
    data_handler = arcnre.tf.data.SentenceREDataHandler(
        text_field=arcnre.tf.data.TextField(max_len=128),
        pos_field=arcnre.tf.data.PositionField(max_len=128),
        label_field=arcnre.tf.data.LabelField())

    train_path = os.path.expanduser(args.train_path)
    val_path = os.path.expanduser(args.val_path)

    train_examples = data_handler.read_examples(train_path)
    data_handler.build_vocab(train_examples)
    print(data_handler.text_field.vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)

    args = parser.parse_args()

    run_train(args)
