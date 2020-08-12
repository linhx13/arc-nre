import os
import argparse

import tensorflow as tf
import arcnre.tf

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run_train(args):
    data_handler = arcnre.tf.data.SentenceREDataHandler(
        text_field=arcnre.tf.data.TextField(max_len=args.max_len),
        pos_field=arcnre.tf.data.PositionField(max_len=args.max_len),
        label_field=arcnre.tf.data.LabelField())

    train_path = os.path.expanduser(args.train_path)
    val_path = os.path.expanduser(args.val_path)

    train_examples = data_handler.read_examples(train_path)
    val_examples = data_handler.read_examples(val_path)
    data_handler.build_vocab(train_examples, val_examples)
    print(data_handler.text_field.vocab)

    train_dataset = data_handler.build_dataset(train_path)
    val_dataset = data_handler.build_dataset(val_path)

    text_embedder = tf.keras.layers.Embedding(
        len(data_handler.text_field.vocab), 200, mask_zero=True)

    model = arcnre.tf.models.CNNModel(data_handler.features,
                                      data_handler.targets,
                                      text_embedder)

    trainer = arcnre.tf.training.Trainer(model, data_handler)
    trainer.train(train_dataset=train_dataset,
                  val_dataset=val_dataset,
                  optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'],
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  model_dir=args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_dir")

    args = parser.parse_args()

    run_train(args)
