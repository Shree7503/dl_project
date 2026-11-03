"""Train a Bidirectional LSTM to classify tweet sentiment (Sentiment140 dataset).

Usage:  python train.py [--samples 200000] [--epochs 4]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import clean_text

DATA_DIR = "data"
MODEL_DIR = "models"
TRAIN_CSV = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
TEST_CSV = os.path.join(DATA_DIR, "testdata.manual.2009.06.14.csv")
COLUMNS = ["target", "id", "date", "flag", "user", "text"]

VOCAB_SIZE = 20000
MAX_LEN = 50
EMBED_DIM = 128
SEED = 42


def load_csv(path):
    df = pd.read_csv(path, encoding="latin-1", header=None, names=COLUMNS)
    df = df[df["target"] != 2]  # keep only negative (0) and positive (4)
    df["label"] = (df["target"] == 4).astype("int32")
    df["clean"] = df["text"].astype(str).map(clean_text)
    return df[["clean", "label"]]


def build_model(train_texts):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN, standardize=None
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(1024))

    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[metric], marker="o", label="train")
        ax.plot(history.history[f"val_{metric}"], marker="o", label="validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200000, help="tweets to sample (balanced)")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_csv(TRAIN_CSV)
    per_class = args.samples // 2
    df = pd.concat(
        [df[df["label"] == lbl].sample(per_class, random_state=SEED) for lbl in (0, 1)],
        ignore_index=True,
    )
    df = df[df["clean"].str.len() > 0]

    texts = df["clean"].to_numpy(dtype=object)
    labels = df["label"].to_numpy(dtype="int32")
    x_train, x_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {len(x_train)}  Val: {len(x_val)}  Test: {len(x_test)}")

    model = build_model(x_train)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=256,
        callbacks=[early_stop],
    )
    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))

    print("\nEvaluating on held-out test split...")
    probs = model.predict(x_test, batch_size=1024).ravel()
    preds = (probs >= 0.5).astype(int)
    test_acc = float((preds == y_test).mean())
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Hand-labelled Sentiment140 test set (independent of training data)
    manual = load_csv(TEST_CSV)
    manual_preds = (model.predict(manual["clean"].to_numpy(dtype=object)).ravel() >= 0.5).astype(int)
    manual_acc = float((manual_preds == manual["label"].to_numpy()).mean())
    print(f"Accuracy on hand-labelled test set ({len(manual)} tweets): {manual_acc:.4f}")

    # Save an uncompiled copy: dropping the optimizer state cuts the file from ~32 MB to ~11 MB
    tf.keras.Model(model.inputs, model.outputs).save(os.path.join(MODEL_DIR, "sentiment_model.keras"))
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "manual_test_accuracy": round(manual_acc, 4),
        "train_samples": int(len(x_train)),
        "epochs_trained": len(history.history["loss"]),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model and metrics to", MODEL_DIR, metrics)


if __name__ == "__main__":
    main()
