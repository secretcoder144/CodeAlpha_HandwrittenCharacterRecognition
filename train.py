"""
CodeAlpha ML Internship - Task 3: Handwritten Character Recognition

Usage:
    pip install tensorflow tensorflow-datasets scikit-learn matplotlib pillow numpy

    python train.py --dataset mnist        # digits 0-9
    python train.py --dataset emnist       # 47 classes: digits + letters (balanced split)
    python train.py --dataset emnist --predict my_char.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# Label order of EMNIST "balanced" (47 classes)
EMNIST_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
MNIST_LABELS = list("0123456789")


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test, MNIST_LABELS


def load_emnist():
    import tensorflow_datasets as tfds

    def get(split):
        x, y = tfds.as_numpy(
            tfds.load("emnist/balanced", split=split, batch_size=-1, as_supervised=True)
        )
        x = x[..., 0]
        # tfds EMNIST images are stored rotated/transposed - fix to upright
        x = np.transpose(x, (0, 2, 1))
        return x, y

    x_train, y_train = get("train")
    x_test, y_test = get("test")
    return x_train, y_train, x_test, y_test, EMNIST_LABELS


def preprocess(x):
    return (x.astype("float32") / 255.0)[..., None]  # (N, 28, 28, 1)


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}_training_curves.png", dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, labels, name):
    cm = confusion_matrix(y_true, y_pred)
    size = 6 if len(labels) <= 10 else 14
    plt.figure(figsize=(size, size))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(labels)), labels, fontsize=6 if len(labels) > 10 else 10)
    plt.yticks(range(len(labels)), labels, fontsize=6 if len(labels) > 10 else 10)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png", dpi=150)
    plt.close()


def plot_samples(x, y_true, y_pred, labels, name, n=16):
    idx = np.random.choice(len(x), n, replace=False)
    plt.figure(figsize=(8, 8))
    for i, j in enumerate(idx):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x[j, :, :, 0], cmap="gray")
        ok = y_true[j] == y_pred[j]
        plt.title(f"T:{labels[y_true[j]]} P:{labels[y_pred[j]]}",
                  color="green" if ok else "red", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{name}_sample_predictions.png", dpi=150)
    plt.close()


def predict_image(model, path, labels):
    """Predict a single character from your own image (dark ink on light paper is fine)."""
    from PIL import Image, ImageOps
    img = Image.open(path).convert("L")
    arr = np.array(img)
    if arr.mean() > 127:            # white background -> invert to match training data
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    x = preprocess(np.array(img))[None, ...]
    probs = model.predict(x, verbose=0)[0]
    top = probs.argsort()[::-1][:3]
    print("Top predictions:")
    for k in top:
        print(f"  {labels[k]}: {probs[k]*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--predict", type=str, default=None, help="path to your own image")
    args = parser.parse_args()

    name = args.dataset
    model_path = f"{name}_cnn.keras"

    # Predict-only mode (uses a model you already trained)
    if args.predict:
        labels = EMNIST_LABELS if name == "emnist" else MNIST_LABELS
        model = tf.keras.models.load_model(model_path)
        predict_image(model, args.predict, labels)
        return

    x_train, y_train, x_test, y_test, labels = (
        load_mnist() if name == "mnist" else load_emnist()
    )
    x_train, x_test = preprocess(x_train), preprocess(x_test)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}, Classes: {len(labels)}")

    model = build_model(len(labels))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc*100:.2f}%")

    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=labels))

    plot_history(history, name)
    plot_confusion(y_test, y_pred, labels, name)
    plot_samples(x_test, y_test, y_pred, labels, name)

    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
