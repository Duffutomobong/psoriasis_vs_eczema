"""
train.py
--------
GET 324 - Laboratory Exercise 10 (Mini-Project)
Binary Image Classification: Psoriasis vs Eczema

Trains a transfer-learning CNN (MobileNetV2 backbone) to distinguish
Psoriasis from Eczema skin lesion images, then saves the trained model
for deployment in the Streamlit application (app.py).

Expected data layout (already created for you under ./data):

    data/
        train/
            psoriasis/   *.jpg
            eczema/      *.jpg
        val/
            psoriasis/   *.jpg
            eczema/      *.jpg
        test/
            psoriasis/   *.jpg
            eczema/      *.jpg

Usage:
    python train.py --epochs 20 --batch_size 32 --fine_tune

Dataset sources (pick one, or combine):
    - Kaggle: "Skin Diseases Image Dataset" (ismailpromus)
      https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
    - Kaggle: "DermNet" (contains Psoriasis and Atopic Dermatitis/Eczema classes)
      https://www.kaggle.com/datasets/shubhamgoel27/dermnet
Only keep the Psoriasis and Eczema folders, then split into train/val/test
(e.g. 70/15/15) using the provided `split_dataset.py` helper.
"""

import os
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
CLASS_NAMES = ["eczema", "psoriasis"]  # alphabetical -> matches flow_from_directory default


def build_data_generators(data_dir, batch_size):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42,
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def build_model(fine_tune=False, fine_tune_at=100):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune
    if fine_tune:
        # Freeze the earlier layers, only fine-tune the later ones
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune", action="store_true",
                         help="Unfreeze top layers of MobileNetV2 for a second fine-tuning phase")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train_gen, val_gen, test_gen = build_data_generators(args.data_dir, args.batch_size)
    print("Class indices:", train_gen.class_indices)

    model, base_model = build_model(fine_tune=False)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print("\n=== Phase 1: Training classifier head (base frozen) ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    plot_history(history, os.path.join(args.output_dir, "phase1_history.png"))

    if args.fine_tune:
        print("\n=== Phase 2: Fine-tuning top layers of MobileNetV2 ===")
        base_model.trainable = True
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=args.lr / 10),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )
        plot_history(history_fine, os.path.join(args.output_dir, "phase2_finetune_history.png"))

    # Final save (Keras native format, loadable directly in app.py)
    final_path = os.path.join(args.model_dir, "psoriasis_eczema_model.keras")
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")

    # Save class index mapping for the Streamlit app
    with open(os.path.join(args.model_dir, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # Quick test-set evaluation
    print("\n=== Test set evaluation ===")
    test_loss, test_acc, test_auc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f} | Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
