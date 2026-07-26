"""
evaluate.py
-----------
Loads the trained model and produces a full evaluation report on the
held-out test set: accuracy, precision, recall, F1, confusion matrix,
ROC curve and AUC. Figures are saved to ./outputs.

Usage:
    python evaluate.py --model_path model/psoriasis_eczema_model.keras
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

IMG_SIZE = (224, 224)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/psoriasis_eczema_model.keras")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--class_indices", type=str, default="model/class_indices.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.class_indices) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = tf.keras.models.load_model(args.model_path)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(args.data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=args.batch_size,
        class_mode="binary",
        classes=class_names,
        shuffle=False,
    )

    y_true = test_gen.classes
    y_prob = model.predict(test_gen).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:\n")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=150)
    plt.close()

    print(f"\nAUC: {roc_auc:.4f}")
    print(f"Figures and report saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
