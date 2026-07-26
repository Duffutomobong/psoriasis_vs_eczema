"""
split_dataset.py
-----------------
Helper utility to split a raw folder of images (organised as
raw_data/psoriasis/*.jpg and raw_data/eczema/*.jpg) into the
train/val/test structure expected by train.py, using a 70/15/15 split.

Usage:
    python split_dataset.py --raw_dir raw_data --out_dir data --train 0.7 --val 0.15 --test 0.15
"""

import os
import shutil
import random
import argparse


def split_class(class_name, raw_dir, out_dir, train_ratio, val_ratio, seed=42):
    src_folder = os.path.join(raw_dir, class_name)
    images = [f for f in os.listdir(src_folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.Random(seed).shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, files in splits.items():
        dst_folder = os.path.join(out_dir, split_name, class_name)
        os.makedirs(dst_folder, exist_ok=True)
        for fname in files:
            shutil.copy2(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))
        print(f"{class_name}/{split_name}: {len(files)} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Folder containing raw_dir/psoriasis and raw_dir/eczema")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "Ratios must sum to 1.0"

    for class_name in ["psoriasis", "eczema"]:
        split_class(class_name, args.raw_dir, args.out_dir, args.train, args.val)

    print("\nDone. Data split into:", args.out_dir)


if __name__ == "__main__":
    main()
