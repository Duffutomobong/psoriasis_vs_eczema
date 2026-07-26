# Project Report — Psoriasis vs Eczema Classifier

Our dataset was sourced from Kaggle's publicly available dermatology
collections (e.g., "Skin Diseases Image Dataset" and DermNet), filtered to
retain only Psoriasis and Eczema images, then split 70/15/15 into
train/validation/test sets. We built a binary classifier by fine-tuning a
MobileNetV2 backbone pretrained on ImageNet, adding a custom dense head
with dropout and batch normalization to reduce overfitting on our modest
dataset size. The trained model was wrapped in a Streamlit application: a
user uploads a lesion photo, the app preprocesses it and returns the
predicted class with a confidence score. Key challenges included class
imbalance and visual similarity between the two conditions, addressed
through aggressive data augmentation, class-weighted loss, and a two-phase
training strategy (frozen backbone, then partial fine-tuning). Future
improvements include collecting more diverse, dermatologist-verified images
and testing ensemble architectures such as EfficientNet.
