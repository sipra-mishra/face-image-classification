# AI-Based Authentic Face Image Classification

Classifying real vs AI-generated face images using ResNet50 and Logistic Regression (97.44% accuracy).

## Business Problem
Fake and AI-generated face images are increasingly used for fraudulent profiles and misinformation. This project builds a model that classifies face images as real or fake.

## Data
- 6,557 labeled face images with metadata (provided through the LaunchED Global internship program)
- A balanced subset of 1,758 images (REAL and FAKE) was used for training and testing
- The dataset is not included in this repository

## Approach
1. Preprocessed images (RGB, resized to 224x224)
2. Extracted 2048-dimension features using a pretrained ResNet50
3. Trained a Logistic Regression classifier (80/20 stratified split)
4. Excluded metadata fields that were almost perfectly correlated with the label, to avoid data leakage
5. Built a Gradio app for live predictions

## Results
| Metric | Value |
|---|---|
| Accuracy | 97.44% |
| Precision | 98.97% |
| Recall | 96.50% |
| F1-score | 97.72% |

Accuracy stayed above 98.8% across gender, age group, and image-quality groups.

## Limitations
The model was trained only on StyleGAN3-generated fakes, so it may not detect fakes made with other methods, such as diffusion models.

## Tools
Python, PyTorch, scikit-learn, Pandas, NumPy, Gradio, Google Colab

## Files
- `Face_Classification_Project.ipynb`: full code, results, and the Gradio demo (final cells)
- `Project_Report.pdf`: full project report, including results and demo screenshots
