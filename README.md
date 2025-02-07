# Spam Detection with Machine Learning

## Project Overview

This project aims to classify messages as spam or not spam using machine learning models. It includes preprocessing, feature extraction, model training, and evaluation steps.

## Installation & Setup

### Prerequisites

- Python 3.8+
- Create a virtual environment and install dependencies using the provided `environment.yaml` file:
  ```bash
  conda env create -f environment.yaml
  conda activate spam
  ```

## Data Preprocessing & Feature Extraction

This project uses two different feature extraction techniques for text data:

- **TF-IDF Features:** Extracts features using `TfidfVectorizer`. This method represents text based on the importance of words in a document relative to the entire dataset. Some additional visualizations for TF-IDF-based feature extraction are included in the `additional_images/` directory.
- **Sentence Embeddings:** Uses `SentenceTransformer`, which captures more contextual information and maintains semantic meaning better than TF-IDF.

Both techniques are used for training different machine learning models, allowing comparison between their performance.

## Model Training & Evaluation

- **Implemented Models:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - K-Nearest Neighbors
  - Support Vector Machines
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.

### Model Performance Comparison

#### **TF-IDF + Machine Learning Models Results**

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| SVM                 | 98.24%   | 98.58%    | 96.34% | 97.45%   |
| Random Forest       | 98.10%   | 97.61%    | 96.93% | 97.27%   |
| Logistic Regression | 97.52%   | 99.27%    | 93.57% | 96.33%   |
| Decision Tree       | 96.00%   | 93.83%    | 94.76% | 94.29%   |
| Naive Bayes         | 95.69%   | 96.44%    | 91.00% | 93.64%   |
| KNN                 | 79.56%   | 100.00%   | 41.35% | 58.50%   |

#### **Transformers + Machine Learning Models Results**

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| SVM                 | 98.43%   | 98.81%    | 96.60% | 97.69%   |
| Logistic Regression | 97.73%   | 97.15%    | 96.21% | 96.68%   |
| Random Forest       | 96.77%   | 98.54%    | 91.93% | 95.12%   |
| KNN                 | 94.73%   | 87.97%    | 98.06% | 92.74%   |
| Decision Tree       | 89.23%   | 83.94%    | 84.84% | 84.39%   |

### Analysis of Results

- **TF-IDF** provides strong performance across traditional machine learning models, showing high precision and recall.
- **Transformers** offer a slight improvement, especially in recall and F1-score, indicating their ability to retain more contextual information.
- Overall, both approaches perform well, with transformers slightly outperforming TF-IDF in most cases.

## Usage

To run the Flask API and test the model using a test script:

```bash
python flask_api.py &  # Start the Flask API
python test_with_flask_api.py  # Run test requests
```

## Assumptions

- The dataset contains a `Label` column indicating spam or not.
- Data preprocessing  normalizes text.
- Models are saved as `.pkl` files for later use.
- The development started on colabratory.
- spam_classification_development_with_colabratory.ipyn file is the file on the colabratory.
- Before testing, unzip the models