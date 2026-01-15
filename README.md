# Life Insurance Risk Prediction with Supervised Learning

## Overview
This project implements an end-to-end supervised learning pipeline to predict applicant risk levels in the life insurance underwriting process. Using the Prudential Life Insurance dataset (59,000+ records and 128 features), multiple machine learning models are developed, tuned, and evaluated to assess their effectiveness in handling high-dimensional, real-world insurance data.

The project focuses on comparing traditional statistical models with more complex machine learning approaches, highlighting performance trade-offs, interpretability considerations, and practical implications for underwriting decision support.

---

## Dataset
- **Source:** Prudential Life Insurance Assessment Dataset (Kaggle)
- **Size:** 59,381 observations, 128 features
- **Feature types:**  
  - Demographic information  
  - Health and medical history  
  - Employment and insurance history  
- **Target variable:** Applicant risk level  
  - Converted to a **binary classification task** for approval vs. rejection

> The dataset is not included in this repository due to licensing restrictions.  
> Please download it directly from Kaggle and place the files in a local `data/` directory when running the notebooks.

---

## Methodology

### Data Preprocessing
- Removed features with excessive missing values
- Handled missing data using **iterative imputation**
- Applied **one-hot encoding** to categorical variables
- Standardized numerical features
- Explored **Principal Component Analysis (PCA)** with:
  - 40% variance retained  
  - 80% variance retained  
  - No dimensionality reduction (baseline)

### Models Implemented
- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- Neural Network (Multi-layer Perceptron)

### Model Training & Evaluation
- Hyperparameter tuning via randomized and grid search
- 5-fold cross-validation for robust model selection
- Evaluation metrics:
  - Accuracy
  - ROC-AUC
  - Mean Absolute Error (MAE)

---

## Results
- **Neural Network** achieved the strongest overall performance  
  - **ROC-AUC:** 0.897  
  - **Test Accuracy:** 82.4%
- **Decision Tree** produced competitive results with strong interpretability and feature importance insights
- PCA did not consistently improve model performance, suggesting that retaining the full feature set was beneficial for this dataset

These results demonstrate the effectiveness of non-linear models in capturing complex relationships within insurance applicant data, while also highlighting interpretability–performance trade-offs relevant to real-world underwriting.

---

## Project Structure
├── README.md
├── SYDE522-Group9-Final Code.ipynb
├── SYDE522-Final Code-Response.ipynb
├── SYDE522-Final Code-Response-NN.ipynb
├── SYDE552-Implementation of Risk Prediction in Life Insurance Industry using Supervised Learning Algorithms.pdf


- **Notebooks:** End-to-end experiments, model training, and evaluation
- **PDF report:** Detailed methodology, analysis, and academic discussion

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/faye0530/life-insurance-risk-prediction.git
2. Install required dependencies (Python 3.8+ recommended)
3. Download the Prudential dataset from Kaggle
4. Open and run the notebooks in sequence using Jupyter Notebook or Jupyter Lab

---

## Key Takeaways
- Supervised learning models can significantly enhance risk assessment in life insurance underwriting
- Neural networks are effective for modeling complex, non-linear relationships in high-dimensional insurance data
- Interpretability remains critical when deploying models in regulated financial and insurance environments

## Future Work

- Explore ensemble methods (e.g., Gradient Boosting, Stacking)
- Apply explainability techniques such as SHAP or feature attribution
- Extend from binary to multi-class risk prediction
- Simulate downstream business impacts such as premium pricing strategies
