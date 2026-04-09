# Cloud Failure Prediction using Machine Learning and Deep Learning

## 1. Introduction

This repository presents the implementation of the research paper:

"Cloud Failure Prediction based on Traditional Machine Learning and Deep Learning" (2022)

Cloud computing environments execute large-scale distributed workloads where failures at the job or task level can significantly impact system reliability, performance, and cost.Predicting such failures in advance enables efficient scheduling, better resource allocation, and improved fault tolerance.

This project reproduces the methodology proposed in the paper using both traditional machine learning and deep learning techniques, with practical adaptations for computational feasibility.

---

## 2. Problem Statement

The objective of this project is to build predictive models that can classify whether a cloud job or task will fail or succeed based on historical execution data.

Two prediction levels are considered:

* Job-level prediction: Determines whether an entire job fails
* Task-level prediction: Determines whether individual tasks fail

---

## 3. Dataset

The implementation is based on the Google Cluster Trace Dataset (2011).

* The original dataset size is approximately 400 GB
* Due to computational limitations, a sampled subset of the dataset is used
* The sampled data preserves key workload characteristics required for modeling

---

## 4. Methodology

### 4.1 Data Preprocessing

* Filtering terminal events (FAIL, FINISH, KILL, etc.)
* Aggregating task-level features for job-level prediction
* Handling missing values
* Feature scaling using StandardScaler

### 4.2 Handling Class Imbalance

* SMOTE (Synthetic Minority Over-sampling Technique) is applied to training data

### 4.3 Model Implementation

Traditional Machine Learning Models:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

Deep Learning Models:

* Single-layer LSTM
* Bi-layer LSTM
* Tri-layer LSTM

### 4.4 Training Strategy

* Train-test split (70:30)
* Validation split for LSTM models
* Early stopping to prevent overfitting
* Experiments conducted on Google Colab using Tesla T4 GPU

---

## 5. Evaluation Metrics

The models are evaluated using:

* Accuracy
* Error Rate
* Precision
* Recall (Sensitivity)
* Specificity
* F1-Score
* Confusion Matrix

---

## 6. Experimental Results

### 6.1 Job-Level Prediction

* Best Model: XGBoost
* Accuracy: 93.89%
* F1-Score: 0.9280

Observations:

* Tree-based models achieved the highest performance (~93–94%)
* Logistic Regression showed lower performance (~61.71%)
* Deep learning models showed moderate performance:

  * Single-layer LSTM: 76.94%
  * Bi-layer LSTM: 84.23%
  * Tri-layer LSTM: 85.08%

---

### 6.2 Task-Level Prediction

* Best Models: Decision Tree / Random Forest
* Accuracy: 95.21%
* F1-Score: 0.9708

Observations:

* Very high performance due to strong predictive features
* Deep learning models achieved competitive performance (~94–95%)
* Logistic Regression also performed well (~92.65%)

---

### 6.3 Feature Importance Insights

Job-level important features:

* disk_space_request
* cpu_request

Task-level dominant feature:

* priority

---

### 6.4 Scalability Analysis

* Logistic Regression is fastest
* XGBoost provides best accuracy-speed balance
* Random Forest is computationally heavier

---

## 7. Key Features of Implementation

* Modular and well-structured code
* Unified pipeline for ML and DL models
* Automatic GPU/CPU detection
* Feature importance analysis
* Scalability evaluation

---

## 8. Project Structure (Explanation)

The repository is organized to ensure clarity and ease of use:

* README.md – Documentation of the project
* requirements.txt – Required Python libraries
* .gitignore – Excludes unnecessary files
* cloud_failure_prediction.py – Complete implementation

---

## 9. Implementation Notes

* The implementation closely follows the methodology described in the research paper
* Dataset sampling is used due to hardware constraints
* The system is optimized to run on Google Colab

---

## 10. Reproducibility and Design Choices

* Fixed random seeds ensure consistent results
* Sampling strategy balances feasibility and performance
* Modular design enables easy experimentation

---

## 11. Comparison with Original Paper

* Same methodology as the research paper
* Uses a subset of dataset instead of full data
* Performance trends remain consistent with the paper

---

## 12. Limitations

* Full dataset (400 GB) could not be used
* Performance may vary with larger datasets
* Deep learning models require higher computational resources

---

## 13. Conclusion

This project demonstrates that traditional machine learning models are highly effective for failure prediction in cloud systems, particularly for structured data.XGBoost achieved the best performance for job-level prediction while Decision Tree and Random Forest performed best for task-level prediction.Deep learning models provide competitive results but involve higher computational overhead.

---

## 14. Author

Devikrishna Reji Kumar Bindu
