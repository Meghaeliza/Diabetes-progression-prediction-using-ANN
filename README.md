# Diabetes Progression Prediction Using Artificial Neural Networks (ANN)

## 1. Project Overview

This project focuses on predicting diabetes disease progression using an Artificial Neural Network (ANN) regression model.

The project uses the Diabetes dataset available in the Scikit-learn library. The dataset contains clinical measurements collected from patients and a quantitative measure of disease progression one year after baseline.

The primary objective is to develop an ANN model that learns relationships between patient-related features and diabetes progression.

Different neural network architectures and regularization techniques are experimented with to improve predictive performance.

**Project Type:** Deep Learning – Artificial Neural Network (ANN)  
**Problem Type:** Regression  
**Framework:** TensorFlow / Keras  
**Dataset:** Scikit-learn Diabetes Dataset

---

## 2. Project Objectives

The main objectives of this project are:

- Load and explore the Diabetes dataset from Scikit-learn.
- Perform data preprocessing and feature normalization.
- Conduct Exploratory Data Analysis (EDA) to understand feature distributions and relationships.
- Design and implement an Artificial Neural Network for regression.
- Train and evaluate the ANN model using appropriate performance metrics.
- Experiment with different ANN architectures and hyperparameters.
- Apply Dropout and L2 regularization to investigate their effect on model performance.
- Compare different models and select the best-performing architecture.

---

## 3. Dataset Description

The dataset is obtained from the `sklearn.datasets` module.

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
```

### Dataset Information

| Property | Description |
|---|---|
| Total observations | 442 |
| Independent variables | 10 |
| Target variables | 1 |
| Total columns | 11 |
| Missing values | 0 |
| Duplicate records | 0 |
| Problem type | Regression |

### Independent Features

The dataset contains the following ten independent variables:

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex |
| bmi | Body Mass Index |
| bp | Average blood pressure |
| s1 | Total serum cholesterol |
| s2 | Low-density lipoproteins |
| s3 | High-density lipoproteins |
| s4 | Total cholesterol / HDL ratio |
| s5 | Log of serum triglycerides |
| s6 | Blood sugar level |

### Target Variable

The target variable represents a quantitative measure of diabetes disease progression one year after baseline.

This is a regression problem because the objective is to predict a continuous numerical value rather than classify patients into categories.

---

## 4. Technologies and Libraries Used

The following Python libraries were used:

| Library | Purpose |
|---|---|
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Dataset loading, preprocessing, splitting and evaluation |
| TensorFlow | Deep learning framework |
| Keras | ANN model development and training |

---

## 5. Data Preprocessing

The following preprocessing steps were performed:

### 5.1 Dataset Loading

The Diabetes dataset was loaded using Scikit-learn and converted into a Pandas DataFrame.

### 5.2 Missing Value Analysis

The dataset was examined for missing values.

Result: No missing values were found.

### 5.3 Duplicate Value Analysis

Duplicate records were checked.

Result: No duplicate records were identified.

### 5.4 Feature Normalization

The original Scikit-learn Diabetes dataset already contains standardized independent features.

StandardScaler was additionally applied after splitting the dataset to ensure that the ANN receives appropriately scaled features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data to prevent data leakage.

### 5.5 Train-Test Split

The dataset was divided into training and testing subsets.

- Training data: 80%
- Testing data: 20%
- Random state: 42

---

## 6. Exploratory Data Analysis (EDA)

EDA was performed to understand the characteristics of the dataset.

The following visualizations were generated:

1. Distribution of the target variable.
2. Distribution of independent features.
3. Correlation heatmap.
4. Scatter plots showing relationships between individual features and the target variable.

These visualizations helped identify feature distributions, correlations and possible relationships with diabetes progression.

---

## 7. ANN Model Development

An Artificial Neural Network was developed using TensorFlow and Keras.

Since the target variable is continuous, a regression-based ANN architecture was selected.

### Model 1: Baseline ANN

The initial model consisted of:

| Layer | Configuration |
|---|---|
| Input Layer | 10 features |
| Hidden Layer 1 | 64 neurons, ReLU |
| Hidden Layer 2 | 32 neurons, ReLU |
| Output Layer | 1 neuron, Linear |

**Hyperparameters:**

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Epochs: Maximum 200
- Batch size: 16
- Early stopping: Patience 20

### Model 2: Simplified ANN

A smaller architecture was introduced to investigate whether reducing network complexity would improve generalization.

| Layer | Configuration |
|---|---|
| Input Layer | 10 features |
| Hidden Layer 1 | 32 neurons, ReLU |
| Hidden Layer 2 | 16 neurons, ReLU |
| Output Layer | 1 neuron, Linear |

**Hyperparameters:**

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: MSE
- Batch size: 16
- Maximum epochs: 300
- Early stopping: Patience 20

### Model 3: Regularized ANN

The third model introduced Dropout and L2 regularization to investigate their influence on model performance.

| Layer | Configuration |
|---|---|
| Input Layer | 10 features |
| Hidden Layer 1 | 32 neurons, ReLU |
| Dropout | 0.2 |
| Hidden Layer 2 | 16 neurons, ReLU |
| Output Layer | 1 neuron, Linear |

**Hyperparameters:**

- Optimizer: Adam
- Learning rate: 0.0005
- Loss function: MSE
- Dropout rate: 0.2
- L2 regularization: 0.001
- Batch size: 16
- Maximum epochs: 300
- Early stopping: Patience 20

---

## 8. Model Training

All three ANN architectures were trained using the training dataset.

The validation split was set to 20% of the training data.

Early stopping was implemented to monitor validation loss and restore the best model weights when training stopped.

The Adam optimizer was used to update network weights, while Mean Squared Error was used as the loss function.

Training and validation loss curves were plotted to observe model learning behaviour.

---

## 9. Model Evaluation

The models were evaluated using the following regression metrics:

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

Lower values indicate better performance.

### Root Mean Squared Error (RMSE)

Represents the square root of MSE and expresses prediction error in the original target units.

Lower values indicate better performance.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

Lower values indicate better performance.

### R-squared Score (R²)

Measures the proportion of variance in the target variable explained by the model.

Higher values generally indicate better predictive performance.

---

## 10. Model Comparison and Results

Three ANN architectures were evaluated using the same testing dataset.

### Performance Comparison

| Model | MSE | RMSE | MAE | R² Score |
|---|---:|---:|---:|---:|
| Baseline ANN | 3088.3247 | 55.5727 | 44.1419 | 0.4171 |
| Simplified ANN | 2972.3454 | 54.5192 | 43.8040 | 0.4390 |
| Regularized ANN | **2921.1430** | **54.0476** | **43.2232** | **0.4486** |

### Best Performing Model

The Regularized ANN achieved the best performance among the three evaluated models.

Its performance was:

- MSE: 2921.1430
- RMSE: 54.0476
- MAE: 43.2232
- R² Score: 0.4486

The final model showed an improvement in all four evaluation metrics compared with the baseline ANN.

### Improvement Analysis

Compared with the baseline ANN:

- MSE decreased by approximately 5.41%.
- RMSE decreased by approximately 2.74%.
- MAE decreased by approximately 2.08%.
- R² improved from 0.4171 to 0.4486.

The experiments demonstrate that changing network architecture and introducing regularization can influence ANN performance.

---

## 11. Model Improvement Techniques

The following experiments were performed:

1. **Architecture Modification:** Reduced the number of neurons from 64 and 32 to 32 and 16.
2. **Dropout Regularization:** Introduced a dropout rate of 0.2 to randomly deactivate neurons during training.
3. **L2 Regularization:** Applied an L2 penalty of 0.001 to discourage excessively large weights.
4. **Learning Rate Adjustment:** Reduced the Adam learning rate from 0.001 to 0.0005 for the regularized model.
5. **Early Stopping:** Used validation loss monitoring to prevent unnecessary training.

Among the three experiments, the regularized architecture achieved the lowest test errors and highest R² score.

---

## 12. Conclusion

In this project, an Artificial Neural Network regression model was developed to predict diabetes disease progression using the Scikit-learn Diabetes dataset.

The project involved data loading, preprocessing, exploratory data analysis, ANN architecture development, model training and evaluation.

Three different ANN architectures were experimented with to investigate improvements in predictive performance.

The baseline ANN achieved an R² score of 0.4171, while the simplified ANN achieved 0.4390. The regularized ANN produced the best result with an R² score of 0.4486.

The final model achieved an MSE of 2921.1430, RMSE of 54.0476 and MAE of 43.2232.

Although the regularized ANN performed better than the other tested architectures, its predictive performance indicates that further experimentation and validation may be beneficial.

This project provided practical experience in applying Artificial Neural Networks to a real-world regression problem and understanding the importance of architecture selection, hyperparameter tuning and model evaluation.

---

## 13. Future Improvements

Potential future improvements include:

- Experimenting with additional ANN architectures.
- Performing systematic hyperparameter optimization.
- Applying K-fold cross-validation for more reliable performance estimates.
- Comparing ANN performance with traditional regression models such as Linear Regression, Random Forest and Gradient Boosting.
- Investigating additional feature engineering techniques.
- Evaluating model performance on independent datasets.

---

## 14. Repository Structure

```text
Diabetes-Progression-ANN/
│
├── Diabetes_Progression_ANN.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 15. How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project directory

```bash
cd Diabetes-Progression-ANN
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Open Jupyter Notebook

```bash
jupyter notebook
```

Open `Diabetes_Progression_ANN.ipynb` and execute the notebook cells sequentially.

---

## 16. Requirements

The project uses the following Python libraries:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
jupyter
```

---

## 17. Disclaimer

This project is developed for educational and academic purposes.

The model predicts a quantitative diabetes disease progression measure from the provided dataset. It is not intended for medical diagnosis, clinical decision-making or treatment recommendations.

---

**Project:** Diabetes Progression Prediction Using Artificial Neural Networks

**Domain:** Deep Learning | Regression | Healthcare Analytics
