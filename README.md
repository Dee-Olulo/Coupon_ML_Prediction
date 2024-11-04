# Coupon Recommendation System Using User Behavior Data

## Objective
The objective of this project is to develop a machine learning model that predicts the likelihood of a user accepting a coupon based on various factors such as weather conditions, passenger information, and time of day. This model aims to assist businesses in better targeting their coupons and optimizing the timing and conditions under which they are offered.

## Data Overview

### Dataset Description
The dataset used for this project is the **In-Vehicle Coupon Recommendation dataset**, obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation) under a CC BY 4.0 license. It contains **12,684 rows** and approximately **20 columns** representing instances where users were offered coupons while driving. The target variable (Y) indicates whether the user accepted the coupon (1) or declined it (0).

- **Source**: UCI Machine Learning Repository
- **Total Rows**: 12,684
- **Total Columns**: Approximately 20 (varies with preprocessing)

### Key Variables
- **destination**: User's intended destination (e.g., "No Urgent Place", "Work").
- **passenger**: Indicates who the user is traveling with (e.g., "Alone", "Friend(s)").
- **weather**: Weather conditions when the coupon was offered (e.g., Sunny, Rainy).
- **temperature**: Outside temperature in Fahrenheit at the time of the offer.
- **time**: Time of day when the coupon was presented (e.g., 10AM, 2PM).
- **coupon**: Type of coupon offered (e.g., Coffee House, Restaurant(<20)).
- **expiration**: Expiration time for the coupon (e.g., 2 hours, 1 day).
- **has_children**: Binary variable indicating if the user has children.
- **direction_same**: Binary variable indicating if the user is heading in the same direction as the coupon destination.
- **Y**: Target variable (1 for accepting the coupon, 0 for rejecting it).

## Tool Setup
To conduct a comprehensive analysis, the following tools were utilized:
- **Data Handling**: `numpy`, `pandas` for data manipulation and exploration.
- **Visualization**: `matplotlib`, `seaborn` for data visualization.
- **Machine Learning and Evaluation**: Various features from `sklearn` for data processing, model training, and performance evaluation.

## Exploratory Data Analysis (EDA)
Key observations from the EDA include:
- Passengers traveling to "No Urgent Place" show higher coupon acceptance.
- Lower acceptance rates among younger passengers (<21) compared to those aged 21-31.
- Increased coupon acceptance correlates with higher temperatures.
- The "Restaurant(<20)" coupon type shows the highest acceptance rates.

## Data Preprocessing
- Defined target variable `Y` and input features `X`.
- Split data into training and test sets.
- Applied preprocessing steps to avoid data leakage.
- Converted all columns to numeric format for model compatibility.
- Scaled the train and test sets for uniform feature contribution.

## Metrics of Evaluation
The primary metric for evaluation is **Recall**, aiming to maximize coupon usage by identifying true positives effectively.

## Modeling
### Models Used
- **Logistic Regression**: Baseline model for evaluation.
- **Decision Trees**: Analyzed for overfitting.
- **RandomForestClassifier**: Investigated but showed significant overfitting.

### Classification Reports
- **Logistic Regression**:
  - Recall for Class 0 (Non-Users): 0.58 (Training: 0.58)
  - Recall for Class 1 (Users): 0.78 (Training: 0.78)

- **Decision Trees**: 
  - Recall for Class 0: 0.63
  - Recall for Class 1: 0.73

- **Random Forest Classifier**: 
  - Training metrics: Precision, Recall, F1-Score = 1.00
  - Test metrics showed drop to 0.73, indicating overfitting.

### Hyperparameter Tuning
- Models were fine-tuned using GridSearchCV to optimize performance and reduce overfitting.

## Model Selection
Logistic Regression was selected as the best performing model due to its balanced performance across training and test sets, demonstrating good generalization with minimal differences in metrics.

## Confusion Matrix Observations
- **True Negatives (TN)**: 645
- **False Positives (FP)**: 483
- **False Negatives (FN)**: 315
- **True Positives (TP)**: 1,094

## Conclusion
The project successfully developed a predictive model for coupon acceptance, allowing the business to enhance its marketing strategies through better-targeted campaigns and optimized coupon offerings.

### Getting Started
To run this project locally:
1. Clone this repository.
2. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Load the dataset and execute the scripts.
!
