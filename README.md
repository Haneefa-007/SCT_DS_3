# SCT_DS_3
Task 3 — Decision Tree Classifier README
Bank Marketing Prediction Using Decision Tree Classifier
Project Overview

This project builds a Machine Learning model using a Decision Tree Classifier to predict whether a customer will purchase a product or service based on demographic and behavioral data.

The dataset used is the Bank Marketing Dataset from the UCI Machine Learning Repository.

Dataset Used

Dataset: urlBank Marketing Dataset - UCI Repositoryhttps://archive.ics.uci.edu/dataset/222/bank+marketing

File used:

bank-full.csv
Technologies Used
Python
Pandas
Matplotlib
Seaborn
Scikit-learn
VS Code / Jupyter Notebook
Libraries Required

Install the required libraries:

pip install pandas matplotlib seaborn scikit-learn
Project Objectives
Perform data preprocessing
Encode categorical variables
Train a Decision Tree model
Evaluate model performance
Visualize model results
Machine Learning Workflow
1. Data Loading

Loaded the Bank Marketing dataset using Pandas.

2. Data Preprocessing
Checked missing values
Encoded categorical columns using Label Encoding
Converted string data into numerical format
3. Feature Selection

Separated:

Features (X)
Target variable (y)

Target variable:

y

Represents whether the customer subscribed to the service.

Model Used

Machine Learning Algorithm:

entity["scientific_concept","Decision Tree","Supervised machine learning algorithm"]

Evaluation Metrics

The following evaluation techniques were used:

Accuracy Score
Classification Report
Confusion Matrix
Feature Importance Analysis
Visualizations

The project includes:

Confusion Matrix Heatmap
Feature Importance Graph
Decision Tree Visualization
Advanced Improvements Added
Feature Importance Analysis
Confusion Matrix
Tree Pruning using max_depth
Random Forest discussion
Hyperparameter Tuning concepts
ROC Curve concepts
Project Structure
Task-3/
│
├── bank-full.csv
├── main.py
└── README.md
How To Run
Open terminal in project folder
Run:
python main.py
Learning Outcomes

This project helped in understanding:

Data preprocessing
Label Encoding
Supervised Machine Learning
Decision Tree algorithms
Model evaluation techniques
Feature importance interpretation
Classification problems
Conclusion

This project demonstrates how Machine Learning can be used to predict customer behavior using demographic and behavioral data. It also highlights the importance of preprocessing, evaluation, and visualization in Data Science workflows.
