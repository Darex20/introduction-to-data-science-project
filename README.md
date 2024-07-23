# Basketball Game Prediction and Improved Solutions

This repository contains two Jupyter notebooks focused on predicting basketball game outcomes using machine learning techniques. The first notebook implements and evaluates various machine learning models for predicting game outcomes. The second notebook improves the prediction accuracy by incorporating detailed game statistics and applying polynomial regression.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Notebook 1: Basketball Game Prediction](#notebook-1-basketball-game-prediction)
- [Notebook 2: Improved Prediction Solution](#notebook-2-improved-prediction-solution)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

The project aims to predict the outcomes of basketball games using historical data. Various machine learning models are applied and evaluated for their prediction accuracy. The first notebook lays the foundation with initial predictions, while the second notebook improves upon these results by using additional features and more sophisticated techniques.

## Dataset

The dataset includes:
- `regular_season_results.csv`: Regular season game results.
- `tourney_results.csv`: Tournament game results.
- `teams.csv`: Information about the teams.
- `seasons.csv`: Season details.
- `tourney_seeds.csv`: Tournament seedings.
- `tourney_slots.csv`: Tournament slots.

## Notebook 1: Basketball Game Prediction

This notebook implements various machine learning models to predict basketball game outcomes. The models used include:
- Support Vector Machines (SVM)
- Random Forest Classifier
- Gradient Boosting Regressor
- K-Nearest Neighbors (KNN)
- AdaBoost Classifier
- Voting Classifier

### Key Steps:
1. Load and preprocess the data.
2. Train multiple models on the training data.
3. Evaluate model performance using accuracy metrics.
4. Predict game outcomes for specific matchups.

## Notebook 2: Improved Prediction Solution

This notebook enhances the prediction accuracy by incorporating detailed game statistics and applying polynomial regression models. It uses additional features such as point difference, steals difference, blocks difference, and field goal percentage.

### Key Steps:
1. Load detailed game statistics.
2. Calculate additional features for training.
3. Apply polynomial regression to model the relationships.
4. Train the improved models and evaluate their performance.

## Results

The notebooks produce various performance metrics such as accuracy and detailed predictions for specific games. The improved notebook demonstrates higher accuracy by using additional features and advanced regression techniques.

## Conclusion

The project successfully demonstrates the application of machine learning models to predict basketball game outcomes. The improved notebook shows that incorporating detailed statistics and advanced modeling techniques can significantly enhance prediction accuracy.

## References

1. Kaggle Dataset: [March Machine Learning Mania 2014](https://www.kaggle.com/c/march-machine-learning-mania-2014/data)
2. Scikit-learn Documentation: [scikit-learn](https://scikit-learn.org/stable/)
3. XGBoost Documentation: [xgboost](https://xgboost.readthedocs.io/en/latest/)
4. Seaborn Documentation: [seaborn](https://seaborn.pydata.org/)

---

Feel free to contribute to the project or raise issues if you encounter any problems. Happy coding!
