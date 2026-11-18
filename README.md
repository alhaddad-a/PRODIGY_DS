# PRODIGY_DS

My work for the **Prodigy InfoTech Data Science Internship**. Each task is in its own folder with a Google Colab notebook, a README explaining the problem, method and findings, and the charts.

| Task | Topic | Skills | Open |
|---|---|---|---|
| [Task 01](Task_01_Population_Distribution) | World population distribution (World Bank, 1960-2024) | Data cleaning, histograms, log scales, bar charts | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alhaddad-a/PRODIGY_DS/blob/main/Task_01_Population_Distribution/Task1_Population_Distribution.ipynb) |
| [Task 02](Task_02_Titanic_EDA) | Titanic: data cleaning and exploratory data analysis | Missing-value imputation, feature engineering, hypothesis testing, confounders | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alhaddad-a/PRODIGY_DS/blob/main/Task_02_Titanic_EDA/Task2_Titanic_EDA.ipynb) |
| [Task 03](Task_03_Bank_Marketing_Decision_Tree) | Bank marketing: predicting term-deposit subscriptions | Decision tree, grid-search tuning, overfitting, class imbalance, leakage check, lift/gains | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alhaddad-a/PRODIGY_DS/blob/main/Task_03_Bank_Marketing_Decision_Tree/Task3_Bank_Marketing_Decision_Tree.ipynb) |
| [Task 04](Task_04_Twitter_Sentiment) | Twitter sentiment toward 32 brands and games | Text cleaning, net sentiment, log-odds word analysis, TF-IDF + logistic regression, group-aware splitting | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alhaddad-a/PRODIGY_DS/blob/main/Task_04_Twitter_Sentiment/Task4_Twitter_Sentiment.ipynb) |
| [Task 05](Task_05_US_Accidents) | US traffic accidents: hotspots and contributing factors (7.7M rows) | Large-data handling, geospatial hotspots, time and weather analysis, gradient boosting, permutation importance | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alhaddad-a/PRODIGY_DS/blob/main/Task_05_US_Accidents/US_Accidents_Analysis.ipynb) |

## Highlights

- **Task 05** analyzes 7.7 million US accidents. Highway accidents are 2.4x more likely to be severe than local-street ones, rain (not snow or fog) has the highest severe share, half of all accidents are in 5 states, and a gradient-boosting model predicts severe accidents with ROC-AUC 0.866. It includes street-level hotspots and an interactive heat map.
- **Task 04** finds that sports games (MaddenNFL, NBA2K, FIFA) get the most negative opinion, driven by bug and server complaints. It also shows how near-duplicate tweets would inflate a classifier's accuracy from 54% to 90% if the data were split carelessly.
- **Task 03** builds an explainable decision tree (ROC-AUC 0.804) that reaches 65% of subscribers by calling only the top 20% of ranked customers. It deliberately excludes call duration, which leaks the outcome.
- **Task 02** frames Titanic survival as a data science problem: five hypotheses, careful imputation (age by title and class), and a confounder check showing that port of embarkation mattered mainly through passenger class.
- **Task 01** shows why a log scale is needed for skewed data, and how the world's population shifted between regions from 1960 to 2024.

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, Folium, Google Colab
