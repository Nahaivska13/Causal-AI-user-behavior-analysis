# Causal-AI-user-behavior-analysis

---

# Causal Analysis of User Behavior Data

## Project Overview

This project demonstrates an end-to-end **causal data analysis pipeline** applied to user behavior data from a video platform.
The main goal is to **handle missing data**, **discover causal structure**, and **estimate causal effects**, going beyond traditional predictive modeling.

The project combines **machine learning**, **causal inference**, and **graph-based methods** to analyze how user engagement metrics influence interaction quality.

---

## Objectives

* Handle missing values using **Random Forest–based imputation**
* Discover causal relationships using the **PC algorithm**
* Construct and visualize **Directed Acyclic Graphs (DAGs)**
* Estimate **Average Treatment Effect (ATE)** using **Double Machine Learning (DML)**
* Quantify uncertainty using **bootstrap confidence intervals**

---

## Methods & Techniques

### Missing Data Imputation

* Column-wise imputation using **Random Forest Regressors**
* Fallback to **median imputation** for small samples
* Comparison of **original vs imputed distributions**

### Causal Graph Discovery

* Applied the **PC algorithm** (constraint-based causal discovery)
* Used **Fisher’s Z test** for conditional independence testing
* Built and visualized **DAGs** with **NetworkX**

### Causal Effect Estimation

* Implemented **Double Machine Learning (DML)** manually
* Estimated the causal effect of **WatchTime** on **InteractionQuality**
* Controlled for key confounders:

  * BatteryWear
  * SessionQuality
  * Recommendations
* Computed **95% bootstrap confidence intervals**

---

## Tech Stack

* **Programming Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Random Forest)
* **Causal Inference:** causallearn (PC algorithm)
* **Graph Analysis:** NetworkX
* **Visualization:** Matplotlib

---
