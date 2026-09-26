# Data Science & AI/ML Practical Exam — Set A

**Student Name:** Deepvejpara
**Student ID:** 10828
**Set:** A
**Repository:** [Deepvejpara/ds-aiml-set-A-10828](https://github.com/Deepvejpara/ds-aiml-set-A-10828)

---

## 📌 Project Overview

This project is a practical Data Science & AI/ML examination project focused on **predicting campaign responses and identifying audience segments** using a synthetic dataset.

The project covers the complete machine learning workflow:

* Data generation and auditing
* Data cleaning and duplicate removal
* Statistical analysis and inference
* Linear algebra and covariance analysis
* Data preprocessing and feature engineering
* Supervised classification
* Unsupervised clustering
* Artificial Neural Network (ANN)
* Model evaluation and comparison
* Reproducible outputs and saved model artifacts

The dataset contains synthetic observations representing customer/audience behavior through variables such as visits, recency, engagement, and spend.

---

## 🎯 Objective

The main objectives of this project are to:

1. Analyze audience engagement using descriptive statistics.
2. Test whether engagement differs between operational groups G1 and G2.
3. Perform covariance and eigenvalue analysis.
4. Clean and preprocess the dataset without introducing data leakage.
5. Engineer an additional feature for campaign-response prediction.
6. Build a Logistic Regression classifier and compare it against a baseline.
7. Segment the audience using K-Means clustering.
8. Build and evaluate a feed-forward Artificial Neural Network.
9. Compare predictive model performance on the same untouched test set.
10. Preserve reproducible outputs, predictions, preprocessing artifacts, and the trained ANN.

---

## 📊 Dataset

The supplied dataset is synthetic and contains **305 rows and 7 columns**, including 5 exact duplicate records.

After removing exact duplicates:

* **Unique records:** 300
* **Features:** `visits`, `recency`, `engagement`, `spend`
* **Categorical feature:** `group`
* **Target:** `response`
* **Identifier:** `record_id`

### Data Dictionary

| Column       | Description                                                    |
| ------------ | -------------------------------------------------------------- |
| `record_id`  | Unique record identifier; excluded from modeling               |
| `visits`     | Audience visit index                                           |
| `recency`    | Recency index                                                  |
| `engagement` | Audience engagement index                                      |
| `spend`      | Spending index                                                 |
| `group`      | Operational cohort: G1 or G2                                   |
| `response`   | Binary campaign response target: 1 = response, 0 = no response |

The numeric variables are **synthetic, dimensionless index measurements**, not physical units.

---

## 🧹 Data Cleaning & Preprocessing

The original dataset contains:

* **305 total rows**
* **5 exact duplicates**
* **300 unique records**

Missing values were present in:

| Feature      | Missing Values |
| ------------ | -------------: |
| `visits`     |             16 |
| `recency`    |             15 |
| `engagement` |              0 |
| `spend`      |              0 |
| `group`      |              0 |
| `response`   |              0 |

Exact duplicates were removed **before splitting the data**.

### Feature Engineering

The following feature was created:

```text
engineered_feature = engagement / (recency + 1)
```

The feature was calculated after numeric imputation and before scaling.

### Preprocessing

* Numeric missing values were handled using **median imputation**.
* `group` was one-hot encoded.
* Numeric features were standardized using `StandardScaler`.
* One-hot encoded categorical features were kept unscaled.
* `record_id` and `response` were excluded from predictive features.
* Preprocessing was fitted only on the training/fit data and reused for validation and test data.

This prevents information from the test set from leaking into model training.

---

## 🔀 Data Splitting

The cleaned dataset was divided using a **stratified split** with `random_state=42`.

| Partition      | Records |
| -------------- | ------: |
| Fit / Training |     192 |
| Validation     |      48 |
| Test           |      60 |
| **Total**      | **300** |

The test set remained untouched until final model evaluation.

The split IDs are stored in:

```text
outputs/splits.csv
```

A disjointness check confirmed that the fit, validation, and test partitions do not overlap.

---

# 📈 Task 1 — Mathematics & Advanced Statistics

## Descriptive Statistics

For the 192 fit records, the observed engagement values were analyzed without imputation for the statistical calculation.

| Statistic                 |   Value |
| ------------------------- | ------: |
| Observed n                |     192 |
| Mean                      | 51.0018 |
| Median                    | 51.4050 |
| Sample Standard Deviation | 10.2890 |

The engagement distribution is also saved as:

```text
outputs/figures/engagement_histogram.png
```

---

## Welch's t-Test

The project tests whether mean engagement differs between groups G1 and G2.

### Hypotheses

**H₀:** Mean engagement in G1 equals mean engagement in G2.

**H₁:** Mean engagement in G1 is not equal to mean engagement in G2.

Results:

| Metric             |   Value |
| ------------------ | ------: |
| G1 observations    |      84 |
| G2 observations    |     108 |
| G1 mean            | 51.6149 |
| G2 mean            | 50.5250 |
| t-statistic        |  0.7230 |
| p-value            |  0.4707 |
| Significance level |    0.05 |

Since the p-value is greater than 0.05, the analysis does not reject the null hypothesis for this synthetic sample.

### 95% Confidence Interval

The 95% confidence interval for the overall observed mean engagement is:

```text
[49.5372, 52.4665]
```

The inference assumes independent observations and an approximately normal sampling distribution of the sample mean.

---

## Linear Algebra

For complete fit rows containing `engagement` and `visits`, the centered matrix was used to calculate the sample covariance matrix.

The covariance matrix was:

```text
[[103.6571, 4.1181],
 [  4.1181, 103.0612]]
```

Eigenvalues:

```text
107.4880
99.2303
```

The largest eigenvalue accounts for approximately:

```text
51.997% of total variance
```

The corresponding principal eigenvector was approximately:

```text
[-0.7322, -0.6811]
```

This indicates that the first principal direction captures slightly more than half of the variance in the two-dimensional `engagement` and `visits` space.

---

# 🤖 Task 2 — Supervised Learning

## Baseline

A `DummyClassifier` using the most-frequent strategy was used as the baseline.

## Logistic Regression

A Logistic Regression classifier was trained using the transformed fit data.

Configuration:

```text
Model: LogisticRegression
max_iter: 1000
Threshold: 0.5
random_state: 42
```

### Test Performance

| Model               | Accuracy | Precision | Recall |     F1 |
| ------------------- | -------: | --------: | -----: | -----: |
| Dummy Baseline      |   0.5833 |    0.0000 | 0.0000 | 0.0000 |
| Logistic Regression |   0.8167 |    0.8333 | 0.8571 | 0.8451 |
| ANN                 |   0.8333 |    0.8378 | 0.8857 | 0.8611 |

The Logistic Regression model provides a substantial improvement over the majority-class baseline on the held-out test set.

The confusion matrix is available at:

```text
outputs/figures/confusion_matrix_logistic.png
```

Test predictions are stored in:

```text
outputs/test_predictions_logistic.csv
```

Each prediction includes:

* `record_id`
* true response
* predicted response
* class-1 probability

---

# 👥 Task 3 — Unsupervised Learning

K-Means clustering was performed using the transformed numeric predictors and engineered feature.

The following values of `k` were evaluated:

```text
k = 2, 3, 4
```

Configuration:

```text
n_init = 10
random_state = 42
```

### K-Means Diagnostics

|  k |  Inertia | Silhouette Score |
| -: | -------: | ---------------: |
|  2 | 713.0304 |       **0.2259** |
|  3 | 613.5511 |           0.1872 |
|  4 | 541.6453 |           0.1907 |

The selected value was:

```text
k = 2
```

because it produced the highest silhouette score.

### Cluster Profiles

#### Cluster 0 — Moderate-Engagement Value Seekers

* Records: 109
* Visits: 49.36
* Recency: 55.42
* Engagement: 46.09
* Spend: 50.18
* Engineered feature: 0.826

A practical campaign action for this segment is to use targeted engagement campaigns designed to increase activity and repeat interaction.

#### Cluster 1 — High-Value Active Advocates

* Records: 83
* Visits: 49.54
* Recency: 43.93
* Engagement: 57.45
* Spend: 50.97
* Engineered feature: 1.289

This segment shows higher engagement and a stronger engineered engagement-to-recency ratio, making it suitable for retention, loyalty, or advocacy-oriented campaigns.

Cluster IDs are arbitrary identifiers and do not represent the target classes.

Cluster diagnostics and profiles are stored in:

```text
outputs/kmeans_diagnostics.json
outputs/cluster_profiles.csv
```

---

# 🧠 Task 4 — Artificial Neural Network

A dense feed-forward ANN was implemented using TensorFlow/Keras.

### Architecture

```text
Input
  ↓
Dense(16, ReLU)
  ↓
Dense(8, ReLU)
  ↓
Dense(1, Sigmoid)
```

### Training Configuration

| Parameter            | Setting                        |
| -------------------- | ------------------------------ |
| Loss                 | Binary Cross-Entropy           |
| Optimizer            | Adam                           |
| Learning Rate        | 0.001                          |
| Batch Size           | 16                             |
| Maximum Epochs       | 50                             |
| Validation           | 48-record validation partition |
| Early Stopping       | Patience = 5                   |
| Restore Best Weights | Yes                            |
| Random Seed          | 42                             |
| Output Activation    | Sigmoid                        |

The sigmoid output is appropriate for binary classification, while binary cross-entropy measures the difference between predicted probabilities and the binary target.

The trained model is saved as:

```text
models/best_ann_model.keras
```

The preprocessing object is saved as:

```text
models/preprocessor.joblib
```

Training and validation loss curves are available at:

```text
outputs/figures/ann_loss_curves.png
```

---

# 📊 Final Model Comparison

All predictive models were evaluated on the **same untouched 60-record test set**.

| Model                     | Accuracy | Precision | Recall | F1-Score |
| ------------------------- | -------: | --------: | -----: | -------: |
| Dummy Baseline            |   58.33% |    0.0000 | 0.0000 |   0.0000 |
| Logistic Regression       |   81.67% |    0.8333 | 0.8571 |   0.8451 |
| Artificial Neural Network |   83.33% |    0.8378 | 0.8857 |   0.8611 |

The ANN achieved an F1-score of **0.8611**, while Logistic Regression achieved **0.8451** on the same test records.

The complete comparison is stored in:

```text
outputs/model_metric_comparison.csv
```

---

# 🔍 Key Findings

### Finding 1 — Engagement by Cohort

The Welch's t-test produced:

```text
p-value = 0.4707
```

At α = 0.05, the sample does not provide sufficient statistical evidence of a difference in mean engagement between G1 and G2.

### Finding 2 — Predictive Modeling

Both Logistic Regression and the ANN performed substantially better than the majority-class baseline on the held-out test set.

* Logistic Regression F1: **0.8451**
* ANN F1: **0.8611**
* Baseline F1: **0.0000**

### Finding 3 — Audience Segmentation

K-Means selected **2 clusters** based on the highest silhouette score:

```text
Silhouette score = 0.2259
```

The resulting segments differ mainly in engagement, recency, and the engineered engagement/recency feature.

---

# 💡 Recommendation & Limitation

For this synthetic dataset, the ANN produced a higher test F1-score than Logistic Regression. However, model selection should consider more than a single metric from a single small holdout set.

The dataset contains only **300 unique synthetic observations**, so the results should not be interpreted as evidence of real-world campaign performance.

Further evaluation on real campaign data, additional validation, and controlled experimentation would be required before deployment.

---

# 🗂️ Repository Structure

```text
ds-aiml-set-A-10828/
│
├── data/
│   └── raw/
│       └── set_d.csv
│
├── models/
│   ├── best_ann_model.keras
│   └── preprocessor.joblib
│
├── outputs/
│   ├── cluster_profiles.csv
│   ├── covariance_eigenvalues.json
│   ├── inference_results.json
│   ├── kmeans_diagnostics.json
│   ├── model_metric_comparison.csv
│   ├── preprocessing_audit.json
│   ├── splits.csv
│   ├── stats_summary.json
│   ├── test_predictions_ann.csv
│   ├── test_predictions_logistic.csv
│   │
│   └── figures/
│       ├── ann_loss_curves.png
│       ├── confusion_matrix_ann.png
│       ├── confusion_matrix_logistic.png
│       └── engagement_histogram.png
│
├── src/
│   └── generate_data.py
│
├── exam.ipynb
├── requirements.txt
└── .gitignore
```

---

# ⚙️ Technologies Used

* Python 3.13
* NumPy
* pandas
* SciPy
* scikit-learn
* Matplotlib
* Seaborn
* TensorFlow / Keras
* Joblib
* Jupyter Notebook / Google Colab
* Git & GitHub

### Package Versions

```text
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
tensorflow==2.16.1
joblib==1.4.2
```

---

# 🚀 Setup & Execution

Clone the repository:

```bash
git clone https://github.com/Deepvejpara/ds-aiml-set-A-10828.git
cd ds-aiml-set-A-10828
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The supplied dataset generator can be run from the repository root with:

```bash
python src/generate_data.py
```

Then open and execute:

```text
exam.ipynb
```

The notebook contains the complete analysis, preprocessing, statistical calculations, supervised learning, clustering, ANN training, evaluation, and output generation.

---

# 📁 Reproducibility

The project uses fixed random seeds where required:

```text
random_state = 42
```

The data generator uses its supplied random seed and should remain unchanged.

The preprocessing pipeline is fitted only on the fit partition. The same preprocessing is then applied to validation and test data.

Both Logistic Regression and ANN are evaluated using the same untouched 60-record test partition.

Saved split IDs can be checked in:

```text
outputs/splits.csv
```

Saved predictions can be compared against:

```text
outputs/test_predictions_logistic.csv
outputs/test_predictions_ann.csv
```

---

# 🎥 Project Explanation Video

**Video Link:** *Add your accessible YouTube/Google Drive video link here*

**Duration:** *Add duration here*

The video covers:

* Problem and dataset
* Data cleaning and splitting
* Statistical analysis
* Preprocessing and feature engineering
* Logistic Regression
* K-Means segmentation
* ANN architecture and training
* Model comparison
* Key findings and limitations
* Repository walkthrough

---

# ⚠️ Limitations

* The dataset is synthetic.
* Only 300 unique records are available.
* Model evaluation is based on one held-out test set.
* The dataset does not represent real customer behavior.
* Clustering results depend on the available synthetic features.
* Real-world deployment would require additional validation and monitoring.
* No causal conclusions should be drawn from these analyses.

---

# 📚 References

* Python
* NumPy
* pandas
* SciPy
* scikit-learn
* Matplotlib
* Seaborn
* TensorFlow / Keras

---

## 👨‍💻 Author

**Deepvejpara**
GitHub: [@Deepvejpara](https://github.com/Deepvejpara)

---

## 📜 Declaration

> All work in this repository is my own except where external libraries, references, or resources have been used and cited.
