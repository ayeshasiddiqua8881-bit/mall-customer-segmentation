# 🛍️ Mall Customer Segmentation
### Unsupervised Machine Learning — KMeans Clustering

> Discover hidden customer groups from raw transactional data — with zero labels.

---

## 📌 Overview

This project applies **KMeans Clustering** to segment mall customers based on their **Annual Income** and **Spending Score**. No labels are used — the algorithm discovers 5 distinct customer groups entirely on its own.

This is a complete, portfolio-ready ML project covering:

- Exploratory Data Analysis (EDA)
- Feature Scaling
- Optimal K selection (Elbow Method + Silhouette Score)
- KMeans training with `k-means++` initialization
- Cluster visualization & Business Insights

---

## 📊 Results — 5 Customer Segments Discovered

| Cluster | Segment | Income | Spending | Marketing Strategy |
|---------|---------|--------|----------|--------------------|
| C0 | Careful | Low | Low | Discounts & value offers |
| C1 | Standard | Mid | Mid | Loyalty programs |
| C2 | Impulsive | Low | High | Flash sales & limited-time deals |
| C3 | Sensible | High | Low | Premium quality messaging |
| **C4** | **Target ★** | **High** | **High** | **Focus budget here first** |

---

## 🖼️ Output Plots

| EDA | Elbow + Silhouette | Final Clusters |
|-----|-------------------|----------------|
| `outputs/eda_plots.png` | `outputs/elbow_silhouette.png` | `outputs/clusters_final.png` |

---

## 📁 Project Structure

```
mall-customer-segmentation/
│
├── clustering.py         ← Main script (fully commented, 7 steps)
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
│
└── outputs/              ← Auto-created when script runs
    ├── eda_plots.png
    ├── elbow_silhouette.png
    └── clusters_final.png
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/mall-customer-segmentation.git
cd mall-customer-segmentation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run with simulated data
```bash
python clustering.py
```

### 4. Run with the real Kaggle dataset
Download [Mall_Customers.csv](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial) and place it in the project folder, then:
```bash
python clustering.py --data Mall_Customers.csv
```

### Optional flags
```bash
# Force a specific K instead of auto-detecting
python clustering.py --k 5

# Save plots to a custom folder
python clustering.py --out-dir my_results
```

---

## 🧠 What I Learned

- **Unsupervised learning** finds structure in data without labels
- **KMeans** minimises Within-Cluster Sum of Squares (WCSS) through iterative centroid updates
- **Elbow Method** looks for the "bend" in WCSS as K increases
- **Silhouette Score** measures how well-separated clusters are (range: −1 to 1, higher is better)
- **StandardScaler** is critical before KMeans — otherwise income (0–137) would dominate spending (0–100) unfairly
- **k-means++ initialization** avoids poor random starts and converges faster

---

## 📐 Key Metrics (K=5)

| Metric | Value |
|--------|-------|
| WCSS (Inertia) | 46.71 |
| Silhouette Score | 0.51 |
| Clusters found | 5 |
| Features used | Annual Income, Spending Score |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-purple?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-green)

---

## 🗂️ Dataset

- **Source:** [Kaggle — Customer Segmentation Tutorial](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)
- **Features used:** `Annual_Income_k`, `Spending_Score`
- **Rows:** 200 customers

The script includes a built-in data simulator so you can run the project immediately without downloading anything.

---

## 👤 Author

**Aladdin**
B.Sc. Information Technology — Maharashtra, India
Aspiring AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
