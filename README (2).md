# 🔍 Transaction Fraud Detection System

A machine learning system combining **supervised** and **unsupervised** anomaly detection to identify fraudulent financial transactions in real-time.

> **Results**: Random Forest achieves **0.998 AUC-ROC** on test data. Ensemble approach (supervised + Isolation Forest) catches novel fraud patterns while maintaining low false alarm rates. Estimated **$97M+ annual fraud prevention** at 10M transaction scale.

---

## Why This Project Is Different

Most fraud detection portfolios train a single classifier. This project demonstrates:

1. **Extreme class imbalance handling** — 500:1 legitimate-to-fraud ratio (realistic)
2. **Unsupervised anomaly detection** — Isolation Forest catches zero-day fraud with no labels
3. **Ensemble scoring** — Combines supervised + unsupervised for production-grade detection
4. **Real-time scoring simulation** — Sub-millisecond latency benchmarks
5. **Business impact analysis** — Profit-optimized thresholds, not just accuracy metrics

---

## Results Summary

### Supervised Models
| Model | AUC-ROC | Avg Precision | Best F1 |
|-------|---------|---------------|---------|
| Logistic Regression | ~0.97 | ~0.30 | ~0.35 |
| **Random Forest** | **~0.998** | **~0.55** | **~0.50** |
| Gradient Boosting | ~0.99 | ~0.45 | ~0.40 |

### Ensemble Performance
| Approach | AUC-ROC |
|----------|---------|
| Supervised Only | 0.998 |
| Isolation Forest Only | 0.918 |
| **Ensemble (70/30)** | **0.998** |

*The ensemble matches supervised AUC while adding resilience to novel fraud patterns.*

---

## Project Structure

```
fraud-detection/
├── fraud_detection.py        # Full pipeline (run this)
├── requirements.txt
├── README.md
└── outputs/
    ├── 01_fraud_patterns.png
    ├── 02_correlation.png
    ├── 03_supervised_comparison.png
    ├── 04_feature_importance.png
    ├── 05_confusion_matrix.png
    ├── 06_anomaly_detection.png
    ├── 07_ensemble.png
    ├── 08_business_impact.png
    └── 09_realtime_scoring.png
```

---

## Key Features & Domain Knowledge

### Fraud Signals Engineered
- **Transaction velocity** — Rapid-fire transactions in short windows
- **Amount anomaly** — Spending deviation from customer baseline
- **Geographic risk** — Distance from home + international flags
- **Temporal risk** — Late-night transactions (midnight–5 AM)
- **Channel risk** — Card-not-present (online) vs in-store
- **Decline signals** — Recent failed transactions indicate testing

### Technical Highlights
- **500:1 class imbalance** handled via balanced class weights + threshold optimization
- **Precision-Recall curves** as primary metric (AUC-ROC is misleading for rare events)
- **Isolation Forest** for unsupervised anomaly detection (no labeled data required)
- **RobustScaler** instead of StandardScaler (better for outlier-heavy transaction data)
- **F1-optimized thresholds** instead of default 0.5

---

## Methodology

### 1. Data Generation
50,000 synthetic transactions across 1,000 customers with realistic fraud patterns (velocity attacks, geographic anomalies, amount spikes).

### 2. Feature Engineering (9 features)
Domain-driven features including velocity risk scores, amount anomaly detection, geographic risk composites, and rapid-fire flags.

### 3. Supervised Detection
Three models trained with stratified cross-validation and class balancing. Threshold optimized for F1 rather than using default 0.5.

### 4. Unsupervised Detection (Isolation Forest)
Trained on all data without labels — detects anomalous transactions that deviate from normal patterns. Critical for catching novel fraud.

### 5. Ensemble Scoring
Weighted combination: 70% supervised + 30% unsupervised. Captures both known fraud patterns and novel anomalies.

### 6. Real-Time Simulation
Demonstrates sub-millisecond per-transaction scoring latency with risk tier classification (Low → Block).

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
python fraud_detection.py
```

---

## Tech Stack

- **Python 3.10+**
- **Scikit-learn** — Supervised models, Isolation Forest, evaluation
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization

---

## Future Work

- [ ] Add autoencoders for deep anomaly detection
- [ ] Implement graph-based fraud detection (transaction networks)
- [ ] Deploy as FastAPI microservice with Redis caching
- [ ] Add real-time streaming with Apache Kafka
- [ ] A/B testing framework for model updates
- [ ] Connect to IEEE-CIS Fraud Detection dataset (Kaggle)

---

## Author

**Theodore Johnson**  
[LinkedIn](www.linkedin.com/in/theodore-johnson-m-s-data-science-403b9110b) | [Email](theodorejohnson384@gmail.com)
