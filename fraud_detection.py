"""
Transaction Fraud Detection System
=====================================
Anomaly detection pipeline for identifying fraudulent financial transactions.

Author: [Your Name]
Date: 2026
Tech Stack: Python, Pandas, Scikit-learn, Matplotlib, Seaborn

Approach:
    1. Supervised: Random Forest, Gradient Boosting (labeled data)
    2. Unsupervised: Isolation Forest (anomaly detection)
    3. Ensemble: Combine both for production-grade detection

Business Context:
    Payment fraud costs $30B+ annually. This system flags suspicious
    transactions while minimizing false positives that block legitimate purchases.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


def generate_transaction_data(n=50000, fraud_rate=0.002, seed=42):
    np.random.seed(seed)
    n_cust = 1000
    cids = np.random.randint(1, n_cust+1, n)
    cavg = {i: np.random.lognormal(3.5, 0.8) for i in range(1, n_cust+1)}
    clat = {i: np.random.uniform(25, 48) for i in range(1, n_cust+1)}

    base = np.array([cavg[c] for c in cids])
    amount = np.abs(base * np.random.lognormal(0, 0.6, n)).clip(0.5, 25000)

    _hp = [.01,.005,.003,.002,.002,.005,.02,.04,.06,.07,.08,.09,
           .10,.08,.07,.06,.05,.04,.05,.06,.05,.04,.03,.02]
    _hp = [p/sum(_hp) for p in _hp]
    hour = np.random.choice(24, n, p=_hp)
    dow = np.random.choice(7, n, p=[.12,.15,.16,.16,.16,.14,.11])
    channel = np.random.choice(['in_store','online','mobile_app','phone'], n, p=[.35,.35,.25,.05])
    merchant = np.random.choice(
        ['grocery','gas_station','restaurant','electronics','clothing',
         'travel','entertainment','healthcare','utilities','other'],
        n, p=[.20,.12,.15,.10,.10,.08,.08,.05,.05,.07])

    dist = np.abs(np.random.exponential(15, n)).clip(0, 5000)
    intl = (np.random.random(n) < 0.05).astype(int)
    t1hr = np.random.poisson(1.5, n).clip(0, 20)
    t24hr = np.random.poisson(5, n).clip(0, 50)
    amt_ratio = amount / (np.array([cavg[c] for c in cids]) + 1)
    uniq_merch = np.random.poisson(3, n).clip(1, 15)
    time_since = np.random.exponential(120, n).clip(0.5, 1440)
    declined = np.random.poisson(0.1, n).clip(0, 5)

    fs = (-6.0 + 1.5*(amount > np.percentile(amount, 95)).astype(float)
        + 2.0*(amount > np.percentile(amount, 99)).astype(float)
        + 1.0*(hour < 5).astype(float) + 0.8*(channel != 'in_store').astype(float)
        + 1.5*(t1hr > 5).astype(float) + 1.0*(t24hr > 15).astype(float)
        + 1.2*(dist > 500).astype(float) + 1.5*intl
        + 0.8*(amt_ratio > 5).astype(float) + 1.0*(declined > 0).astype(float)
        + 0.5*(time_since < 5).astype(float) + 0.8*(merchant == 'electronics').astype(float)
        + np.random.normal(0, 0.8, n))
    fp = 1/(1+np.exp(-fs))
    fraud = (fp > np.percentile(fp, 100*(1-fraud_rate))).astype(int)

    return pd.DataFrame({
        'customer_id': cids, 'transaction_amount': np.round(amount, 2),
        'hour_of_day': hour, 'day_of_week': dow,
        'is_weekend': (dow >= 5).astype(int),
        'is_online': (channel != 'in_store').astype(int),
        'card_present': (channel == 'in_store').astype(int),
        'merchant_category': merchant,
        'distance_from_home_km': np.round(dist, 1),
        'is_international': intl,
        'txns_last_1hr': t1hr, 'txns_last_24hr': t24hr,
        'amount_ratio_to_avg': np.round(amt_ratio, 2),
        'unique_merchants_24hr': uniq_merch,
        'time_since_last_txn_min': np.round(time_since, 1),
        'declined_last_24hr': declined, 'is_fraud': fraud
    })


def run_eda(df, save_dir='outputs'):
    import os; os.makedirs(save_dir, exist_ok=True)
    nf = df['is_fraud'].sum(); nl = len(df) - nf
    print("="*60+"\nEXPLORATORY DATA ANALYSIS\n"+"="*60)
    print(f"\n{len(df):,} transactions | Fraud: {nf:,} ({df['is_fraud'].mean():.2%}) | Ratio: {nl//max(nf,1)}:1")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Transaction Fraud Patterns', fontsize=16, fontweight='bold', y=1.02)

    for l, c, nm in [(0,'#2e7d32','Legit'),(1,'#d32f2f','Fraud')]:
        axes[0,0].hist(np.log10(df[df['is_fraud']==l]['transaction_amount']+1),
                       bins=50, alpha=0.6, color=c, label=nm, density=True)
    axes[0,0].set_title('Amount (log10)', fontweight='bold'); axes[0,0].legend()

    fr_hr = df.groupby('hour_of_day')['is_fraud'].mean()
    axes[0,1].bar(fr_hr.index, fr_hr.values*100, color='#d32f2f', edgecolor='white')
    axes[0,1].set_title('Fraud Rate by Hour', fontweight='bold'); axes[0,1].set_ylabel('Fraud Rate %')

    fr_ch = df.groupby('is_online')['is_fraud'].mean()
    axes[0,2].bar(['In-Store','Online/Mobile'], fr_ch.values*100, color=['#2e7d32','#d32f2f'], edgecolor='white')
    axes[0,2].set_title('Fraud: In-Store vs Online', fontweight='bold')

    fr_v = df.groupby(pd.cut(df['txns_last_1hr'], bins=[0,2,5,10,20]))['is_fraud'].mean()
    axes[1,0].bar(range(len(fr_v)), fr_v.values*100, color=['#2e7d32','#fbc02d','#f57c00','#d32f2f'], edgecolor='white')
    axes[1,0].set_xticks(range(len(fr_v))); axes[1,0].set_xticklabels(['0-2','3-5','6-10','11-20'])
    axes[1,0].set_title('Fraud Rate by Velocity (1hr)', fontweight='bold')

    fr_d = df.groupby(pd.cut(df['distance_from_home_km'], bins=[0,10,50,200,5000]))['is_fraud'].mean().dropna()
    dcolors = ['#2e7d32','#fbc02d','#f57c00','#d32f2f'][:len(fr_d)]
    axes[1,1].bar(range(len(fr_d)), fr_d.values*100, color=dcolors, edgecolor='white')
    axes[1,1].set_xticks(range(len(fr_d))); axes[1,1].set_xticklabels([str(x) for x in fr_d.index])
    axes[1,1].set_title('Fraud by Distance from Home', fontweight='bold')

    fr_m = df.groupby('merchant_category')['is_fraud'].mean().sort_values()
    axes[1,2].barh(fr_m.index, fr_m.values*100, color='#1976d2', edgecolor='white')
    axes[1,2].set_title('Fraud by Merchant', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_fraud_patterns.png', dpi=150, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(12, 9))
    num = df.select_dtypes(include=[np.number]).columns.drop('customer_id')
    corr = df[num].corr(); mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
    ax.set_title('Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_correlation.png', dpi=150, bbox_inches='tight'); plt.close()
    print("✓ Saved EDA charts")
    return df


def engineer_features(df):
    df = df.copy()
    df['log_amount'] = np.log1p(df['transaction_amount'])
    df['is_night_txn'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] < 5)).astype(int)
    df['velocity_risk'] = (df['txns_last_1hr']/3 + df['txns_last_24hr']/10) / 2
    df['amount_anomaly'] = np.log1p(df['amount_ratio_to_avg'])
    df['geo_risk'] = np.log1p(df['distance_from_home_km']) * (1 + df['is_international'])
    df['rapid_fire'] = (df['time_since_last_txn_min'] < 5).astype(int)
    df['combined_risk'] = (df['velocity_risk']*.25 + df['amount_anomaly']*.25
        + df['geo_risk']/10*.2 + df['is_night_txn']*.15 + df['is_online']*.15)
    df['high_risk_merchant'] = df['merchant_category'].isin(['electronics','travel']).astype(int)
    df['has_recent_declines'] = (df['declined_last_24hr'] > 0).astype(int)
    print(f"✓ Engineered 9 features (total: {df.shape[1]-1})")
    return df


def preprocess(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)
    drop = ['is_fraud', 'customer_id']
    fcols = [c for c in df.columns if c not in drop]
    X, y = df[fcols], df['is_fraud']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = RobustScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xte = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)
    print(f"✓ Train: {len(Xtr):,} | Test: {len(Xte):,} | Features: {Xtr.shape[1]}")
    print(f"  Fraud rate — Train: {ytr.mean():.3%} | Test: {yte.mean():.3%}")
    return Xtr, Xte, ytr, yte, fcols, scaler


def train_supervised(Xtr, Xte, ytr, yte, fcols, save_dir='outputs'):
    import os; os.makedirs(save_dir, exist_ok=True)
    print("\n"+"="*60+"\nSUPERVISED FRAUD DETECTION\n"+"="*60)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
            min_samples_leaf=10, random_state=42, subsample=0.8)
    }

    results = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(model, Xtr, ytr, cv=cv, scoring='roc_auc')
        model.fit(Xtr, ytr)
        yp = model.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, yp)
        ap = average_precision_score(yte, yp)

        prec, rec, thr = precision_recall_curve(yte, yp)
        f1s = 2*prec*rec/(prec+rec+1e-8)
        best_t = thr[np.argmax(f1s[:-1])]
        ypred = (yp >= best_t).astype(int)
        bf1 = f1_score(yte, ypred)
        caught = ((ypred==1)&(yte==1)).sum(); total_fraud = (yte==1).sum()
        false_alarms = ((ypred==1)&(yte==0)).sum()

        print(f"  CV AUC: {cv_auc.mean():.4f} | Test AUC: {auc:.4f} | AP: {ap:.4f} | F1: {bf1:.4f}")
        print(f"  Fraud caught: {caught}/{total_fraud} ({caught/max(total_fraud,1):.1%})")
        print(f"  False alarms: {false_alarms} ({false_alarms/len(yte):.2%})")

        results[name] = {'model':model,'y_pred':ypred,'y_prob':yp,'auc':auc,'ap':ap,'f1':bf1,'threshold':best_t}

    colors = {'Logistic Regression':'#1976d2','Random Forest':'#2e7d32','Gradient Boosting':'#d32f2f'}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for n, r in results.items():
        fpr, tpr, _ = roc_curve(yte, r['y_prob'])
        axes[0].plot(fpr, tpr, label=f"{n} (AUC={r['auc']:.3f})", color=colors[n], linewidth=2)
        pr, rc, _ = precision_recall_curve(yte, r['y_prob'])
        axes[1].plot(rc, pr, label=f"{n} (AP={r['ap']:.3f})", color=colors[n], linewidth=2)
    axes[0].plot([0,1],[0,1],'k--',alpha=0.3)
    axes[0].set_title('ROC Curves', fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title('Precision-Recall (Key for Imbalanced Data)', fontweight='bold')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_supervised_comparison.png', dpi=150, bbox_inches='tight'); plt.close()

    best_name = max(results, key=lambda k: results[k]['auc'])
    bm = results[best_name]['model']
    if hasattr(bm, 'feature_importances_'):
        imp = pd.Series(bm.feature_importances_, index=fcols).nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(10, 8))
        imp.plot(kind='barh', ax=ax, color='#1976d2', edgecolor='white')
        ax.set_title(f'Top 15 Fraud Predictors ({best_name})', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/04_feature_importance.png', dpi=150, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(yte, results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'])
    ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/05_confusion_matrix.png', dpi=150, bbox_inches='tight'); plt.close()

    print(f"\n✓ Best supervised: {best_name} (AUC={results[best_name]['auc']:.4f})")
    return results, best_name


def train_anomaly_detection(Xtr, Xte, yte, save_dir='outputs'):
    import os; os.makedirs(save_dir, exist_ok=True)
    print("\n"+"="*60+"\nUNSUPERVISED: ISOLATION FOREST\n"+"="*60)

    iso = IsolationForest(n_estimators=300, contamination=0.005, random_state=42, n_jobs=-1)
    iso.fit(Xtr)
    labels = iso.predict(Xte)
    scores = -iso.score_samples(Xte)
    ypred = (labels == -1).astype(int)

    nf = yte.sum(); caught = ((ypred==1)&(yte==1)).sum(); fa = ((ypred==1)&(yte==0)).sum()
    print(f"  Flagged: {ypred.sum():,} | Caught: {caught}/{nf} ({caught/max(nf,1):.1%}) | False alarms: {fa}")

    if nf > 0:
        auc = roc_auc_score(yte, scores); ap = average_precision_score(yte, scores)
        print(f"  AUC: {auc:.4f} | AP: {ap:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for l, c, nm in [(0,'#2e7d32','Legit'),(1,'#d32f2f','Fraud')]:
        axes[0].hist(scores[yte==l], bins=60, alpha=0.6, color=c, label=nm, density=True)
    axes[0].set_title('Isolation Forest Anomaly Scores', fontweight='bold'); axes[0].legend()

    if nf > 0:
        fpr, tpr, _ = roc_curve(yte, scores)
        axes[1].plot(fpr, tpr, color='#7b1fa2', linewidth=2, label=f'IF (AUC={auc:.3f})')
        axes[1].plot([0,1],[0,1],'k--',alpha=0.3)
        axes[1].set_title('Isolation Forest ROC', fontweight='bold'); axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_anomaly_detection.png', dpi=150, bbox_inches='tight'); plt.close()
    print(f"✓ Saved anomaly detection charts")
    return iso, scores


def ensemble_and_business(sup_probs, anom_scores, yte, save_dir='outputs'):
    import os; os.makedirs(save_dir, exist_ok=True)
    print("\n"+"="*60+"\nENSEMBLE + BUSINESS IMPACT\n"+"="*60)

    an = (anom_scores - anom_scores.min()) / (anom_scores.max() - anom_scores.min() + 1e-8)
    ens = 0.7 * sup_probs + 0.3 * an

    auc_s = roc_auc_score(yte, sup_probs)
    auc_u = roc_auc_score(yte, anom_scores)
    auc_e = roc_auc_score(yte, ens)
    ap_e = average_precision_score(yte, ens)

    print(f"\n  Supervised AUC:   {auc_s:.4f}")
    print(f"  Unsupervised AUC: {auc_u:.4f}")
    print(f"  Ensemble AUC:     {auc_e:.4f} | AP: {ap_e:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for sc, nm, c in [(sup_probs,'Supervised','#1976d2'),(anom_scores,'Isolation Forest','#7b1fa2'),(ens,'Ensemble','#d32f2f')]:
        fpr, tpr, _ = roc_curve(yte, sc)
        axes[0].plot(fpr, tpr, label=f"{nm} (AUC={roc_auc_score(yte, sc):.3f})", color=c, linewidth=2)
    axes[0].plot([0,1],[0,1],'k--',alpha=0.3)
    axes[0].set_title('Ensemble vs Individual', fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.3)

    for l, c, nm in [(0,'#2e7d32','Legit'),(1,'#d32f2f','Fraud')]:
        axes[1].hist(ens[yte==l], bins=60, alpha=0.6, color=c, label=nm, density=True)
    axes[1].set_title('Ensemble Score Distribution', fontweight='bold'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/07_ensemble.png', dpi=150, bbox_inches='tight'); plt.close()

    # Business impact
    avg_fraud_loss = 5000; alert_cost = 15; n_txns_annual = 10_000_000
    prec_arr, rec_arr, thr_arr = precision_recall_curve(yte, ens)
    profits = []
    for i in range(0, len(thr_arr), max(1, len(thr_arr)//50)):
        t = thr_arr[i]; p = prec_arr[i]; r = rec_arr[i]
        yp = (ens >= t).astype(int)
        caught = ((yp==1)&(yte==1)).sum(); alerts = yp.sum()
        fraud_saved = caught * avg_fraud_loss; alert_costs = alerts * alert_cost
        profits.append({'threshold':t, 'precision':p, 'recall':r,
                        'net_savings': fraud_saved - alert_costs, 'alerts': alerts})
    pdf = pd.DataFrame(profits)
    opt = pdf.iloc[pdf['net_savings'].idxmax()]
    scale = n_txns_annual / len(yte)

    print(f"\n  BUSINESS IMPACT (scaled to {n_txns_annual/1e6:.0f}M txns/year):")
    print(f"  Optimal threshold: {opt['threshold']:.3f}")
    print(f"  Precision: {opt['precision']:.1%} | Recall: {opt['recall']:.1%}")
    print(f"  Est. annual fraud prevented: ${opt['net_savings']*scale:,.0f}")
    print(f"  Est. annual alerts: {opt['alerts']*scale:,.0f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pdf['threshold'], pdf['net_savings']/1000, color='#1976d2', linewidth=2.5)
    ax.axvline(x=opt['threshold'], color='#d32f2f', linestyle='--', label=f"Optimal: {opt['threshold']:.3f}")
    ax.scatter([opt['threshold']], [opt['net_savings']/1000], color='#d32f2f', s=100, zorder=5)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Net Savings ($K, test set)')
    ax.set_title('Fraud Detection: Profit Optimization', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/08_business_impact.png', dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n✓ Saved ensemble + business charts")
    return ens


def simulate_realtime(model, scaler_unused, Xte, yte, save_dir='outputs'):
    """Simulate real-time fraud scoring to demonstrate production readiness."""
    import os, time; os.makedirs(save_dir, exist_ok=True)
    print("\n"+"="*60+"\nREAL-TIME SCORING SIMULATION\n"+"="*60)

    sample = Xte.sample(n=min(100, len(Xte)), random_state=42)
    sample_y = yte.loc[sample.index]

    start = time.time()
    probs = model.predict_proba(sample)[:,1]
    elapsed = time.time() - start

    print(f"\n  Scored {len(sample)} transactions in {elapsed*1000:.1f}ms")
    print(f"  Avg latency: {elapsed/len(sample)*1000:.2f}ms per transaction")
    print(f"  Throughput: {len(sample)/elapsed:,.0f} txns/second")

    # Risk tiers
    tiers = pd.cut(probs, bins=[0, 0.1, 0.4, 0.7, 1.0],
                   labels=['Low Risk', 'Medium Risk', 'High Risk', 'Block'])
    tier_counts = tiers.value_counts().sort_index()

    print(f"\n  Risk Tier Distribution:")
    for tier, count in tier_counts.items():
        print(f"    {tier}: {count} ({count/len(sample):.0%})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tier_counts.plot(kind='bar', ax=axes[0], color=['#2e7d32','#fbc02d','#f57c00','#d32f2f'], edgecolor='white')
    axes[0].set_title('Real-Time Risk Tier Distribution', fontweight='bold')
    axes[0].set_ylabel('Count'); axes[0].tick_params(axis='x', rotation=0)

    axes[1].hist(probs, bins=30, color='#1976d2', edgecolor='white', alpha=0.8)
    for t, c in [(0.1,'#fbc02d'),(0.4,'#f57c00'),(0.7,'#d32f2f')]:
        axes[1].axvline(x=t, color=c, linestyle='--', linewidth=1.5)
    axes[1].set_title('Real-Time Fraud Score Distribution', fontweight='bold')
    axes[1].set_xlabel('Fraud Probability')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/09_realtime_scoring.png', dpi=150, bbox_inches='tight'); plt.close()
    print(f"✓ Saved real-time scoring chart")


if __name__ == "__main__":
    print("╔"+"═"*58+"╗")
    print("║  TRANSACTION FRAUD DETECTION SYSTEM                      ║")
    print("║  Supervised + Unsupervised + Ensemble                    ║")
    print("╚"+"═"*58+"╝")

    print("\n[1/7] Generating data...")
    df = generate_transaction_data(50000)

    print("\n[2/7] EDA...")
    df = run_eda(df)

    print("\n[3/7] Feature engineering...")
    df = engineer_features(df)

    print("\n[4/7] Preprocessing...")
    Xtr, Xte, ytr, yte, fcols, scaler = preprocess(df)

    print("\n[5/7] Supervised models...")
    results, best = train_supervised(Xtr, Xte, ytr, yte, fcols)

    print("\n[6/7] Anomaly detection...")
    iso, anom_scores = train_anomaly_detection(Xtr, Xte, yte)

    print("\n[7/7] Ensemble + Business impact...")
    ens = ensemble_and_business(results[best]['y_prob'], anom_scores, yte)

    simulate_realtime(results[best]['model'], scaler, Xte, yte)

    print("\n"+"="*60)
    print(f"✅ COMPLETE | Best supervised: {best} (AUC={results[best]['auc']:.4f})")
    print("   9 charts saved to outputs/")
    print("="*60)
