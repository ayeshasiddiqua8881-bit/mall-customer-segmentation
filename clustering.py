# ============================================================
#  Mall Customer Segmentation — KMeans Clustering
#  Author  : Aladdin
#  GitHub  : github.com/YOUR_USERNAME/mall-customer-segmentation
# ============================================================
#
#  USAGE:
#    With simulated data  →  python clustering.py
#    With real Kaggle CSV →  python clustering.py --data Mall_Customers.csv
#
#  Kaggle Dataset:
#    https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial
# ============================================================

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Colour palette ───────────────────────────────────────────
COLORS  = ['#534AB7', '#1D9E75', '#BA7517', '#993556', '#185FA5']
LCOLORS = ['#EEEDFE', '#E1F5EE', '#FAEEDA', '#FBEAF0', '#E6F1FB']

SEGMENT_NAMES = [
    'Careful\n(Low Inc, Low Spend)',
    'Standard\n(Mid Inc, Mid Spend)',
    'Impulsive\n(Low Inc, High Spend)',
    'Sensible\n(High Inc, Low Spend)',
    'Target ★\n(High Inc, High Spend)',
]

SEGMENT_INSIGHTS = [
    ("Careful",   "Low income, low spending. Price-sensitive. Target with discounts & value offers."),
    ("Standard",  "Mid income, mid spending. Average customers. Loyalty programs work well here."),
    ("Impulsive", "Low income, high spending. Impulse buyers. Target with flash sales & limited-time offers."),
    ("Sensible",  "High income, low spending. Wealthy but cautious. Target with premium quality messaging."),
    ("Target ★",  "High income, high spending. IDEAL customers. Focus marketing budget here first!"),
]


# ============================================================
#  DATA
# ============================================================

def generate_data(n=200):
    """Simulate mall customer data similar to the Kaggle dataset."""
    segments = [
        # (age_mean, income_mean, score_mean, proportion)
        (44, 26, 20, 0.20),   # Careful
        (38, 55, 50, 0.20),   # Standard
        (25, 25, 78, 0.20),   # Impulsive
        (45, 85, 17, 0.20),   # Sensible
        (32, 85, 82, 0.20),   # Target
    ]
    rows = []
    cid = 1
    for age_m, inc_m, sc_m, frac in segments:
        size = int(n * frac)
        for _ in range(size):
            rows.append({
                'CustomerID':      cid,
                'Gender':          np.random.choice(['Male', 'Female']),
                'Age':             int(np.clip(np.random.normal(age_m,  7), 18, 70)),
                'Annual_Income_k': int(np.clip(np.random.normal(inc_m, 12), 15, 137)),
                'Spending_Score':  int(np.clip(np.random.normal(sc_m,  12),  1,  99)),
            })
            cid += 1
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_data(path=None):
    """Load real Kaggle CSV or fall back to simulated data."""
    if path:
        df = pd.read_csv(path)
        # Normalise column names (Kaggle uses different capitalisations)
        df.columns = [c.strip().replace(' ', '_') for c in df.columns]
        rename = {
            'Annual Income (k$)': 'Annual_Income_k',
            'Spending Score (1-100)': 'Spending_Score',
        }
        df.rename(columns=rename, inplace=True)
        print(f"  Loaded real dataset: {path}  ({len(df)} rows)")
    else:
        df = generate_data(200)
        print("  Using simulated data (200 rows).")
        print("  Tip: run  python clustering.py --data Mall_Customers.csv")
        print("       to use the real Kaggle dataset.")
    return df


# ============================================================
#  STEP 2 — EDA
# ============================================================

def plot_eda(df, save_dir='outputs'):
    print("\n── STEP 2: EDA ──────────────────────────────────────")
    print(f"  Shape      : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Nulls      : {df.isnull().sum().sum()}")
    print(f"  Gender     : {df['Gender'].value_counts().to_dict()}")
    print("\n  Statistical Summary:")
    print(df[['Age', 'Annual_Income_k', 'Spending_Score']].describe().round(2).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('EDA — Mall Customer Dataset', fontsize=14, fontweight='bold')

    features = ['Age', 'Annual_Income_k', 'Spending_Score']
    xlabels  = ['Age', 'Annual Income (k$)', 'Spending Score (1–100)']

    # Histograms
    for i, (feat, lbl, col) in enumerate(zip(features, xlabels, COLORS)):
        ax = axes[0, i]
        ax.hist(df[feat], bins=20, color=col, alpha=0.75, edgecolor='white')
        ax.axvline(df[feat].mean(), color='#D85A30', linestyle='--',
                   linewidth=1.5, label=f'Mean = {df[feat].mean():.1f}')
        ax.set_title(f'Distribution — {lbl}', fontsize=11)
        ax.set_xlabel(lbl); ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    # Gender boxplots
    for i, (feat, lbl) in enumerate(zip(features[:2], xlabels[:2])):
        ax = axes[1, i]
        bp = ax.boxplot(
            [df[df['Gender'] == 'Male'][feat], df[df['Gender'] == 'Female'][feat]],
            patch_artist=True, labels=['Male', 'Female'], widths=0.5,
        )
        bp['boxes'][0].set_facecolor('#B5D4F4')
        bp['boxes'][1].set_facecolor('#F4C0D1')
        for med in bp['medians']:
            med.set_color('#3C3489'); med.set_linewidth(2)
        ax.set_title(f'{lbl} by Gender', fontsize=11)
        ax.set_ylabel(lbl)
        ax.spines[['top', 'right']].set_visible(False)

    # Key scatter: Income vs Spending
    ax = axes[1, 2]
    sc = ax.scatter(df['Annual_Income_k'], df['Spending_Score'],
                    c=df['Age'], cmap='viridis', alpha=0.7, s=35)
    plt.colorbar(sc, ax=ax, label='Age')
    ax.set_xlabel('Annual Income (k$)'); ax.set_ylabel('Spending Score')
    ax.set_title('Income vs Spending (colour = Age)', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = f'{save_dir}/eda_plots.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved → {path}")


# ============================================================
#  STEP 3 — Feature Scaling
# ============================================================

def prepare_features(df):
    print("\n── STEP 3: Feature Scaling ──────────────────────────")
    X = df[['Annual_Income_k', 'Spending_Score']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("  Features : Annual_Income_k, Spending_Score")
    print("  Scaler   : StandardScaler  →  mean=0, std=1")
    return X_scaled, scaler


# ============================================================
#  STEP 4 — Elbow Method + Silhouette Score
# ============================================================

def find_optimal_k(X_scaled, save_dir='outputs'):
    print("\n── STEP 4: Finding Optimal K ────────────────────────")
    K_range = range(2, 11)
    wcss_vals, sil_vals = [], []

    for k in K_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        wcss_vals.append(km.inertia_)
        sil_vals.append(silhouette_score(X_scaled, km.labels_))
        print(f"  K={k:2d}  WCSS={km.inertia_:7.2f}  Silhouette={sil_vals[-1]:.4f}")

    best_k = list(K_range)[sil_vals.index(max(sil_vals))]
    print(f"\n  → Optimal K = {best_k}  (Silhouette peak + Elbow bend)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Finding Optimal Number of Clusters (K)', fontsize=13, fontweight='bold')

    kl = list(K_range)
    axes[0].plot(kl, wcss_vals, 'o-', color='#534AB7', linewidth=2, markersize=7)
    axes[0].axvline(best_k, color='#D85A30', linestyle='--', linewidth=1.5,
                    label=f'Elbow at K={best_k}')
    axes[0].set_title('Elbow Method (WCSS / Inertia)', fontsize=11)
    axes[0].set_xlabel('K'); axes[0].set_ylabel('WCSS')
    axes[0].legend(); axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].bar(kl, sil_vals, color='#1D9E75', alpha=0.75, edgecolor='white')
    axes[1].bar(best_k, max(sil_vals), color='#D85A30', alpha=0.9,
                label=f'Best K={best_k}')
    axes[1].set_title('Silhouette Score (higher = better)', fontsize=11)
    axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette Score')
    axes[1].legend(); axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = f'{save_dir}/elbow_silhouette.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"  Saved → {path}")
    return best_k


# ============================================================
#  STEP 5 — Train KMeans
# ============================================================

def train_kmeans(X_scaled, k):
    print(f"\n── STEP 5: Training KMeans (K={k}) ──────────────────")
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    print(f"  WCSS (Inertia)   : {km.inertia_:.2f}")
    print(f"  Silhouette Score : {silhouette_score(X_scaled, km.labels_):.4f}")
    return km


# ============================================================
#  STEP 6 — Visualise Clusters
# ============================================================

def plot_clusters(df, km, scaler, k, save_dir='outputs'):
    # Order clusters by income (low → high) for consistent labelling
    cluster_income = df.groupby('Cluster')['Annual_Income_k'].mean().sort_values()
    order_map = {old: new for new, old in enumerate(cluster_income.index)}
    df['Cluster_Ordered'] = df['Cluster'].map(order_map)

    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    centers_sorted = sorted(centers_orig, key=lambda c: c[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'KMeans Customer Segmentation Results (K={k})',
                 fontsize=14, fontweight='bold')

    # ── Left: scatter ─────────────────────────────────────
    ax = axes[0]
    for c in range(k):
        mask = df['Cluster_Ordered'] == c
        ax.scatter(df.loc[mask, 'Annual_Income_k'],
                   df.loc[mask, 'Spending_Score'],
                   color=COLORS[c], s=55, alpha=0.75,
                   edgecolors='white', linewidths=0.5)

    for c, (inc, sp) in enumerate(centers_sorted):
        ax.scatter(inc, sp, color='black', marker='X', s=200, zorder=5)
        ax.annotate(f'C{c}', (inc + 1.5, sp + 1.5),
                    fontsize=9, fontweight='bold', color=COLORS[c])

    patches = [mpatches.Patch(color=COLORS[c],
               label=SEGMENT_NAMES[c].replace('\n', ' ')) for c in range(k)]
    ax.legend(handles=patches, fontsize=8, loc='upper left')
    ax.set_xlabel('Annual Income (k$)', fontsize=11)
    ax.set_ylabel('Spending Score (1–100)', fontsize=11)
    ax.set_title('Income vs Spending by Cluster', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Right: profile bars ────────────────────────────────
    ax2 = axes[1]
    x = np.arange(k); w = 0.28
    for i, (feat, lbl, col) in enumerate(zip(
        ['Age', 'Annual_Income_k', 'Spending_Score'],
        ['Avg Age', 'Avg Income (k$)', 'Avg Spend Score'],
        [COLORS[0], COLORS[1], COLORS[2]],
    )):
        vals = [df[df['Cluster_Ordered'] == c][feat].mean() for c in range(k)]
        ax2.bar(x + (i - 1) * w, vals, w, label=lbl, color=col, alpha=0.82,
                edgecolor='white')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{c}' for c in range(k)])
    ax2.set_title('Cluster Profiles (Average Values)', fontsize=11)
    ax2.set_ylabel('Value'); ax2.legend(fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = f'{save_dir}/clusters_final.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"  Saved → {path}")

    # Cluster size summary
    print("\n  Cluster Sizes:")
    print(df['Cluster_Ordered'].value_counts().sort_index().to_string())
    print("\n  Cluster Profiles (mean):")
    print(df.groupby('Cluster_Ordered')[
        ['Age', 'Annual_Income_k', 'Spending_Score']].mean().round(1).to_string())


# ============================================================
#  STEP 7 — Business Insights
# ============================================================

def print_insights(k):
    print("\n" + "=" * 55)
    print("  BUSINESS INSIGHTS — Customer Segments")
    print("=" * 55)
    for i, (name, insight) in enumerate(SEGMENT_INSIGHTS[:k]):
        print(f"\n  C{i} — {name}")
        print(f"  → {insight}")
    print()


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Mall Customer Segmentation using KMeans'
    )
    parser.add_argument('--data',    type=str, default=None,
                        help='Path to Mall_Customers.csv (optional)')
    parser.add_argument('--k',       type=int, default=None,
                        help='Force a specific K instead of auto-detecting')
    parser.add_argument('--out-dir', type=str, default='outputs',
                        help='Folder to save plots (default: outputs)')
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 55)
    print("  MALL CUSTOMER SEGMENTATION — KMeans Project")
    print("=" * 55)

    # Steps
    df                  = load_data(args.data)             # Step 1
    plot_eda(df, args.out_dir)                             # Step 2
    X_scaled, scaler    = prepare_features(df)             # Step 3
    best_k              = args.k or find_optimal_k(        # Step 4
                              X_scaled, args.out_dir)
    km                  = train_kmeans(X_scaled, best_k)   # Step 5
    df['Cluster']       = km.labels_
    plot_clusters(df, km, scaler, best_k, args.out_dir)    # Step 6
    print_insights(best_k)                                 # Step 7

    print("=" * 55)
    print("  PROJECT COMPLETE")
    print(f"  All plots saved in → {args.out_dir}/")
    print("=" * 55)


if __name__ == '__main__':
    main()
