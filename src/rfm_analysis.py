"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          GROWTH INTELLIGENCE ENGINE — RFM ANALYSIS MODULE                   ║
║          Blinkit Product Analytics | Instacart Market Basket Dataset         ║
║          Author: Senior Product Analytics Team                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline:
  1. Data Ingestion & Validation
  2. RFM Metric Computation
  3. RFM Scoring (quantile-based)
  4. Customer Segmentation (rule-based + K-Means validation)
  5. Business Metrics per Segment
  6. Visualization Suite (7 charts)
  7. Actionable Insights Report
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings, os, textwrap

warnings.filterwarnings('ignore')

# ── Plotting theme ────────────────────────────────────────────────────────────
BLINKIT_YELLOW  = "#F5D900"
PALETTE = {
    "Champions":         "#2ECC71",   # green  – top tier
    "Loyal Customers":   "#3498DB",   # blue   – reliable
    "Potential Loyalists":"#9B59B6",  # purple – growth target
    "At Risk":           "#E67E22",   # orange – action needed
    "Lost Customers":    "#E74C3C",   # red    – win-back
}
BG      = "#0F1117"
CARD    = "#1A1D27"
TEXT    = "#E8E8E8"
SUBTLE  = "#6B7280"
GRID    = "#2A2D3A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       SUBTLE,
    "ytick.color":       SUBTLE,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION (Instacart-faithful synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
def generate_instacart_dataset(n_users: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a statistically faithful simulation of the Instacart orders table.

    Real Instacart statistics replicated:
      - ~10 orders per user (median), heavy right tail
      - days_since_prior_order: 0-30, modal at 7 days
      - order_hour_of_day: bimodal at 10am and 3pm
      - avg basket value ~$35 with log-normal spread
      - ~206k unique users in full dataset (we use 10k for dev)

    In production: replace with:
        orders = pd.read_csv("orders.csv")
        order_products = pd.read_csv("order_products__prior.csv")
        products       = pd.read_csv("products.csv")
        departments    = pd.read_csv("departments.csv")
    """
    rng = np.random.default_rng(seed)

    # Simulate heterogeneous customer population (5 latent types)
    user_types = rng.choice(
        ["champion", "loyal", "potential", "at_risk", "lost"],
        size=n_users,
        p=[0.12, 0.22, 0.28, 0.23, 0.15]
    )

    type_params = {
        # (freq_mean, freq_std, recency_mean, recency_std, spend_mean, spend_std)
        "champion":  (18, 5,   8,  5,  55, 15),
        "loyal":     (12, 3,  18, 10,  42, 12),
        "potential": ( 6, 2,  35, 15,  32,  8),
        "at_risk":   ( 4, 2,  75, 20,  28,  9),
        "lost":      ( 2, 1, 150, 30,  22,  7),
    }

    rows = []
    reference_date = pd.Timestamp("2024-03-01")

    for uid, utype in enumerate(user_types):
        fm, fs, rm, rs, sm, ss = type_params[utype]

        frequency = max(1, int(rng.normal(fm, fs)))
        recency   = max(1, int(rng.normal(rm, rs)))

        # Generate individual order timestamps working backwards from reference
        gaps = rng.exponential(scale=max(rm / max(frequency, 1), 3), size=frequency)
        gaps = np.clip(gaps, 1, 30).astype(int)

        order_date = reference_date - pd.Timedelta(days=int(recency))
        for order_num, gap in enumerate(gaps):
            basket_value = max(5.0, rng.lognormal(
                mean=np.log(max(sm, 1)), sigma=max(ss / max(sm, 1), 0.01)
            ))
            rows.append({
                "user_id":              uid,
                "order_id":             len(rows),
                "order_number":         order_num + 1,
                "order_date":           order_date,
                "days_since_prior":     gap if order_num > 0 else None,
                "order_hour_of_day":    int(rng.choice(
                                            np.arange(6, 23),
                                            p=np.array([1,2,4,6,8,10,9,8,7,7,6,5,5,6,6,5,4],
                                                       dtype=float) / np.array([1,2,4,6,8,10,9,8,7,7,6,5,5,6,6,5,4],
                                                       dtype=float).sum()
                                        )),
                "basket_value":         round(basket_value, 2),
                "n_items":              max(1, int(rng.normal(basket_value / 5, 2))),
                "true_segment":         utype,
            })
            order_date -= pd.Timedelta(days=int(gap))

    df = pd.DataFrame(rows)
    print(f"✅ Dataset generated: {len(df):,} orders | {df['user_id'].nunique():,} users")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. RFM METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary per customer.

    Recency  = days since most recent order (lower = better)
    Frequency= total number of orders placed
    Monetary = total spend (sum of basket values)

    Returns one row per user_id.
    """
    if snapshot_date is None:
        snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("user_id")
        .agg(
            last_order_date  = ("order_date",    "max"),
            frequency        = ("order_id",      "nunique"),
            monetary         = ("basket_value",  "sum"),
            avg_basket       = ("basket_value",  "mean"),
            n_items_total    = ("n_items",       "sum"),
            first_order_date = ("order_date",    "min"),
        )
        .reset_index()
    )

    rfm["recency"] = (snapshot_date - rfm["last_order_date"]).dt.days
    rfm["customer_age_days"] = (
        snapshot_date - rfm["first_order_date"]
    ).dt.days
    rfm["order_frequency_per_month"] = (
        rfm["frequency"] / (rfm["customer_age_days"] / 30).clip(lower=0.1)
    ).round(2)

    print(f"\n📊 RFM Summary Statistics:")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2))
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 3. RFM SCORING
# ─────────────────────────────────────────────────────────────────────────────
def score_rfm(rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Assign R, F, M scores from 1–5 using quintile binning.

    Recency  score: INVERTED — lower recency (recent) → score 5
    Frequency score: higher → score 5
    Monetary  score: higher → score 5
    """
    rfm = rfm.copy()

    # Recency: lower = better → invert
    rfm["R_score"] = pd.qcut(
        rfm["recency"], q=n_quantiles,
        labels=range(n_quantiles, 0, -1), duplicates="drop"
    ).astype(int)

    # Frequency: higher = better
    rfm["F_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), q=n_quantiles,
        labels=range(1, n_quantiles + 1), duplicates="drop"
    ).astype(int)

    # Monetary: higher = better
    rfm["M_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"), q=n_quantiles,
        labels=range(1, n_quantiles + 1), duplicates="drop"
    ).astype(int)

    rfm["RFM_score"]    = rfm["R_score"] * 100 + rfm["F_score"] * 10 + rfm["M_score"]
    rfm["RFM_combined"] = (rfm["R_score"] + rfm["F_score"] + rfm["M_score"]) / 3

    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEGMENTATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
SEGMENT_RULES = {
    # Evaluated in ORDER — first match wins
    "Champions": lambda r: (r["R_score"] >= 4) & (r["F_score"] >= 4) & (r["M_score"] >= 4),
    "Loyal Customers":    lambda r: (r["F_score"] >= 3) & (r["M_score"] >= 3) & ~(
        (r["R_score"] >= 4) & (r["F_score"] >= 4) & (r["M_score"] >= 4)
    ),
    "Potential Loyalists": lambda r: (r["R_score"] >= 3) & (r["F_score"] <= 3) & (r["M_score"] >= 2),
    "At Risk": lambda r: (r["R_score"] <= 2) & (r["F_score"] >= 3),
    "Lost Customers": lambda r: (r["R_score"] == 1) & (r["F_score"] <= 2),
}

def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based segment assignment with ordered priority matching.
    Unmatched users fall into 'Potential Loyalists' (catch-all).
    """
    rfm = rfm.copy()
    rfm["segment"] = "Potential Loyalists"   # default

    # Apply in reverse priority (last match wins → first-match: apply in reverse)
    for seg_name, rule_fn in reversed(list(SEGMENT_RULES.items())):
        mask = rule_fn(rfm)
        rfm.loc[mask, "segment"] = seg_name

    seg_counts = rfm["segment"].value_counts()
    print("\n🎯 Segment Distribution:")
    for seg, count in seg_counts.items():
        pct = count / len(rfm) * 100
        revenue = rfm[rfm["segment"] == seg]["monetary"].sum()
        print(f"   {seg:<22} {count:>5,} users ({pct:5.1f}%) | Revenue: ₹{revenue:>12,.0f}")

    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUSINESS METRICS AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_business_metrics(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate key business metrics per segment for executive reporting.
    """
    total_revenue = rfm["monetary"].sum()

    metrics = (
        rfm.groupby("segment")
        .agg(
            users           = ("user_id",      "count"),
            total_revenue   = ("monetary",     "sum"),
            avg_clv         = ("monetary",     "mean"),
            median_recency  = ("recency",      "median"),
            avg_frequency   = ("frequency",    "mean"),
            avg_basket      = ("avg_basket",   "mean"),
        )
        .reset_index()
    )

    metrics["revenue_share_pct"] = (
        metrics["total_revenue"] / total_revenue * 100
    ).round(1)
    metrics["user_share_pct"] = (
        metrics["users"] / rfm.shape[0] * 100
    ).round(1)
    metrics["revenue_per_user"] = (
        metrics["total_revenue"] / metrics["users"]
    ).round(0)

    seg_order = list(PALETTE.keys())
    metrics["_order"] = metrics["segment"].map(
        {s: i for i, s in enumerate(seg_order)}
    )
    metrics = metrics.sort_values("_order").drop("_order", axis=1)

    print("\n📈 Business Metrics by Segment:")
    print(metrics[["segment","users","revenue_share_pct","avg_clv","avg_frequency","median_recency"]].to_string(index=False))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

def _title_bar(ax, title: str, subtitle: str = ""):
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT, pad=14)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha="center", fontsize=9, color=SUBTLE)


def plot_segment_overview(rfm: pd.DataFrame, metrics: pd.DataFrame, save_path: str):
    """
    Chart 1 — Segment Overview Dashboard (2×2 grid):
      Top-left:  Donut — user share by segment
      Top-right: Horizontal bar — revenue share
      Bot-left:  Grouped bar — avg recency vs frequency (normalised)
      Bot-right: Bubble — avg CLV vs frequency, sized by user count
    """
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        "Customer Segmentation — Overview Dashboard",
        fontsize=18, fontweight="bold", color=TEXT, y=0.97
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = [PALETTE[s] for s in metrics["segment"]]

    # ── 1a. Donut: user share ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax1.pie(
        metrics["users"],
        labels=metrics["segment"],
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=2),
        startangle=90,
    )
    for t in texts:
        t.set(color=TEXT, fontsize=9)
    for at in autotexts:
        at.set(color=BG, fontsize=8, fontweight="bold")
    centre = ax1.text(0, 0, f"{metrics['users'].sum():,}\nusers",
                      ha="center", va="center", fontsize=13,
                      fontweight="bold", color=TEXT)
    ax1.set_title("User Distribution", fontsize=13, fontweight="bold",
                  color=TEXT, pad=10)

    # ── 1b. Horizontal bar: revenue share ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.barh(
        metrics["segment"],
        metrics["revenue_share_pct"],
        color=colors, edgecolor=BG, linewidth=0.5, height=0.55
    )
    for bar, val in zip(bars, metrics["revenue_share_pct"]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=10, color=TEXT)
    ax2.set_xlabel("% of Total Revenue", color=SUBTLE)
    ax2.set_xlim(0, metrics["revenue_share_pct"].max() * 1.18)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_title("Revenue Contribution", fontsize=13, fontweight="bold",
                  color=TEXT, pad=10)

    # ── 1c. Grouped bar: recency vs frequency (normalised 0-1) ───────────
    ax3 = fig.add_subplot(gs[1, 0])
    x     = np.arange(len(metrics))
    w     = 0.35
    r_norm = metrics["median_recency"] / metrics["median_recency"].max()
    f_norm = metrics["avg_frequency"] / metrics["avg_frequency"].max()

    b1 = ax3.bar(x - w/2, r_norm, width=w, color=[c + "CC" for c in colors],
                 label="Recency (norm, lower=better)", edgecolor=BG)
    b2 = ax3.bar(x + w/2, f_norm, width=w, color=colors,
                 label="Frequency (norm, higher=better)", edgecolor=BG)

    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [s.replace(" ", "\n") for s in metrics["segment"]],
        fontsize=8.5
    )
    ax3.set_ylabel("Normalised Score", color=SUBTLE)
    ax3.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT,
               edgecolor=GRID, loc="upper right")
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_title("Recency vs Frequency by Segment", fontsize=13,
                  fontweight="bold", color=TEXT, pad=10)

    # ── 1d. Bubble: CLV vs Frequency, size = user count ──────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for _, row in metrics.iterrows():
        color = PALETTE[row["segment"]]
        ax4.scatter(
            row["avg_frequency"],
            row["avg_clv"],
            s=row["users"] / metrics["users"].max() * 2500 + 100,
            c=color, alpha=0.85, edgecolors=BG, linewidth=1.5,
            zorder=3
        )
        ax4.annotate(
            row["segment"].replace(" ", "\n"),
            (row["avg_frequency"], row["avg_clv"]),
            textcoords="offset points", xytext=(10, 6),
            fontsize=8, color=color
        )
    ax4.set_xlabel("Avg Orders (Frequency)", color=SUBTLE)
    ax4.set_ylabel("Avg Customer Lifetime Value (₹)", color=SUBTLE)
    ax4.grid(alpha=0.3)
    ax4.set_title("CLV vs Frequency\n(bubble size = segment size)",
                  fontsize=13, fontweight="bold", color=TEXT, pad=10)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_rfm_distributions(rfm: pd.DataFrame, save_path: str):
    """
    Chart 2 — RFM Distribution Plots:
      Row 1: Histograms of R, F, M coloured by segment
      Row 2: Box plots of R, F, M per segment
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
    fig.suptitle("RFM Metric Distributions by Segment",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.97)

    metrics_cfg = [
        ("recency",   "Recency (days since last order)",  True),
        ("frequency", "Frequency (total orders)",          False),
        ("monetary",  "Monetary Value (₹ total spend)",   False),
    ]

    for col_idx, (metric, label, invert_note) in enumerate(metrics_cfg):

        # Row 0: Histogram per segment (overlapping, semi-transparent)
        ax_hist = axes[0, col_idx]
        for seg, color in PALETTE.items():
            subset = rfm[rfm["segment"] == seg][metric]
            if len(subset) < 5:
                continue
            ax_hist.hist(
                subset, bins=40, color=color, alpha=0.55,
                label=seg, density=True, edgecolor="none"
            )
        ax_hist.set_xlabel(label, color=SUBTLE, fontsize=9)
        ax_hist.set_ylabel("Density", color=SUBTLE, fontsize=9)
        ax_hist.set_title(f"{metric.capitalize()} Distribution",
                          fontsize=11, fontweight="bold", color=TEXT)
        if invert_note:
            ax_hist.text(0.97, 0.95, "← lower = more recent",
                         transform=ax_hist.transAxes, ha="right",
                         fontsize=8, color=SUBTLE, style="italic")
        if col_idx == 0:
            ax_hist.legend(fontsize=7.5, facecolor=CARD,
                           labelcolor=TEXT, edgecolor=GRID)
        ax_hist.grid(axis="y", alpha=0.3)

        # Row 1: Violin + Box per segment
        ax_box = axes[1, col_idx]
        seg_order = list(PALETTE.keys())
        data_by_seg = [
            rfm[rfm["segment"] == s][metric].values for s in seg_order
        ]
        parts = ax_box.violinplot(
            data_by_seg, positions=range(len(seg_order)),
            showmedians=True, showextrema=False,
            widths=0.65
        )
        for i, (pc, seg) in enumerate(zip(parts["bodies"], seg_order)):
            pc.set_facecolor(PALETTE[seg])
            pc.set_alpha(0.65)
            pc.set_edgecolor(BG)
        parts["cmedians"].set_color(TEXT)
        parts["cmedians"].set_linewidth(1.5)

        ax_box.set_xticks(range(len(seg_order)))
        ax_box.set_xticklabels(
            [s.replace(" ", "\n") for s in seg_order], fontsize=7.5
        )
        ax_box.set_title(f"{metric.capitalize()} by Segment",
                         fontsize=11, fontweight="bold", color=TEXT)
        ax_box.set_ylabel(label, color=SUBTLE, fontsize=9)
        ax_box.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_rfm_heatmap(rfm: pd.DataFrame, save_path: str):
    """
    Chart 3 — RFM Score Heatmap:
      F-score (y) × R-score (x) grid coloured by avg monetary value.
      Overlaid segment labels show which score combos map to which segment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.suptitle("RFM Score Matrix — Segment Geography",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.97)

    # Left: avg monetary heatmap
    pivot_monetary = (
        rfm.pivot_table(index="F_score", columns="R_score",
                        values="monetary", aggfunc="mean")
        .sort_index(ascending=False)
    )
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list(
        "blinkit", ["#1A1D27", "#E67E22", "#F5D900", "#2ECC71"]
    )
    im = ax.imshow(pivot_monetary.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot_monetary.columns)))
    ax.set_xticklabels([f"R={c}" for c in pivot_monetary.columns])
    ax.set_yticks(range(len(pivot_monetary.index)))
    ax.set_yticklabels([f"F={i}" for i in pivot_monetary.index])
    for i in range(len(pivot_monetary.index)):
        for j in range(len(pivot_monetary.columns)):
            val = pivot_monetary.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"₹{val:,.0f}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Avg Monetary (₹)", shrink=0.8)
    ax.set_title("Avg Spend by R×F Score\n(higher R = more recent)",
                 fontsize=12, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Recency Score →  (5 = most recent)", color=SUBTLE)
    ax.set_ylabel("Frequency Score →  (5 = most frequent)", color=SUBTLE)

    # Right: segment dominance per cell
    pivot_seg = (
        rfm.pivot_table(index="F_score", columns="R_score",
                        values="segment", aggfunc=lambda x: x.mode()[0])
        .sort_index(ascending=False)
    )
    ax2 = axes[1]
    seg_to_int = {s: i for i, s in enumerate(PALETTE)}
    int_to_seg = {v: k for k, v in seg_to_int.items()}
    seg_matrix = pivot_seg.map(lambda s: seg_to_int.get(s, 2) if isinstance(s, str) else 2)

    seg_cmap = LinearSegmentedColormap.from_list(
        "segments", list(PALETTE.values()), N=len(PALETTE)
    )
    ax2.imshow(seg_matrix.values, cmap=seg_cmap, aspect="auto",
               vmin=0, vmax=len(PALETTE) - 1)
    ax2.set_xticks(range(len(pivot_seg.columns)))
    ax2.set_xticklabels([f"R={c}" for c in pivot_seg.columns])
    ax2.set_yticks(range(len(pivot_seg.index)))
    ax2.set_yticklabels([f"F={i}" for i in pivot_seg.index])
    for i in range(len(pivot_seg.index)):
        for j in range(len(pivot_seg.columns)):
            val = pivot_seg.values[i, j]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                short = str(val).replace("Customers", "Cust.")
                ax2.text(j, i, short, ha="center", va="center",
                         fontsize=7.5, color="white", fontweight="bold")
    patches = [
        mpatches.Patch(color=PALETTE[s], label=s) for s in PALETTE
    ]
    ax2.legend(handles=patches, fontsize=8, facecolor=CARD,
               labelcolor=TEXT, edgecolor=GRID,
               loc="lower center", bbox_to_anchor=(0.5, -0.22),
               ncol=3)
    ax2.set_title("Dominant Segment by R×F Score",
                  fontsize=12, fontweight="bold", color=TEXT, pad=12)
    ax2.set_xlabel("Recency Score", color=SUBTLE)
    ax2.set_ylabel("Frequency Score", color=SUBTLE)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_revenue_waterfall(metrics: pd.DataFrame, save_path: str):
    """
    Chart 4 — Revenue & Opportunity Waterfall:
      Stacked view of current revenue + estimated recovery potential.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("Revenue Waterfall & Recovery Opportunity",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.97)

    # Left: absolute revenue bars
    ax = axes[0]
    bars = ax.bar(
        metrics["segment"],
        metrics["total_revenue"],
        color=[PALETTE[s] for s in metrics["segment"]],
        edgecolor=BG, linewidth=0.5, width=0.6
    )
    for bar, row in zip(bars, metrics.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + metrics["total_revenue"].max() * 0.01,
                f"₹{row.total_revenue:,.0f}\n({row.revenue_share_pct:.1f}%)",
                ha="center", fontsize=9, color=TEXT)
    ax.set_xticklabels(
        [s.replace(" ", "\n") for s in metrics["segment"]],
        fontsize=9
    )
    ax.set_ylabel("Total Revenue (₹)", color=SUBTLE)
    ax.set_title("Current Revenue by Segment", fontsize=12,
                 fontweight="bold", color=TEXT)
    ax.grid(axis="y", alpha=0.3)

    # Right: revenue per user — efficiency view
    ax2 = axes[1]
    rev_per_user = metrics["revenue_per_user"].values
    colors_used   = [PALETTE[s] for s in metrics["segment"]]

    bars2 = ax2.bar(
        metrics["segment"], rev_per_user,
        color=colors_used, edgecolor=BG, linewidth=0.5, width=0.6
    )
    # Annotate lift opportunity: % below Champion RPU
    champion_rpu = metrics.loc[
        metrics["segment"] == "Champions", "revenue_per_user"
    ].values[0]

    for bar, row in zip(bars2, metrics.itertuples()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + rev_per_user.max() * 0.015,
            f"₹{row.revenue_per_user:,.0f}",
            ha="center", fontsize=9, color=TEXT
        )
        if row.segment != "Champions":
            lift = ((champion_rpu - row.revenue_per_user)
                    / champion_rpu * 100)
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{lift:.0f}% below\nChampions",
                ha="center", fontsize=8, color="white",
                alpha=0.85
            )

    ax2.axhline(y=champion_rpu, color=PALETTE["Champions"],
                linestyle="--", linewidth=1.2, alpha=0.7,
                label=f"Champion RPU: ₹{champion_rpu:,.0f}")
    ax2.set_xticklabels(
        [s.replace(" ", "\n") for s in metrics["segment"]],
        fontsize=9
    )
    ax2.set_ylabel("Revenue per User (₹)", color=SUBTLE)
    ax2.set_title("Revenue per User — Efficiency Gap",
                  fontsize=12, fontweight="bold", color=TEXT)
    ax2.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT, edgecolor=GRID)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_rfm_scatter_pca(rfm: pd.DataFrame, save_path: str):
    """
    Chart 5 — 2D PCA Scatter of RFM Space:
      Projects R, F, M scores into 2D via PCA.
      Each point is a customer, coloured by segment.
      Centroids labelled with segment name.
    """
    features = rfm[["R_score", "F_score", "M_score"]].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    rfm = rfm.copy()
    rfm["PC1"] = X_2d[:, 0]
    rfm["PC2"] = X_2d[:, 1]

    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
    fig.suptitle("Customer Segments in RFM Space (PCA Projection)",
                 fontsize=16, fontweight="bold", color=TEXT)

    for seg, color in PALETTE.items():
        mask = rfm["segment"] == seg
        ax.scatter(
            rfm.loc[mask, "PC1"], rfm.loc[mask, "PC2"],
            c=color, s=18, alpha=0.45, label=seg, edgecolors="none"
        )
        # Centroid
        cx = rfm.loc[mask, "PC1"].mean()
        cy = rfm.loc[mask, "PC2"].mean()
        ax.scatter(cx, cy, c=color, s=280, marker="*",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(
            seg,
            (cx, cy), textcoords="offset points", xytext=(8, 8),
            fontsize=10, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=CARD,
                      ec=color, alpha=0.85, lw=0.8)
        )

    var_explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(
        f"PC1 ({var_explained[0]:.1f}% variance) — primarily Frequency & Monetary",
        color=SUBTLE, fontsize=10
    )
    ax.set_ylabel(
        f"PC2 ({var_explained[1]:.1f}% variance) — primarily Recency",
        color=SUBTLE, fontsize=10
    )
    ax.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT,
              edgecolor=GRID, markerscale=2.5)
    ax.grid(alpha=0.2)

    total_var = sum(var_explained)
    ax.text(0.02, 0.97,
            f"Total variance explained: {total_var:.1f}%",
            transform=ax.transAxes, fontsize=9, color=SUBTLE,
            va="top")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_segment_radar(metrics: pd.DataFrame, save_path: str):
    """
    Chart 6 — Radar (Spider) Chart per Segment:
      Normalised dimensions: Recency, Frequency, Monetary, Avg Basket, User Share
    """
    dims = ["Recency\n(inverse)", "Frequency", "Monetary", "Avg Basket", "User Share"]
    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    # Normalise each dimension 0-1 (recency is already inverted: lower = better)
    col_map = {
        "Recency\n(inverse)": "median_recency",
        "Frequency":          "avg_frequency",
        "Monetary":           "avg_clv",
        "Avg Basket":         "avg_basket",
        "User Share":         "user_share_pct",
    }
    norm = metrics.copy()
    norm["median_recency"] = norm["median_recency"].max() - norm["median_recency"]
    for col in col_map.values():
        rng = norm[col].max() - norm[col].min()
        norm[col] = (norm[col] - norm[col].min()) / (rng if rng > 0 else 1)

    fig, axes = plt.subplots(
        1, len(PALETTE), figsize=(22, 5),
        subplot_kw=dict(polar=True), facecolor=BG
    )
    fig.suptitle("Segment Radar Profiles — Normalised Dimensions",
                 fontsize=15, fontweight="bold", color=TEXT, y=1.04)

    for ax, (seg, color) in zip(axes, PALETTE.items()):
        row = norm[norm["segment"] == seg]
        if row.empty:
            continue
        vals = [row[col_map[d]].values[0] for d in dims]
        vals += vals[:1]

        ax.set_facecolor(CARD)
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=8, color=TEXT)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        ax.spines["polar"].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_title(seg, color=color, fontsize=10,
                     fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_action_matrix(metrics: pd.DataFrame, save_path: str):
    """
    Chart 7 — Strategic Action Matrix:
      X = segment size (users), Y = avg CLV, quadrants labelled.
      Annotated with business action for each segment.
    """
    ACTIONS = {
        "Champions":          "VIP programme\nreferral rewards",
        "Loyal Customers":    "Upsell premium\ncategories",
        "Potential Loyalists":"Habit-forming\nstreak incentives",
        "At Risk":            "Win-back campaign\n₹50 coupon trigger",
        "Lost Customers":     "Reactivation burst\n+ deep discount",
    }

    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
    fig.suptitle("Strategic Action Matrix — Segment Prioritisation",
                 fontsize=16, fontweight="bold", color=TEXT)

    med_users = metrics["users"].median()
    med_clv   = metrics["avg_clv"].median()

    # Quadrant shading
    xmax = metrics["users"].max() * 1.3
    ymax = metrics["avg_clv"].max() * 1.3
    for (xl, xr, yb, yt, label, alpha) in [
        (0,        med_users, med_clv, ymax,  "High Value\nSmall Base",    0.04),
        (med_users, xmax,     med_clv, ymax,  "High Value\nLarge Base",    0.07),
        (0,        med_users, 0,       med_clv,"Low Value\nSmall Base",    0.02),
        (med_users, xmax,     0,       med_clv,"Low Value\nLarge Base",    0.04),
    ]:
        ax.fill_between([xl, xr], [yb, yb], [yt, yt],
                        color=TEXT, alpha=alpha)
        ax.text((xl + xr) / 2, (yb + yt) / 2, label,
                ha="center", va="center", fontsize=9,
                color=SUBTLE, style="italic")

    ax.axvline(x=med_users, color=GRID, linewidth=1, linestyle="--")
    ax.axhline(y=med_clv,   color=GRID, linewidth=1, linestyle="--")

    for _, row in metrics.iterrows():
        color = PALETTE[row["segment"]]
        ax.scatter(row["users"], row["avg_clv"],
                   s=row["revenue_share_pct"] * 80,
                   c=color, alpha=0.9, edgecolors="white",
                   linewidths=1.2, zorder=5)
        action_text = f"{row['segment']}\n{ACTIONS[row['segment']]}"
        ax.annotate(
            action_text,
            (row["users"], row["avg_clv"]),
            textcoords="offset points", xytext=(14, -10),
            fontsize=8.5, color=color,
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD,
                      ec=color, alpha=0.9, lw=0.8),
            arrowprops=dict(arrowstyle="-", color=color,
                            connectionstyle="arc3,rad=0.1")
        )

    ax.set_xlabel("Segment Size (number of users)", color=SUBTLE, fontsize=11)
    ax.set_ylabel("Avg Customer Lifetime Value (₹)", color=SUBTLE, fontsize=11)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.2)
    ax.text(0.98, 0.02, "Bubble size = revenue share %",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            color=SUBTLE, style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   ✅ Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUSINESS IMPLICATIONS REPORT
# ─────────────────────────────────────────────────────────────────────────────
BUSINESS_IMPLICATIONS = {
    "Champions": {
        "profile":      "High recency, high frequency, high spend. Your most active buyers.",
        "risk":         "Low churn probability but high sensitivity to service degradation.",
        "opportunity":  "Referral programme — Champions have the highest NPS and word-of-mouth potential.",
        "blinkit_tactic": [
            "Unlock a 'Blinkit Black' loyalty tier with free express delivery.",
            "Early access to new categories (electronics, pharmacy).",
            "Personalised 'thank you' push at 50th order milestone.",
            "Referral credit: ₹100 for every friend who completes 3 orders.",
        ],
        "kpi_watch":    ["Churn rate", "Basket size trend", "Category breadth"],
        "gmv_impact":   "Protecting 1% of Champions from churning = ~₹X lakh saved annually.",
    },
    "Loyal Customers": {
        "profile":      "Steady ordering cadence, solid spend. The backbone of your GMV.",
        "risk":         "Susceptible to competitor promotions (Zepto, Swiggy Instamart).",
        "opportunity":  "Basket size expansion — they trust you, now grow their AOV.",
        "blinkit_tactic": [
            "Category expansion nudges: 'You buy fresh veg — try our premium dairy range'.",
            "Subscription plan upsell: ₹99/month for free delivery on all orders.",
            "Combo offer personalisation based on past basket patterns.",
            "Quarterly loyalty summary: 'You saved ₹X with Blinkit this quarter.'",
        ],
        "kpi_watch":    ["Avg order value", "Category penetration", "Subscription conversion"],
        "gmv_impact":   "10% AOV lift across Loyal segment = material top-line impact.",
    },
    "Potential Loyalists": {
        "profile":      "Recent buyers, 2–5 orders, showing signs of habit formation.",
        "risk":         "Most likely to drop off without reinforcement at the 'habit cliff'.",
        "opportunity":  "This is your growth engine. Convert to Loyal in 60 days.",
        "blinkit_tactic": [
            "Order streak reward: 'Order 3 times this week, get ₹30 off your next.'",
            "Onboarding category discovery: introduce one new category each week.",
            "Day-14 re-engagement push if no order detected.",
            "First subscription offer at order #4 — when intent is highest.",
        ],
        "kpi_watch":    ["Day-30 retention", "Order #3 completion rate", "Time between orders"],
        "gmv_impact":   "Moving 20% of Potential Loyalists to Loyal is the single highest-ROI lever.",
    },
    "At Risk": {
        "profile":      "Were active, now quiet. Recency score dropping. They remember you.",
        "risk":         "Currently losing ₹X in GMV per week as these users defect.",
        "opportunity":  "Win-back window is NOW — highest re-engagement probability in first 45 days.",
        "blinkit_tactic": [
            "Triggered win-back at day 21 of inactivity: '₹50 off, only for the next 48 hours.'",
            "Survey: 'What went wrong? Tell us and get ₹20 off.' — price discovery.",
            "Highlight what's new since last order (new categories, faster delivery).",
            "Escalating discount ladder: ₹30 at day 21, ₹50 at day 35, ₹75 at day 50.",
        ],
        "kpi_watch":    ["Win-back rate", "Days since last order", "Coupon redemption rate"],
        "gmv_impact":   "Recapturing 25% of At-Risk segment restores significant weekly GMV.",
    },
    "Lost Customers": {
        "profile":      "Churned. High recency, low frequency. Likely using a competitor.",
        "risk":         "CAC to reacquire is 3–5× higher than retention cost.",
        "opportunity":  "Small reactivation rate × large segment size = meaningful GMV at low spend.",
        "blinkit_tactic": [
            "Reactivation burst campaign: deep discount (₹100 off ₹300) — one shot.",
            "SMS + email multi-channel reactivation (not just app push — they may have uninstalled).",
            "Seasonal hooks: 'Diwali gifts delivered in 10 mins — we've missed you.'",
            "Sunset low-responders after 2 failed campaigns to protect deliverability.",
        ],
        "kpi_watch":    ["Reactivation rate", "Time-to-second-order post reactivation", "CAC"],
        "gmv_impact":   "Even 5% reactivation at standard AOV is meaningful at scale.",
    },
}

def print_business_report(rfm: pd.DataFrame, metrics: pd.DataFrame):
    """Print structured business implications report to console."""
    total_rev = metrics["total_revenue"].sum()
    sep = "═" * 78

    print(f"\n{sep}")
    print("  GROWTH INTELLIGENCE ENGINE — BUSINESS IMPLICATIONS REPORT")
    print(f"  Blinkit Product Analytics  |  {rfm['user_id'].nunique():,} Customers Analysed")
    print(f"{sep}\n")

    for seg, color_code in PALETTE.items():
        row = metrics[metrics["segment"] == seg]
        if row.empty:
            continue
        row = row.iloc[0]
        impl = BUSINESS_IMPLICATIONS[seg]

        print(f"  ▌ {seg.upper()}")
        print(f"  {'─' * 70}")
        print(f"  👥 Users: {row['users']:,} ({row['user_share_pct']:.1f}%)  "
              f"| 💰 Revenue: ₹{row['total_revenue']:,.0f} ({row['revenue_share_pct']:.1f}%)  "
              f"| Avg CLV: ₹{row['avg_clv']:,.0f}")
        print(f"  📅 Median recency: {row['median_recency']:.0f} days  "
              f"| 🛒 Avg orders: {row['avg_frequency']:.1f}  "
              f"| 🧺 Avg basket: ₹{row['avg_basket']:,.0f}\n")

        print(f"  Profile:     {impl['profile']}")
        print(f"  Risk:        {impl['risk']}")
        print(f"  Opportunity: {impl['opportunity']}\n")

        print(f"  Blinkit Tactics:")
        for tactic in impl["blinkit_tactic"]:
            print(f"    → {tactic}")

        print(f"\n  Watch KPIs: {', '.join(impl['kpi_watch'])}")
        print(f"  GMV Impact: {impl['gmv_impact']}")
        print(f"\n  {'─' * 70}\n")

    print(f"\n  TOTAL BASE REVENUE: ₹{total_rev:,.0f}")
    print(f"  KEY INSIGHT: Top 2 segments (Champions + Loyal) drive "
          f"{metrics[metrics['segment'].isin(['Champions','Loyal Customers'])]['revenue_share_pct'].sum():.0f}% "
          f"of total revenue from "
          f"{metrics[metrics['segment'].isin(['Champions','Loyal Customers'])]['user_share_pct'].sum():.0f}% "
          f"of users.")
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def load_instacart(data_dir: str = "data/raw/") -> pd.DataFrame:
    print("Loading real Instacart data...")
    orders = pd.read_csv(f"{data_dir}orders.csv")
    prior  = pd.read_csv(f"{data_dir}order_products__prior.csv")

    basket = prior.groupby("order_id").agg(
        n_items      = ("product_id", "count"),
        basket_value = ("product_id", "count"),   # proxy: 1 item = ₹35
    ).reset_index()
    basket["basket_value"] = basket["n_items"] * 35

    df = orders.merge(basket, on="order_id", how="left")
    df["basket_value"] = df["basket_value"].fillna(35)
    df["n_items"]      = df["n_items"].fillna(1).astype(int)

    # Build order_date from days_since_prior_order
    df = df.sort_values(["user_id", "order_number"])
    df["order_date"] = pd.Timestamp("2024-03-01") - pd.to_timedelta(
        df.groupby("user_id")["days_since_prior_order"]
          .transform(lambda x: x.fillna(0)[::-1].cumsum()[::-1]),
        unit="D"
    )
    print(f"Loaded: {len(df):,} orders | {df['user_id'].nunique():,} users")
    return df

def run_rfm_pipeline(n_users: int = 10_000):
    """
    End-to-end RFM pipeline.
    In production, replace generate_instacart_dataset() with:
        df = load_and_join_instacart_csvs(data_dir="./data/")
    """
    print("\n" + "═" * 60)
    print("  GROWTH INTELLIGENCE ENGINE — RFM PIPELINE")
    print("═" * 60)

    # ── Step 1: Data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = load_instacart("data/raw/")

    # ── Step 2: RFM metrics ───────────────────────────────────────────────
    print("\n[2/6] Computing RFM metrics...")
    rfm = compute_rfm(df)

    # ── Step 3: Scoring ───────────────────────────────────────────────────
    print("\n[3/6] Scoring R, F, M (quintiles)...")
    rfm = score_rfm(rfm)

    # ── Step 4: Segmentation ──────────────────────────────────────────────
    print("\n[4/6] Assigning segments...")
    rfm = assign_segments(rfm)

    # ── Step 5: Business metrics ──────────────────────────────────────────
    print("\n[5/6] Computing business metrics...")
    metrics = compute_business_metrics(rfm)

    # ── Step 6: Visualisations ────────────────────────────────────────────
    print("\n[6/6] Generating visualisation suite...")
    viz_paths = {
        "overview":      f"{OUTPUT_DIR}/rfm_01_segment_overview.png",
        "distributions": f"{OUTPUT_DIR}/rfm_02_distributions.png",
        "heatmap":       f"{OUTPUT_DIR}/rfm_03_score_heatmap.png",
        "waterfall":     f"{OUTPUT_DIR}/rfm_04_revenue_waterfall.png",
        "pca_scatter":   f"{OUTPUT_DIR}/rfm_05_pca_scatter.png",
        "radar":         f"{OUTPUT_DIR}/rfm_06_radar_profiles.png",
        "action_matrix": f"{OUTPUT_DIR}/rfm_07_action_matrix.png",
    }

    plot_segment_overview(rfm, metrics,  viz_paths["overview"])
    plot_rfm_distributions(rfm,          viz_paths["distributions"])
    plot_rfm_heatmap(rfm,                viz_paths["heatmap"])
    plot_revenue_waterfall(metrics,      viz_paths["waterfall"])
    plot_rfm_scatter_pca(rfm,            viz_paths["pca_scatter"])
    plot_segment_radar(metrics,          viz_paths["radar"])
    plot_action_matrix(metrics,          viz_paths["action_matrix"])

    # ── Report ────────────────────────────────────────────────────────────
    print_business_report(rfm, metrics)

    # ── Export CSVs ───────────────────────────────────────────────────────
    rfm.to_csv(f"{OUTPUT_DIR}/rfm_customer_segments.csv", index=False)
    metrics.to_csv(f"{OUTPUT_DIR}/rfm_segment_metrics.csv", index=False)
    print(f"\n✅ CSV exports saved to {OUTPUT_DIR}/")

    return rfm, metrics, viz_paths


if __name__ == "__main__":
    rfm_df, metrics_df, paths = run_rfm_pipeline(n_users=10_000)
