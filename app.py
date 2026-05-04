import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SVM Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Navy Blue Theme CSS ─────────────────────────────────────────────────────
# Navy palette:
#   bg-deep:    #0a0f1e   (darkest navy)
#   bg-card:    #0d1b2a   (card background)
#   bg-mid:     #112240   (mid-tone navy)
#   accent:     #4fc3f7   (light blue accent)
#   accent2:    #00e5ff   (cyan highlight)
#   text:       #ccd6f6
#   muted:      #8892b0

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"], .stApp {
    background-color: #0a0f1e !important;
    font-family: 'DM Sans', sans-serif;
    color: #ccd6f6;
}

h1, h2, h3, h4 {
    font-family: 'JetBrains Mono', monospace !important;
    color: #4fc3f7 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1b2a !important;
    border-right: 1px solid #1e3a5f;
}

[data-testid="stSidebar"] * {
    color: #ccd6f6 !important;
}

/* ── Main content area ── */
[data-testid="stAppViewContainer"] > .main {
    background-color: #0a0f1e !important;
}

[data-testid="block-container"] {
    background-color: #0a0f1e !important;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0a1628 100%);
    padding: 2.2rem 2rem;
    border-radius: 14px;
    margin-bottom: 2rem;
    border: 1px solid #1e3a5f;
    text-align: center;
    box-shadow: 0 4px 30px #00e5ff0d;
}

.main-header h1 {
    color: #4fc3f7 !important;
    font-size: 2.4rem;
    margin: 0;
    letter-spacing: -1px;
    text-shadow: 0 0 40px #4fc3f766;
}

.main-header p {
    color: #8892b0;
    margin-top: 0.6rem;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
}

/* ── Concept Cards ── */
.concept-card {
    background: linear-gradient(135deg, #0d1b2a, #112240);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.4rem;
    margin: 0.8rem 0;
    color: #ccd6f6;
}

.concept-card h3 {
    color: #4fc3f7 !important;
    margin-top: 0;
    font-size: 1rem;
}

.concept-card ul {
    padding-left: 1.2rem;
    margin: 0.5rem 0 0 0;
    color: #a8b2d8;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ── Highlight bar ── */
.highlight {
    background: linear-gradient(90deg, #4fc3f711, transparent);
    border-left: 3px solid #4fc3f7;
    padding: 0.5rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.7rem 0;
    color: #ccd6f6;
    font-size: 0.9rem;
}

/* ── Metric boxes ── */
.metric-box {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}

.metric-box .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #4fc3f7;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}

.metric-box .label {
    color: #8892b0;
    font-size: 0.8rem;
    margin-top: 0.2rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: #0d1b2a;
    border-radius: 8px;
    color: #8892b0;
    border: 1px solid #1e3a5f;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    padding: 0.5rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #112240, #0d1b2a) !important;
    color: #4fc3f7 !important;
    border-color: #4fc3f766 !important;
}

/* ── Sliders & Inputs ── */
.stSlider > div > div > div {
    background: #4fc3f7 !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px;
}

/* ── Info / Success ── */
.stAlert {
    background: #112240 !important;
    border: 1px solid #1e3a5f !important;
    color: #ccd6f6 !important;
    border-radius: 8px;
}

/* ── Selectbox / Radio ── */
[data-testid="stSelectbox"] > div,
[data-testid="stRadio"] > div {
    background: #0d1b2a !important;
}

div[data-baseweb="select"] > div {
    background: #112240 !important;
    border-color: #1e3a5f !important;
    color: #ccd6f6 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 SVM Explorer</h1>
    <p>Support Vector Machine — Interactive Learning &amp; Visualization Tool</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    st.markdown("---")

    kernel = st.selectbox(
        "🔧 Kernel",
        ["linear", "rbf", "poly", "sigmoid"],
        help="Kernel function used by the SVM"
    )

    C = st.slider(
        "📏 C (Regularization)",
        min_value=0.01, max_value=20.0,
        value=1.0, step=0.01,
        help="Larger C = harder margin, more sensitive to outliers"
    )

    if kernel == "rbf":
        gamma_mode = st.radio("Gamma", ["scale", "auto", "custom"])
        if gamma_mode == "custom":
            gamma = st.slider("γ Value", 0.001, 5.0, 0.5, 0.001)
        else:
            gamma = gamma_mode
    elif kernel == "poly":
        degree = st.slider("🔢 Degree", 2, 6, 3)
        gamma = "scale"
    else:
        gamma = "scale"
        degree = 3

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    dataset_name = st.selectbox(
        "Dataset",
        ["Classification (2 classes)", "Iris (2 features)", "Moons", "Circles", "Blobs"]
    )

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42)

# ─── Data Generation ────────────────────────────────────────────────────────
@st.cache_data
def get_dataset(name, random_state=42):
    rs = int(random_state)
    if name == "Classification (2 classes)":
        X, y = datasets.make_classification(
            n_samples=200, n_features=2, n_redundant=0,
            n_informative=2, random_state=rs, n_clusters_per_class=1
        )
    elif name == "Iris (2 features)":
        iris = datasets.load_iris()
        X = iris.data[:100, :2]
        y = iris.target[:100]
    elif name == "Moons":
        X, y = datasets.make_moons(n_samples=200, noise=0.2, random_state=rs)
    elif name == "Circles":
        X, y = datasets.make_circles(n_samples=200, noise=0.15, factor=0.4, random_state=rs)
    else:  # Blobs
        X, y = datasets.make_blobs(n_samples=200, centers=2, random_state=rs)
    return X, y

# Navy matplotlib style helper
def style_ax(ax, fig=None):
    if fig:
        fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0d1b2a')
    ax.tick_params(colors='#8892b0', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.grid(True, alpha=0.12, color='#4fc3f7')

X, y = get_dataset(dataset_name, random_state)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=int(random_state)
)

# ─── Train Model ────────────────────────────────────────────────────────────
try:
    if kernel == "poly":
        clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=True)
    else:
        clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    n_support = clf.n_support_
    model_trained = True
except Exception as e:
    st.error(f"Model training error: {e}")
    model_trained = False

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 What is SVM?",
    "🎯 Decision Boundary",
    "🔬 Kernel Comparison",
    "📊 Model Evaluation"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Concepts
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## What is SVM?")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
<div class="concept-card">
<h3>🎯 Core Idea</h3>
<p>SVM (Support Vector Machine) is a classification algorithm that finds the <strong>optimal separating hyperplane</strong> between two classes by maximizing the margin between them.</p>
<br>
<div class="highlight">Goal: Maximize the margin between classes</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
<h3>📌 Support Vectors</h3>
<p>The data points closest to the decision boundary that define the margin. Removing all other points has no effect on the model!</p>
<br>
<div class="highlight">SVM only cares about the hardest-to-classify points</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
<h3>🌀 The Kernel Trick</h3>
<p>When data isn't linearly separable, the kernel function maps data to a higher-dimensional space where a linear separator exists.</p>
<br>
<div class="highlight">Like folding paper to separate points with a straight cut!</div>
</div>
""", unsafe_allow_html=True)

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.patch.set_facecolor('#0a0f1e')

        for ax in axes:
            style_ax(ax)

        # Left: Margin concept
        ax = axes[0]
        np.random.seed(42)
        X_pos = np.random.randn(15, 2) + [2, 2]
        X_neg = np.random.randn(15, 2) + [-2, -2]
        ax.scatter(X_pos[:, 0], X_pos[:, 1], c='#4fc3f7', s=60, zorder=5, label='Class +1')
        ax.scatter(X_neg[:, 0], X_neg[:, 1], c='#ef5350', s=60, zorder=5, label='Class -1')
        x_line = np.linspace(-5, 5, 100)
        ax.plot(x_line, x_line * 0, color='#00e5ff', lw=2.5, label='Hyperplane')
        ax.fill_between(x_line, -0.8, 0.8, alpha=0.12, color='#00e5ff', label='Margin')
        ax.axhline(y=0.8, color='#00e5ff', lw=1, ls='--', alpha=0.6)
        ax.axhline(y=-0.8, color='#00e5ff', lw=1, ls='--', alpha=0.6)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_title('Margin & Hyperplane', color='#4fc3f7', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')

        # Right: Kernel trick
        ax = axes[1]
        theta = np.linspace(0, 2*np.pi, 50)
        r_in, r_out = 0.8, 1.8
        X_in = np.column_stack([r_in*np.cos(theta[:25]), r_in*np.sin(theta[:25])])
        X_out = np.column_stack([r_out*np.cos(theta), r_out*np.sin(theta)])
        ax.scatter(X_in[:, 0], X_in[:, 1], c='#4fc3f7', s=60, zorder=5, label='Class +1')
        ax.scatter(X_out[:, 0], X_out[:, 1], c='#ef5350', s=60, zorder=5, label='Class -1')
        circle = plt.Circle((0, 0), 1.3, fill=False, color='#00e5ff', lw=2.5, label='Decision Boundary')
        ax.add_patch(circle)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title('Kernel Trick (RBF)', color='#4fc3f7', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("## 🔧 Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class="concept-card">
<h3>C — Regularization</h3>
<ul>
<li>Small C → Wide margin, tolerates misclassifications</li>
<li>Large C → Narrow margin, penalizes errors heavily</li>
<li>Very large C → Risk of overfitting</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="concept-card">
<h3>Kernel Types</h3>
<ul>
<li><strong>Linear</strong>: Linearly separable data</li>
<li><strong>RBF</strong>: Most versatile, general use</li>
<li><strong>Poly</strong>: Polynomial decision boundaries</li>
<li><strong>Sigmoid</strong>: Similar to neural networks</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="concept-card">
<h3>Gamma (γ)</h3>
<ul>
<li>Only for RBF and Poly kernels</li>
<li>High γ → Complex boundary, overfitting risk</li>
<li>Low γ → Smooth boundary, underfitting risk</li>
<li>Tip: Start with "scale"</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Decision Boundary
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎯 Decision Boundary")

    if model_trained:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        metrics = [
            ("🎯 Accuracy", f"{acc:.1%}"),
            ("⚙️ Kernel", kernel.upper()),
            ("📏 C", f"{C:.2f}"),
            ("🔵 Support Vectors", str(sum(n_support)))
        ]
        for col, (label, val) in zip([col_m1, col_m2, col_m3, col_m4], metrics):
            with col:
                st.markdown(f"""
<div class="metric-box">
    <div class="value">{val}</div>
    <div class="label">{label}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        style_ax(ax, fig)

        h = 0.03
        x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
        y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        cmap_bg = ListedColormap(['#ef535018', '#4fc3f718'])
        ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.7)
        ax.contour(xx, yy, Z, colors=['#00e5ff'], linewidths=2, alpha=0.9)

        colors = ['#ef5350', '#4fc3f7']
        for i, color in enumerate(colors):
            ax.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
                       c=color, s=50, alpha=0.8, edgecolors='white', linewidth=0.5, label=f'Train Class {i}')
            ax.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
                       c=color, s=80, marker='D', alpha=1.0, edgecolors='white', linewidth=1.0, label=f'Test Class {i}')

        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
                   edgecolors='#00e5ff', linewidth=2, zorder=10, label='Support Vectors')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'Decision Boundary — {kernel.upper()} Kernel | C={C:.2f}',
                     color='#4fc3f7', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=9, facecolor='#0d1b2a',
                  labelcolor='#ccd6f6', edgecolor='#1e3a5f')
        ax.set_xlabel("Feature 1 (normalized)", color='#8892b0')
        ax.set_ylabel("Feature 2 (normalized)", color='#8892b0')

        st.pyplot(fig)
        plt.close()

        st.info(f"🔵 **{sum(n_support)}** support vectors in this model  |  "
                f"Diamonds (◆) are test points")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Kernel Comparison
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🔬 Kernel Comparison")
    st.markdown("Same dataset, same C value — four different kernels side by side:")

    kernels = ["linear", "rbf", "poly", "sigmoid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('#0a0f1e')
    axes = axes.ravel()

    h = 0.05
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    results = {}
    for i, k in enumerate(kernels):
        ax = axes[i]
        style_ax(ax)

        try:
            m = svm.SVC(kernel=k, C=C, degree=3 if k == "poly" else 3, probability=True)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            acc_k = accuracy_score(y_test, pred)
            results[k] = acc_k

            Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            cmap_bg = ListedColormap(['#ef535018', '#4fc3f718'])
            ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.7)
            ax.contour(xx, yy, Z, colors=['#00e5ff'], linewidths=2, alpha=0.9)

            for cls, color in enumerate(['#ef5350', '#4fc3f7']):
                ax.scatter(X_scaled[y == cls, 0], X_scaled[y == cls, 1],
                           c=color, s=28, alpha=0.75, edgecolors='none')

            sv = m.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=110, facecolors='none',
                       edgecolors='#00e5ff', linewidth=1.5, zorder=10)

            ax.set_title(f'{k.upper()} — Accuracy: {acc_k:.1%}',
                         color='#4fc3f7', fontsize=11, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", transform=ax.transAxes,
                    ha='center', va='center', color='#ef5350', fontsize=9)
            ax.set_title(k.upper(), color='#8892b0')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Feature 1", color='#8892b0', fontsize=8)
        ax.set_ylabel("Feature 2", color='#8892b0', fontsize=8)

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    if results:
        st.markdown("### 📊 Comparison Table")
        df_results = pd.DataFrame({
            "Kernel": list(results.keys()),
            "Accuracy": [f"{v:.1%}" for v in results.values()],
            "Score": list(results.values())
        }).sort_values("Score", ascending=False).drop("Score", axis=1)
        df_results.index = range(1, len(df_results) + 1)
        st.dataframe(df_results, use_container_width=True)

        best_kernel = max(results, key=results.get)
        st.success(f"✅ Best kernel on this dataset: **{best_kernel.upper()}** with accuracy **{results[best_kernel]:.1%}**")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Model Evaluation
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📊 Model Evaluation")

    if model_trained:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            style_ax(ax, fig)

            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        cmap=sns.light_palette("#4fc3f7", as_cmap=True),
                        linewidths=0.5,
                        annot_kws={"size": 18, "weight": "bold", "color": "#0a0f1e"},
                        cbar_kws={'shrink': 0.8})
            ax.set_xlabel("Predicted", color='#8892b0')
            ax.set_ylabel("Actual", color='#8892b0')
            ax.set_title("Confusion Matrix", color='#4fc3f7', fontsize=12, fontweight='bold')
            ax.tick_params(colors='#8892b0')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True, height=250)

            st.markdown("### 🔍 Model Details")
            info_data = {
                "Parameter": ["Kernel", "C", "Support Vectors (Class 0)", "Support Vectors (Class 1)", "Train Size", "Test Size"],
                "Value": [
                    kernel, C,
                    n_support[0] if len(n_support) > 0 else "-",
                    n_support[1] if len(n_support) > 1 else "-",
                    len(X_train), len(X_test)
                ]
            }
            st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📈 Effect of C on Model Performance")

        C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
        train_accs, test_accs, sv_counts = [], [], []

        for c_val in C_values:
            try:
                m = svm.SVC(kernel=kernel, C=c_val)
                m.fit(X_train, y_train)
                train_accs.append(accuracy_score(y_train, m.predict(X_train)))
                test_accs.append(accuracy_score(y_test, m.predict(X_test)))
                sv_counts.append(sum(m.n_support_))
            except:
                train_accs.append(0); test_accs.append(0); sv_counts.append(0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0a0f1e')
        style_ax(ax1); style_ax(ax2)

        ax1.semilogx(C_values, train_accs, 'o-', color='#4fc3f7', lw=2, label='Train Accuracy')
        ax1.semilogx(C_values, test_accs, 's--', color='#ef5350', lw=2, label='Test Accuracy')
        ax1.axvline(x=C, color='#00e5ff', lw=1.5, ls=':', alpha=0.9, label=f'Current C={C}')
        ax1.set_xlabel("C (log scale)", color='#8892b0')
        ax1.set_ylabel("Accuracy", color='#8892b0')
        ax1.set_title("Accuracy vs C", color='#4fc3f7', fontsize=11, fontweight='bold')
        ax1.legend(facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f', fontsize=9)

        ax2.semilogx(C_values, sv_counts, 'o-', color='#00e5ff', lw=2)
        ax2.axvline(x=C, color='#ef5350', lw=1.5, ls=':', alpha=0.9, label=f'Current C={C}')
        ax2.set_xlabel("C (log scale)", color='#8892b0')
        ax2.set_ylabel("Number of Support Vectors", color='#8892b0')
        ax2.set_title("Support Vectors vs C", color='#4fc3f7', fontsize=11, fontweight='bold')
        ax2.legend(facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892b0; font-family:'JetBrains Mono',monospace; font-size:0.78rem; padding: 0.5rem 0;">
    SVM Explorer &nbsp;·&nbsp; Built with Streamlit &amp; scikit-learn &nbsp;·&nbsp; 🧠
</div>
""", unsafe_allow_html=True)
