import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, mean_squared_error, r2_score)
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #0a0f1e !important;
    font-family: 'DM Sans', sans-serif;
    color: #ccd6f6;
}
h1,h2,h3,h4 { font-family:'JetBrains Mono',monospace !important; color:#4fc3f7 !important; }
[data-testid="stSidebar"] { background-color:#0d1b2a !important; border-right:1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color:#ccd6f6 !important; }
[data-testid="stAppViewContainer"]>.main,
[data-testid="block-container"] { background-color:#0a0f1e !important; }

.main-header {
    background:linear-gradient(135deg,#0d1b2a,#112240,#0a1628);
    padding:1.8rem 2rem; border-radius:14px; margin-bottom:1.5rem;
    border:1px solid #1e3a5f; text-align:center;
    box-shadow:0 4px 30px #00e5ff0d;
}
.main-header h1 { color:#4fc3f7 !important; font-size:2rem; margin:0; text-shadow:0 0 40px #4fc3f766; }
.main-header p  { color:#8892b0; margin-top:0.4rem; font-size:0.9rem; }

.algo-badge {
    display:inline-block; padding:0.25rem 0.75rem; border-radius:20px;
    font-family:'JetBrains Mono',monospace; font-size:0.75rem;
    background:#112240; border:1px solid #4fc3f766; color:#4fc3f7;
    margin-bottom:1rem;
}
.card {
    background:linear-gradient(135deg,#0d1b2a,#112240);
    border:1px solid #1e3a5f; border-radius:12px;
    padding:1.2rem; margin:0.6rem 0; color:#ccd6f6;
}
.card h3 { color:#4fc3f7 !important; margin-top:0; font-size:0.95rem; }
.card ul  { padding-left:1.2rem; margin:0.4rem 0 0; color:#a8b2d8; font-size:0.88rem; line-height:1.7; }
.highlight {
    background:linear-gradient(90deg,#4fc3f711,transparent);
    border-left:3px solid #4fc3f7; padding:0.45rem 1rem;
    border-radius:0 8px 8px 0; margin:0.6rem 0; color:#ccd6f6; font-size:0.88rem;
}
.metric-box {
    background:#0d1b2a; border:1px solid #1e3a5f;
    border-radius:10px; padding:0.9rem; text-align:center; margin-bottom:0.5rem;
}
.metric-box .value { font-size:1.6rem; font-weight:700; color:#4fc3f7; font-family:'JetBrains Mono',monospace; }
.metric-box .label { color:#8892b0; font-size:0.78rem; margin-top:0.15rem; }

.upload-hint {
    background:#0d1b2a; border:2px dashed #1e3a5f; border-radius:12px;
    padding:1.2rem; text-align:center; color:#8892b0; font-size:0.88rem; margin:0.5rem 0;
}

.stTabs [data-baseweb="tab-list"] { gap:5px; background:transparent; }
.stTabs [data-baseweb="tab"] {
    background:#0d1b2a; border-radius:8px; color:#8892b0;
    border:1px solid #1e3a5f; font-family:'JetBrains Mono',monospace;
    font-size:0.8rem; padding:0.45rem 0.9rem;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#112240,#0d1b2a) !important;
    color:#4fc3f7 !important; border-color:#4fc3f766 !important;
}
.stAlert { background:#112240 !important; border:1px solid #1e3a5f !important; color:#ccd6f6 !important; border-radius:8px; }
div[data-baseweb="select"]>div { background:#112240 !important; border-color:#1e3a5f !important; color:#ccd6f6 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def style_ax(ax, fig=None):
    if fig: fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0d1b2a')
    ax.tick_params(colors='#8892b0', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')
    ax.grid(True, alpha=0.12, color='#4fc3f7')

def metric_box(val, label):
    return f'<div class="metric-box"><div class="value">{val}</div><div class="label">{label}</div></div>'

COLORS = ['#ef5350','#4fc3f7','#66bb6a','#ffa726','#ab47bc']

def plot_boundary(ax, clf, X_scaled, y, X_train, y_train, X_test, y_test, feat1="F1", feat2="F2"):
    h = 0.04
    xmn,xmx = X_scaled[:,0].min()-.6, X_scaled[:,0].max()+.6
    ymn,ymx = X_scaled[:,1].min()-.6, X_scaled[:,1].max()+.6
    xx,yy = np.meshgrid(np.arange(xmn,xmx,h), np.arange(ymn,ymx,h))
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    n_cls = len(np.unique(y))
    bg_colors = ['#ef535014','#4fc3f714','#66bb6a14','#ffa72614'][:n_cls]
    ax.contourf(xx,yy,Z, cmap=ListedColormap(bg_colors), alpha=0.8)
    ax.contour(xx,yy,Z, colors=['#00e5ff'], linewidths=1.5, alpha=0.7)
    for i in range(n_cls):
        c = COLORS[i]
        ax.scatter(X_train[y_train==i,0],X_train[y_train==i,1],
                   c=c,s=45,alpha=0.8,edgecolors='white',lw=0.4,label=f'Train cls {i}')
        ax.scatter(X_test[y_test==i,0],X_test[y_test==i,1],
                   c=c,s=70,marker='D',alpha=1,edgecolors='white',lw=0.8,label=f'Test cls {i}')
    ax.set_xlim(xmn,xmx); ax.set_ylim(ymn,ymx)
    ax.set_xlabel(f"{feat1} (norm.)", color='#8892b0', fontsize=8)
    ax.set_ylabel(f"{feat2} (norm.)", color='#8892b0', fontsize=8)
    ax.legend(fontsize=7, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f', loc='upper right')

def plot_confusion(ax, fig, y_test, y_pred):
    style_ax(ax, fig)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette("#4fc3f7", as_cmap=True),
                linewidths=0.5, annot_kws={"size":14,"weight":"bold","color":"#0a0f1e"},
                cbar_kws={'shrink':0.8})
    ax.set_xlabel("Predicted", color='#8892b0')
    ax.set_ylabel("Actual", color='#8892b0')
    ax.set_title("Confusion Matrix", color='#4fc3f7', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#8892b0')

# ── CSV Upload helper ──────────────────────────────────────────
def csv_uploader(key, task="classification"):
    """Returns (X, y, feat1_name, feat2_name) or None if not uploaded."""
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], key=key,
        help="Numeric feature columns + one label column"
    )
    if uploaded is None:
        return None
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); return None

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols  = df.columns.tolist()
    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns."); return None

    c1,c2,c3 = st.columns(3)
    with c1: f1 = st.selectbox("Feature 1", num_cols, index=0, key=key+"_f1")
    with c2: f2 = st.selectbox("Feature 2", num_cols, index=min(1,len(num_cols)-1), key=key+"_f2")
    with c3: lc = st.selectbox("Label column", all_cols, index=len(all_cols)-1, key=key+"_lc")

    try:
        valid = df[[f1,f2,lc]].dropna()
        X = valid[[f1,f2]].values.astype(float)
        y_raw = valid[lc].values
        uniq = np.unique(y_raw)
        if task == "classification" and len(uniq) < 2:
            st.error("Label must have at least 2 unique values."); return None
        lmap = {v:i for i,v in enumerate(uniq)}
        y = np.array([lmap[v] for v in y_raw])
    except Exception as e:
        st.error(f"Column error: {e}"); return None

    st.success(f"✅ {len(X)} samples loaded — {len(uniq)} classes")
    with st.expander("👀 Preview"):
        st.dataframe(valid[[f1,f2,lc]].head(10), use_container_width=True)
    return X, y, f1, f2

@st.cache_data
def builtin_clf(name, rs=42):
    if name == "Moons":       return datasets.make_moons(200,noise=0.25,random_state=rs)
    if name == "Circles":     return datasets.make_circles(200,noise=0.15,factor=0.4,random_state=rs)
    if name == "Blobs":       return datasets.make_blobs(200,centers=3,random_state=rs)
    if name == "Iris":
        d=datasets.load_iris(); return d.data[:,:2], d.target
    return datasets.make_classification(200,n_features=2,n_redundant=0,n_informative=2,
                                         random_state=rs,n_clusters_per_class=1)

@st.cache_data
def builtin_reg(name, rs=42):
    np.random.seed(rs)
    if name == "Linear":
        X=np.linspace(-3,3,200).reshape(-1,1); y=2.5*X.ravel()+np.random.randn(200)*1.2; return X,y
    if name == "Polynomial":
        X=np.linspace(-3,3,200).reshape(-1,1); y=0.8*X.ravel()**3-2*X.ravel()**2+np.random.randn(200)*2; return X,y
    if name == "Noisy":
        X=np.linspace(-3,3,200).reshape(-1,1); y=np.sin(X.ravel()*2)+np.random.randn(200)*1.5; return X,y
    d=datasets.load_diabetes(); return d.data[:,np.newaxis,2], d.target   # Boston-like fallback

# ══════════════════════════════════════════════════════════════
# SIDEBAR — Algorithm Selector
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🤖 ML Explorer")
    st.markdown("---")
    algo = st.radio(
        "Algorithm",
        ["⚡ SVM", "📈 Regression", "🌳 Decision Tree / RF",
         "🔵 K-Means Clustering", "👥 KNN", "🧠 Neural Network (MLP)",
         "🔁 AutoEncoder", "🎲 VAE (Variational)"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div style="color:#8892b0;font-size:0.78rem;">Tune parameters below ↓</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ██  SVM
# ══════════════════════════════════════════════════════════════
if algo == "⚡ SVM":
    st.markdown('<div class="main-header"><h1>⚡ Support Vector Machine</h1><p>Find the maximum-margin hyperplane between classes</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### SVM Settings")
        kernel = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"])
        C      = st.slider("C", 0.01, 20.0, 1.0, 0.01)
        if kernel=="rbf":
            gm = st.radio("Gamma",["scale","auto","custom"])
            gamma = st.slider("γ",0.001,5.0,0.5,0.001) if gm=="custom" else gm
            degree=3
        elif kernel=="poly":
            degree=st.slider("Degree",2,6,3); gamma="scale"
        else:
            gamma="scale"; degree=3
        st.markdown("---")
        st.markdown("### Data")
        src = st.radio("Source",["Built-in","Upload CSV"], key="svm_src")
        if src=="Built-in":
            dsn = st.selectbox("Dataset",["Moons","Circles","Blobs","Iris","Classification"])
        ts  = st.slider("Test size",0.1,0.5,0.2,0.05,key="svm_ts")
        rs  = st.number_input("Seed",0,100,42,key="svm_rs")

    # data
    if src=="Upload CSV":
        result = csv_uploader("svm_csv","classification")
        if result is None: st.stop()
        X,y,f1,f2 = result
    else:
        X,y = builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=ts,random_state=int(rs))
    clf=svm.SVC(kernel=kernel,C=C,gamma=gamma,degree=degree,probability=True)
    clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)

    tab1,tab2,tab3 = st.tabs(["📖 Concept","🎯 Decision Boundary","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>🎯 What is SVM? (Simple Version)</h3>
<p>Imagine red and blue dots scattered on a table. SVM tries to draw the <strong>widest possible street</strong> between the two groups. The center line is the decision boundary — left side = one class, right side = other.</p>
<div class="highlight">💡 Goal: draw the fattest dividing line between two groups</div></div>
<div class="card"><h3>📌 What are Support Vectors?</h3>
<p>The dots <strong>closest to the boundary street</strong>. They are the only ones that decide where the boundary goes. Delete all other dots → model stays exactly the same!</p>
<div class="highlight">💡 Only the "borderline" cases matter, easy cases are ignored</div></div>
<div class="card"><h3>🌀 What is the Kernel Trick?</h3>
<p>When no straight line can separate the groups (e.g. one class is a ring), the kernel <strong>lifts data into higher dimensions</strong> where a flat cut works — like crumpling paper so two groups land on different levels.</p>
<div class="highlight">💡 RBF kernel is the safest default for most problems</div></div>
<div class="card"><h3>🎛️ What does C control?</h3><ul>
<li><strong>Small C</strong> → wide street, allows some errors → simpler model</li>
<li><strong>Large C</strong> → narrow street, tries to get all points right → risks overfitting</li>
<li>Start with C=1.0, tune from there</li></ul></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,2,figsize=(9,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            np.random.seed(0)
            Xp=np.random.randn(15,2)+[2,2]; Xn=np.random.randn(15,2)+[-2,-2]
            xl=np.linspace(-5,5,100)
            axes[0].scatter(Xp[:,0],Xp[:,1],c='#4fc3f7',s=55,zorder=5,label='Class +1')
            axes[0].scatter(Xn[:,0],Xn[:,1],c='#ef5350',s=55,zorder=5,label='Class -1')
            axes[0].plot(xl,xl*0,color='#00e5ff',lw=2.5,label='Hyperplane')
            axes[0].fill_between(xl,-0.8,0.8,alpha=0.12,color='#00e5ff')
            axes[0].axhline(0.8,color='#00e5ff',lw=1,ls='--',alpha=0.5)
            axes[0].axhline(-0.8,color='#00e5ff',lw=1,ls='--',alpha=0.5)
            axes[0].set_xlim(-5,5); axes[0].set_ylim(-5,5)
            axes[0].set_title('Margin & Hyperplane',color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[0].legend(fontsize=7,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            th=np.linspace(0,2*np.pi,50)
            axes[1].scatter(0.8*np.cos(th[:25]),0.8*np.sin(th[:25]),c='#4fc3f7',s=55,zorder=5)
            axes[1].scatter(1.8*np.cos(th),1.8*np.sin(th),c='#ef5350',s=55,zorder=5)
            axes[1].add_patch(plt.Circle((0,0),1.3,fill=False,color='#00e5ff',lw=2.5))
            axes[1].set_xlim(-2.5,2.5); axes[1].set_ylim(-2.5,2.5); axes[1].set_aspect('equal')
            axes[1].set_title('Kernel Trick (RBF)',color='#4fc3f7',fontsize=10,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["🎯 Accuracy","⚙️ Kernel","📏 C","Support Vectors"],
                                [f"{acc:.1%}",kernel.upper(),f"{C:.2f}",str(sum(clf.n_support_))]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        plot_boundary(ax,clf,Xs,y,Xtr,ytr,Xte,yte,f1,f2)
        sv=clf.support_vectors_
        ax.scatter(sv[:,0],sv[:,1],s=180,facecolors='none',edgecolors='#00e5ff',lw=2,zorder=10,label='Support Vecs')
        ax.set_title(f'SVM — {kernel.upper()} | C={C:.2f} | Acc={acc:.1%}',color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=7,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); plot_confusion(ax,fig,yte,yp); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(pd.DataFrame(classification_report(yte,yp,output_dict=True)).T.round(3),
                         use_container_width=True,height=220)

# ══════════════════════════════════════════════════════════════
# ██  REGRESSION
# ══════════════════════════════════════════════════════════════
elif algo == "📈 Regression":
    st.markdown('<div class="main-header"><h1>📈 Regression</h1><p>Predict continuous values — Linear, Polynomial, Ridge, Lasso</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Regression Settings")
        reg_type = st.selectbox("Model", ["Linear","Polynomial","Ridge","Lasso"])
        if reg_type=="Polynomial":
            deg = st.slider("Degree",2,8,3)
        alpha = st.slider("Alpha (Ridge/Lasso)",0.001,10.0,1.0,0.001) if reg_type in ["Ridge","Lasso"] else 1.0
        st.markdown("---")
        st.markdown("### Data")
        src = st.radio("Source",["Built-in","Upload CSV"],key="reg_src")
        if src=="Built-in":
            dsn = st.selectbox("Dataset",["Linear","Polynomial","Noisy"])
        ts  = st.slider("Test size",0.1,0.5,0.2,0.05,key="reg_ts")
        rs  = st.number_input("Seed",0,100,42,key="reg_rs")

    if src=="Upload CSV":
        st.markdown("### 📂 Upload your CSV")
        uploaded=st.file_uploader("CSV file",type=["csv"],key="reg_csv")
        if uploaded is None:
            st.info("⬆️ Upload a CSV or switch to Built-in."); st.stop()
        df=pd.read_csv(uploaded)
        num_cols=df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols)<2: st.error("Need at least 2 numeric columns."); st.stop()
        c1,c2=st.columns(2)
        with c1: fx=st.selectbox("Feature (X)",num_cols,index=0,key="reg_fx")
        with c2: fy=st.selectbox("Target (Y)",num_cols,index=len(num_cols)-1,key="reg_fy")
        valid=df[[fx,fy]].dropna()
        X=valid[[fx]].values.astype(float); y=valid[fy].values.astype(float)
        st.success(f"✅ {len(X)} rows loaded")
        f_xlabel,f_ylabel=fx,fy
    else:
        X,y=builtin_reg(dsn,rs); f_xlabel,f_ylabel="X","y"

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts,random_state=int(rs))

    if reg_type=="Linear":      model=LinearRegression()
    elif reg_type=="Polynomial": model=Pipeline([("poly",PolynomialFeatures(deg,include_bias=False)),("lr",LinearRegression())])
    elif reg_type=="Ridge":      model=Pipeline([("poly",PolynomialFeatures(3,include_bias=False)),("r",Ridge(alpha=alpha))])
    else:                        model=Pipeline([("poly",PolynomialFeatures(3,include_bias=False)),("l",Lasso(alpha=alpha))])

    model.fit(Xtr,ytr)
    yp_tr=model.predict(Xtr); yp_te=model.predict(Xte)
    r2_tr=r2_score(ytr,yp_tr); r2_te=r2_score(yte,yp_te)
    rmse=np.sqrt(mean_squared_error(yte,yp_te))

    tab1,tab2,tab3=st.tabs(["📖 Concept","📈 Fit","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>📈 What is Regression?</h3>
<p>Regression predicts a <strong>number</strong> (not a category). For example: "given a house's size, predict its price." It finds a mathematical relationship between input and output.</p>
<div class="highlight">💡 If the answer is a number → use regression. If it's a category → use classification</div></div>

<div class="card"><h3>📏 Linear Regression</h3>
<p>Fits the simplest possible line: <strong>y = w×x + b</strong>. It minimizes the total squared distance between the line and all data points.</p>
<ul>
<li><strong>w</strong> = slope (how steep the line is)</li>
<li><strong>b</strong> = intercept (where it crosses y-axis)</li>
<li>Works great when the relationship is roughly a straight line</li></ul>
<div class="highlight">💡 Always try linear first — simplest models are easiest to trust</div></div>

<div class="card"><h3>🔢 Polynomial Regression</h3>
<p>When data curves, we add higher powers (x², x³...) to bend the line. <strong>Degree</strong> controls how curvy the fit is.</p>
<ul>
<li>Degree 1 = straight line</li>
<li>Degree 2 = parabola (U-shape)</li>
<li>Degree 8+ = very wiggly → overfitting risk!</li></ul>
<div class="highlight">💡 Higher degree = more flexible but easier to overfit</div></div>

<div class="card"><h3>🔒 Ridge & Lasso (Regularization)</h3>
<p>When a polynomial model fits training data too perfectly (memorizes noise), we add a <strong>penalty</strong> for large coefficients to force simplicity.</p>
<ul>
<li><strong>Ridge (L2)</strong>: shrinks all coefficients toward zero — none become exactly 0</li>
<li><strong>Lasso (L1)</strong>: can force some coefficients to exactly 0 → automatic feature selection</li>
<li><strong>Alpha</strong>: how strong the penalty is. Higher alpha = simpler model</li></ul>
<div class="highlight">💡 Use Ridge when all features matter; Lasso when you want feature selection</div></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,2,figsize=(9,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            xs=np.linspace(-3,3,200).reshape(-1,1)
            np.random.seed(0)
            Xd=np.linspace(-3,3,60).reshape(-1,1)
            yd_lin=2*Xd.ravel()+np.random.randn(60)*0.8
            yd_pol=Xd.ravel()**3-2*Xd.ravel()+np.random.randn(60)*1.5
            axes[0].scatter(Xd,yd_lin,c='#4fc3f7',s=25,alpha=0.7,label='data')
            m=LinearRegression().fit(Xd,yd_lin)
            axes[0].plot(xs,m.predict(xs),color='#00e5ff',lw=2.5,label='Linear fit')
            axes[0].set_title('Linear Regression',color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[0].legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            axes[1].scatter(Xd,yd_pol,c='#4fc3f7',s=25,alpha=0.7,label='data')
            for d2,col,lbl in [(2,'#ffa726','deg 2'),(4,'#00e5ff','deg 4'),(8,'#ef5350','deg 8')]:
                p=Pipeline([("p",PolynomialFeatures(d2)),("r",LinearRegression())]).fit(Xd,yd_pol)
                axes[1].plot(xs,p.predict(xs),color=col,lw=1.8,label=lbl)
            axes[1].set_title('Polynomial Degrees',color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[1].legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["📈 Model","R² Train","R² Test","RMSE"],
                                [reg_type,f"{r2_tr:.3f}",f"{r2_te:.3f}",f"{rmse:.3f}"]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(9,5)); style_ax(ax,fig)
        ax.scatter(Xtr,ytr,c='#4fc3f7',s=35,alpha=0.7,label='Train data')
        ax.scatter(Xte,yte,c='#ffa726',s=55,marker='D',alpha=0.9,label='Test data')
        xs=np.linspace(X.min(),X.max(),300).reshape(-1,1)
        ax.plot(xs,model.predict(xs),color='#00e5ff',lw=2.5,label=f'{reg_type} fit')
        ax.set_xlabel(f_xlabel,color='#8892b0'); ax.set_ylabel(f_ylabel,color='#8892b0')
        ax.set_title(f'{reg_type} Regression | R²={r2_te:.3f} | RMSE={rmse:.3f}',
                     color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); style_ax(ax,fig)
            ax.scatter(yte,yp_te,c='#4fc3f7',s=40,alpha=0.7,edgecolors='white',lw=0.3)
            mn,mx=min(yte.min(),yp_te.min()),max(yte.max(),yp_te.max())
            ax.plot([mn,mx],[mn,mx],'--',color='#00e5ff',lw=2,label='Perfect fit')
            ax.set_xlabel("Actual",color='#8892b0'); ax.set_ylabel("Predicted",color='#8892b0')
            ax.set_title("Actual vs Predicted",color='#4fc3f7',fontsize=11,fontweight='bold')
            ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            st.pyplot(fig); plt.close()
        with c2:
            fig,ax=plt.subplots(figsize=(5,4)); style_ax(ax,fig)
            res=yte-yp_te
            ax.scatter(yp_te,res,c='#4fc3f7',s=40,alpha=0.7,edgecolors='white',lw=0.3)
            ax.axhline(0,color='#00e5ff',lw=2,ls='--')
            ax.set_xlabel("Predicted",color='#8892b0'); ax.set_ylabel("Residual",color='#8892b0')
            ax.set_title("Residual Plot",color='#4fc3f7',fontsize=11,fontweight='bold')
            st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# ██  DECISION TREE / RANDOM FOREST
# ══════════════════════════════════════════════════════════════
elif algo == "🌳 Decision Tree / RF":
    st.markdown('<div class="main-header"><h1>🌳 Decision Tree & Random Forest</h1><p>Rule-based splitting meets ensemble power</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Tree Settings")
        model_type=st.radio("Model",["Decision Tree","Random Forest"])
        max_depth =st.slider("Max Depth",1,15,4)
        if model_type=="Random Forest":
            n_est=st.slider("# Estimators",10,200,100,10)
        st.markdown("---")
        st.markdown("### Data")
        src=st.radio("Source",["Built-in","Upload CSV"],key="tree_src")
        if src=="Built-in":
            dsn=st.selectbox("Dataset",["Moons","Circles","Blobs","Iris","Classification"])
        ts =st.slider("Test size",0.1,0.5,0.2,0.05,key="tree_ts")
        rs =st.number_input("Seed",0,100,42,key="tree_rs")

    if src=="Upload CSV":
        result=csv_uploader("tree_csv","classification")
        if result is None: st.stop()
        X,y,f1,f2=result
    else:
        X,y=builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=ts,random_state=int(rs))

    if model_type=="Decision Tree":
        clf=DecisionTreeClassifier(max_depth=max_depth,random_state=int(rs))
    else:
        clf=RandomForestClassifier(n_estimators=n_est,max_depth=max_depth,random_state=int(rs))
    clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)

    tab1,tab2,tab3=st.tabs(["📖 Concept","🌳 Boundary","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>🌳 What is a Decision Tree?</h3>
<p>A decision tree is literally a flowchart of yes/no questions. At each step it asks "is feature X greater than value Y?" and follows the branch. At the end (a leaf) it gives a prediction.</p>
<p>Example: <em>"Is age > 30? → Yes → Is salary > 50k? → Yes → Approve loan"</em></p>
<div class="highlight">💡 It's the most human-readable ML model — you can follow its logic step by step</div></div>

<div class="card"><h3>📏 What is Max Depth?</h3>
<p>Max depth limits how many questions the tree is allowed to ask. Think of it as how tall the tree can grow.</p>
<ul>
<li><strong>Depth 1</strong> = one yes/no question, very simple</li>
<li><strong>Depth 4–6</strong> = usually a good balance</li>
<li><strong>Depth unlimited</strong> = the tree memorizes training data → fails on new data (overfitting)</li></ul>
<div class="highlight">💡 If your model is 100% accurate on training but bad on test → try reducing depth</div></div>

<div class="card"><h3>🌲🌲 What is a Random Forest?</h3>
<p>Instead of one tree, build <strong>hundreds of trees</strong> — each trained on a random subset of data and features. Then take a majority vote across all trees.</p>
<p>One tree might be wrong, but 100 trees voting together are much harder to fool.</p>
<ul>
<li>More trees = more stable predictions (but slower to train)</li>
<li>Each tree sees different data → diversity = strength</li></ul>
<div class="highlight">💡 Random Forest is one of the best "out of the box" algorithms — try it first!</div></div>

<div class="card"><h3>🏆 Feature Importance</h3>
<p>Random Forest tells you which features were most useful for making decisions — great for understanding your data!</p>
<div class="highlight">💡 High importance = that feature splits data cleanly and often</div></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,2,figsize=(9,4.5)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            np.random.seed(42)
            Xm,ym=datasets.make_moons(100,noise=0.2,random_state=42)
            sc2=StandardScaler(); Xms=sc2.fit_transform(Xm)
            h=0.05; xmn,xmx=Xms[:,0].min()-.5,Xms[:,0].max()+.5
            ymn2,ymx2=Xms[:,1].min()-.5,Xms[:,1].max()+.5
            xx2,yy2=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn2,ymx2,h))
            for ax,md,title in zip(axes,[2,8],['Depth=2 (simple)','Depth=8 (overfit risk)']):
                dt=DecisionTreeClassifier(max_depth=md).fit(Xms,ym)
                Z=dt.predict(np.c_[xx2.ravel(),yy2.ravel()]).reshape(xx2.shape)
                ax.contourf(xx2,yy2,Z,cmap=ListedColormap(['#ef535018','#4fc3f718']),alpha=0.8)
                ax.contour(xx2,yy2,Z,colors=['#00e5ff'],linewidths=1.5,alpha=0.7)
                ax.scatter(Xms[ym==0,0],Xms[ym==0,1],c='#ef5350',s=35,alpha=0.8,edgecolors='white',lw=0.3)
                ax.scatter(Xms[ym==1,0],Xms[ym==1,1],c='#4fc3f7',s=35,alpha=0.8,edgecolors='white',lw=0.3)
                ax.set_title(title,color='#4fc3f7',fontsize=10,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        n_leaves=clf.get_n_leaves() if hasattr(clf,'get_n_leaves') else "—"
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["🎯 Accuracy","🌳 Model","📏 Max Depth","🍃 Leaves"],
                                [f"{acc:.1%}",model_type,str(max_depth),str(n_leaves)]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        plot_boundary(ax,clf,Xs,y,Xtr,ytr,Xte,yte,f1,f2)
        ax.set_title(f'{model_type} | Depth={max_depth} | Acc={acc:.1%}',
                     color='#4fc3f7',fontsize=12,fontweight='bold')
        st.pyplot(fig); plt.close()

        if model_type=="Random Forest":
            st.markdown("### 🏆 Feature Importance")
            fi=clf.feature_importances_
            fig,ax=plt.subplots(figsize=(5,2.5)); style_ax(ax,fig)
            ax.barh([f2,f1],fi[::-1],color=['#4fc3f7','#00e5ff'])
            ax.set_xlabel("Importance",color='#8892b0')
            ax.set_title("Feature Importances",color='#4fc3f7',fontsize=10,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); plot_confusion(ax,fig,yte,yp); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(pd.DataFrame(classification_report(yte,yp,output_dict=True)).T.round(3),
                         use_container_width=True,height=220)

# ══════════════════════════════════════════════════════════════
# ██  K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════
elif algo == "🔵 K-Means Clustering":
    st.markdown('<div class="main-header"><h1>🔵 K-Means Clustering</h1><p>Unsupervised grouping — discover hidden structure in data</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### K-Means Settings")
        k       = st.slider("Number of Clusters (K)",2,8,3)
        max_it  = st.slider("Max Iterations",10,500,300,10)
        n_init  = st.slider("N Init",1,20,10)
        st.markdown("---")
        st.markdown("### Data")
        src=st.radio("Source",["Built-in","Upload CSV"],key="km_src")
        if src=="Built-in":
            dsn=st.selectbox("Dataset",["Blobs","Moons","Circles","Iris"])
        rs=st.number_input("Seed",0,100,42,key="km_rs")

    if src=="Upload CSV":
        result=csv_uploader("km_csv","clustering")
        if result is None: st.stop()
        X,_,f1,f2=result
    else:
        X,y_true=builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    km=KMeans(n_clusters=k,max_iter=max_it,n_init=n_init,random_state=int(rs))
    labels=km.fit_predict(Xs)
    inertia=km.inertia_

    tab1,tab2,tab3=st.tabs(["📖 Concept","🔵 Clusters","📊 Elbow Curve"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>🔵 What is K-Means? (No Labels Needed!)</h3>
<p>K-Means is <strong>unsupervised</strong> — you don't give it any labels. You just say "find me K groups" and it discovers natural clusters on its own.</p>
<p>Think of dropping K magnets on a map of cities. Each city sticks to its nearest magnet. Then magnets move to the center of their cities. Repeat until stable.</p>
<div class="highlight">💡 Use K-Means when you want to discover hidden groups in data</div></div>

<div class="card"><h3>🔄 How the Algorithm Works (Step by Step)</h3><ul>
<li><strong>Step 1</strong>: Randomly place K "centroids" (imaginary center points)</li>
<li><strong>Step 2</strong>: Each data point joins the nearest centroid's cluster</li>
<li><strong>Step 3</strong>: Each centroid moves to the average position of its cluster</li>
<li><strong>Step 4</strong>: Repeat steps 2–3 until nothing changes</li></ul>
<div class="highlight">💡 Usually converges in under 50 iterations</div></div>

<div class="card"><h3>📏 How to Choose K?</h3>
<p>This is the hardest part! Use the <strong>Elbow Method</strong>: run K-Means for K=1,2,3...10 and plot the "inertia" (total distance from points to their centroid). The plot bends like an elbow — pick that K.</p>
<ul>
<li>Too few clusters → groups are too big and mixed</li>
<li>Too many clusters → every point is its own group (useless)</li></ul>
<div class="highlight">💡 The elbow point is the sweet spot — diminishing returns after it</div></div>

<div class="card"><h3>⚠️ Limitations to Know</h3><ul>
<li>Assumes clusters are roughly round (spherical)</li>
<li>Sensitive to outliers — one far point skews a centroid</li>
<li>You must choose K in advance</li>
<li>Results can vary — run multiple times (n_init does this automatically)</li></ul></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,2,figsize=(9,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            Xb,_=datasets.make_blobs(150,centers=3,random_state=0)
            sc3=StandardScaler(); Xbs=sc3.fit_transform(Xb)
            km3=KMeans(3,random_state=0,n_init=10).fit(Xbs)
            for i in range(3):
                axes[0].scatter(Xbs[km3.labels_==i,0],Xbs[km3.labels_==i,1],
                                c=COLORS[i],s=35,alpha=0.8,edgecolors='white',lw=0.3)
            axes[0].scatter(km3.cluster_centers_[:,0],km3.cluster_centers_[:,1],
                            c='#00e5ff',s=200,marker='*',zorder=10,edgecolors='white',lw=1,label='Centroids')
            axes[0].set_title('K=3 Clusters',color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[0].legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            ks=range(1,9); inertias=[KMeans(ki,n_init=10,random_state=0).fit(Xbs).inertia_ for ki in ks]
            axes[1].plot(ks,inertias,'o-',color='#4fc3f7',lw=2)
            axes[1].set_xlabel("K",color='#8892b0'); axes[1].set_ylabel("Inertia",color='#8892b0')
            axes[1].set_title("Elbow Curve",color='#4fc3f7',fontsize=10,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3=st.columns(3)
        for col,lbl,val in zip([m1,m2,m3],["🔵 K","📉 Inertia","📦 Samples"],
                                [str(k),f"{inertia:.1f}",str(len(Xs))]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        for i in range(k):
            mask=labels==i
            ax.scatter(Xs[mask,0],Xs[mask,1],c=COLORS[i%len(COLORS)],s=45,
                       alpha=0.8,edgecolors='white',lw=0.3,label=f'Cluster {i}')
        ctrs=km.cluster_centers_
        ax.scatter(ctrs[:,0],ctrs[:,1],c='#00e5ff',s=250,marker='*',zorder=10,
                   edgecolors='white',lw=1,label='Centroids')
        ax.set_xlabel(f"{f1} (norm.)",color='#8892b0'); ax.set_ylabel(f"{f2} (norm.)",color='#8892b0')
        ax.set_title(f'K-Means | K={k} | Inertia={inertia:.1f}',color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("### 📊 Elbow Curve — Find the Best K")
        ks=range(1,11)
        inertias=[KMeans(ki,max_iter=max_it,n_init=n_init,random_state=int(rs)).fit(Xs).inertia_ for ki in ks]
        fig,ax=plt.subplots(figsize=(8,4)); style_ax(ax,fig)
        ax.plot(ks,inertias,'o-',color='#4fc3f7',lw=2.5,markersize=8)
        ax.axvline(k,color='#00e5ff',lw=2,ls='--',alpha=0.8,label=f'Current K={k}')
        ax.set_xlabel("Number of Clusters (K)",color='#8892b0')
        ax.set_ylabel("Inertia",color='#8892b0')
        ax.set_title("Elbow Curve",color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        st.pyplot(fig); plt.close()
        st.info("💡 Pick the K where the curve bends sharply — that's your elbow.")

# ══════════════════════════════════════════════════════════════
# ██  KNN
# ══════════════════════════════════════════════════════════════
elif algo == "👥 KNN":
    st.markdown('<div class="main-header"><h1>👥 K-Nearest Neighbors</h1><p>Classify by majority vote of the K closest points</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### KNN Settings")
        k       = st.slider("K (neighbors)",1,25,5)
        weights = st.radio("Weights",["uniform","distance"])
        metric  = st.selectbox("Distance Metric",["euclidean","manhattan","minkowski"])
        st.markdown("---")
        st.markdown("### Data")
        src=st.radio("Source",["Built-in","Upload CSV"],key="knn_src")
        if src=="Built-in":
            dsn=st.selectbox("Dataset",["Moons","Circles","Blobs","Iris","Classification"])
        ts =st.slider("Test size",0.1,0.5,0.2,0.05,key="knn_ts")
        rs =st.number_input("Seed",0,100,42,key="knn_rs")

    if src=="Upload CSV":
        result=csv_uploader("knn_csv","classification")
        if result is None: st.stop()
        X,y,f1,f2=result
    else:
        X,y=builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=ts,random_state=int(rs))
    clf=KNeighborsClassifier(n_neighbors=k,weights=weights,metric=metric)
    clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)

    tab1,tab2,tab3=st.tabs(["📖 Concept","👥 Boundary","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>👥 What is KNN? (The Simplest Idea in ML)</h3>
<p>KNN asks one question: <strong>"Who are my closest neighbors, and what class are they?"</strong></p>
<p>To classify a new point, it looks at the K nearest data points it already knows, and takes a majority vote. If 3 out of 5 neighbors are "cat", the prediction is "cat".</p>
<p>There's literally no training — KNN just memorizes all the data and compares at prediction time.</p>
<div class="highlight">💡 Analogy: you move to a new neighborhood and guess the area's vibe by talking to your K nearest neighbors</div></div>

<div class="card"><h3>🔢 Choosing K — The Most Important Decision</h3><ul>
<li><strong>K=1</strong>: classify by the single nearest neighbor → very sensitive to noise, overfits</li>
<li><strong>K=3 or 5</strong>: small neighborhood vote → good balance, recommended starting point</li>
<li><strong>K=20+</strong>: very large vote → too blurry, starts ignoring local structure (underfits)</li>
<li><strong>Odd K</strong>: avoid ties in binary problems (K=5 instead of K=4)</li></ul>
<div class="highlight">💡 Use the Accuracy vs K chart in the Boundary tab to find the sweet spot</div></div>

<div class="card"><h3>📏 Distance Metrics — How "Close" is Measured</h3><ul>
<li><strong>Euclidean</strong>: straight-line distance (like measuring with a ruler) — most common</li>
<li><strong>Manhattan</strong>: distance along grid lines (like walking city blocks) — better when features are independent</li>
<li><strong>Minkowski</strong>: a generalization that includes both above</li></ul>
<div class="highlight">💡 Always scale your features before KNN — unscaled features bias the distance!</div></div>

<div class="card"><h3>⚖️ Weights: Uniform vs Distance</h3><ul>
<li><strong>Uniform</strong>: all K neighbors vote equally</li>
<li><strong>Distance</strong>: closer neighbors get a stronger vote — usually slightly better</li></ul></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,3,figsize=(12,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            Xm,ym=datasets.make_moons(120,noise=0.2,random_state=0)
            sc4=StandardScaler(); Xms=sc4.fit_transform(Xm)
            h=0.06; xmn,xmx=Xms[:,0].min()-.5,Xms[:,0].max()+.5
            ymn3,ymx3=Xms[:,1].min()-.5,Xms[:,1].max()+.5
            xx3,yy3=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn3,ymx3,h))
            for ax,kk,title in zip(axes,[1,5,20],['K=1 (overfit)','K=5 (balanced)','K=20 (underfit)']):
                kn=KNeighborsClassifier(n_neighbors=kk).fit(Xms,ym)
                Z=kn.predict(np.c_[xx3.ravel(),yy3.ravel()]).reshape(xx3.shape)
                ax.contourf(xx3,yy3,Z,cmap=ListedColormap(['#ef535018','#4fc3f718']),alpha=0.8)
                ax.contour(xx3,yy3,Z,colors=['#00e5ff'],linewidths=1.5,alpha=0.7)
                ax.scatter(Xms[ym==0,0],Xms[ym==0,1],c='#ef5350',s=30,alpha=0.8,edgecolors='white',lw=0.3)
                ax.scatter(Xms[ym==1,0],Xms[ym==1,1],c='#4fc3f7',s=30,alpha=0.8,edgecolors='white',lw=0.3)
                ax.set_title(title,color='#4fc3f7',fontsize=9,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["🎯 Accuracy","👥 K","⚖️ Weights","📏 Metric"],
                                [f"{acc:.1%}",str(k),weights,metric]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        plot_boundary(ax,clf,Xs,y,Xtr,ytr,Xte,yte,f1,f2)
        ax.set_title(f'KNN | K={k} | {weights} | Acc={acc:.1%}',color='#4fc3f7',fontsize=12,fontweight='bold')
        st.pyplot(fig); plt.close()

        st.markdown("### 📈 Accuracy vs K")
        ks=range(1,31); accs=[accuracy_score(yte,KNeighborsClassifier(ki,weights=weights,metric=metric).fit(Xtr,ytr).predict(Xte)) for ki in ks]
        fig,ax=plt.subplots(figsize=(8,3.5)); style_ax(ax,fig)
        ax.plot(ks,accs,'o-',color='#4fc3f7',lw=2,markersize=5)
        ax.axvline(k,color='#00e5ff',lw=2,ls='--',alpha=0.9,label=f'Current K={k}')
        ax.set_xlabel("K",color='#8892b0'); ax.set_ylabel("Test Accuracy",color='#8892b0')
        ax.set_title("Accuracy vs K",color='#4fc3f7',fontsize=11,fontweight='bold')
        ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); plot_confusion(ax,fig,yte,yp); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(pd.DataFrame(classification_report(yte,yp,output_dict=True)).T.round(3),
                         use_container_width=True,height=220)

# ══════════════════════════════════════════════════════════════
# ██  MLP / NEURAL NETWORK
# ══════════════════════════════════════════════════════════════
elif algo == "🧠 Neural Network (MLP)":
    st.markdown('<div class="main-header"><h1>🧠 Neural Network (MLP)</h1><p>Multi-layer perceptron — universal function approximator</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### MLP Settings")
        h1     = st.slider("Layer 1 neurons",4,128,64,4)
        h2     = st.slider("Layer 2 neurons",0,128,32,4)
        h3     = st.slider("Layer 3 neurons",0,64,0,4)
        layers = tuple(n for n in [h1,h2,h3] if n>0)
        act    = st.selectbox("Activation",["relu","tanh","logistic"])
        lr     = st.select_slider("Learning Rate",options=[0.0001,0.001,0.01,0.1],value=0.001)
        max_it = st.slider("Max Iterations",50,500,200,50)
        st.markdown("---")
        st.markdown("### Data")
        src=st.radio("Source",["Built-in","Upload CSV"],key="mlp_src")
        if src=="Built-in":
            dsn=st.selectbox("Dataset",["Moons","Circles","Blobs","Iris","Classification"])
        ts =st.slider("Test size",0.1,0.5,0.2,0.05,key="mlp_ts")
        rs =st.number_input("Seed",0,100,42,key="mlp_rs")

    if src=="Upload CSV":
        result=csv_uploader("mlp_csv","classification")
        if result is None: st.stop()
        X,y,f1,f2=result
    else:
        X,y=builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=ts,random_state=int(rs))
    clf=MLPClassifier(hidden_layer_sizes=layers,activation=act,
                      learning_rate_init=lr,max_iter=max_it,random_state=int(rs))
    clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)

    tab1,tab2,tab3=st.tabs(["📖 Concept","🧠 Boundary","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown(f"""
<div class="card"><h3>🧠 What is a Neural Network?</h3>
<p>A neural network is loosely inspired by the human brain. It's made of layers of <strong>neurons</strong> — each neuron takes some numbers in, does a simple math operation, and passes a number out.</p>
<p>Stack many layers of these neurons and the network can learn incredibly complex patterns — from recognizing faces to translating languages.</p>
<div class="highlight">💡 Think of it as: many tiny math functions chained together that learn from mistakes</div></div>

<div class="card"><h3>🏗️ What are Layers?</h3><ul>
<li><strong>Input Layer</strong>: your raw data goes in here (2 features in our case)</li>
<li><strong>Hidden Layers</strong>: where learning happens — each layer extracts higher-level patterns</li>
<li><strong>Output Layer</strong>: gives the final answer ({len(np.unique(y))} classes here)</li></ul>
<p>More hidden layers = deeper network = can learn more complex patterns (but needs more data)</p>
<div class="highlight">💡 Your current network: {" → ".join([str(n) for n in [2]+list(layers)+[len(np.unique(y))]])}</div></div>

<div class="card"><h3>⚡ What is an Activation Function?</h3>
<p>Without activations, stacking layers would just be one big linear equation — useless. Activations add <strong>non-linearity</strong> (curves) so the network can fit complex shapes.</p>
<ul>
<li><strong>ReLU</strong>: outputs max(0, x) — fast, great default for most problems</li>
<li><strong>Tanh</strong>: smooth S-curve between -1 and 1 — good for centered data</li>
<li><strong>Logistic</strong>: outputs 0 to 1 — classic, but slower to train</li></ul>
<div class="highlight">💡 Start with ReLU — it works well in most situations</div></div>

<div class="card"><h3>📉 How Does it Learn? (Backpropagation)</h3>
<p>The network makes a prediction, measures how wrong it was (loss), then works <strong>backwards</strong> through each layer adjusting weights to reduce the error. Repeat thousands of times.</p>
<ul>
<li><strong>Learning Rate</strong>: how big each adjustment step is — too high = overshoots, too low = slow</li>
<li><strong>Iterations</strong>: how many times the full dataset is passed through</li></ul>
<div class="highlight">💡 Watch the Loss Curve — if it's still going down, train longer!</div></div>
""", unsafe_allow_html=True)
        with c2:
            # Network diagram
            fig,ax=plt.subplots(figsize=(8,4.5)); fig.patch.set_facecolor('#0a0f1e'); ax.axis('off')
            ax.set_facecolor('#0a0f1e')
            all_layers=[2]+list(layers)+[len(np.unique(y))]
            max_n=max(all_layers); lx=np.linspace(0.1,0.9,len(all_layers))
            node_pos={}
            for li,n in enumerate(all_layers):
                ys=np.linspace(0.5-(n-1)*0.08,0.5+(n-1)*0.08,n)
                for ni,y_pos in enumerate(ys):
                    node_pos[(li,ni)]=(lx[li],y_pos)
                    c_node='#4fc3f7' if li==0 else ('#00e5ff' if li==len(all_layers)-1 else '#112240')
                    ec='#4fc3f7' if li in [0,len(all_layers)-1] else '#1e3a5f'
                    circle=plt.Circle((lx[li],y_pos),0.025,color=c_node,ec=ec,lw=1.5,zorder=5)
                    ax.add_patch(circle)
            for li in range(len(all_layers)-1):
                n1,n2=all_layers[li],all_layers[li+1]
                for ni in range(n1):
                    for nj in range(n2):
                        x1,y1=node_pos[(li,ni)]; x2,y2=node_pos[(li+1,nj)]
                        ax.plot([x1,x2],[y1,y2],color='#1e3a5f',lw=0.6,alpha=0.5,zorder=1)
            labels=['Input']+[f'Hidden {i+1}' for i in range(len(layers))]+['Output']
            for li,(label,x) in enumerate(zip(labels,lx)):
                ax.text(x,0.08,label,ha='center',va='center',color='#8892b0',fontsize=8,
                        fontfamily='monospace')
                ax.text(x,0.97,str(all_layers[li]),ha='center',va='center',color='#4fc3f7',
                        fontsize=9,fontweight='bold',fontfamily='monospace')
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_title(f'Network: {" → ".join(map(str,all_layers))}',
                         color='#4fc3f7',fontsize=11,fontweight='bold')
            st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["🎯 Accuracy","🔢 Layers","⚡ Activation","🔄 Iters"],
                                [f"{acc:.1%}",str(len(layers)),act,str(max_it)]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        plot_boundary(ax,clf,Xs,y,Xtr,ytr,Xte,yte,f1,f2)
        ax.set_title(f'MLP {list(layers)} | {act} | Acc={acc:.1%}',color='#4fc3f7',fontsize=12,fontweight='bold')
        st.pyplot(fig); plt.close()

        if hasattr(clf,'loss_curve_'):
            st.markdown("### 📉 Training Loss Curve")
            fig,ax=plt.subplots(figsize=(8,3.5)); style_ax(ax,fig)
            ax.plot(clf.loss_curve_,color='#4fc3f7',lw=2)
            ax.set_xlabel("Iteration",color='#8892b0'); ax.set_ylabel("Loss",color='#8892b0')
            ax.set_title("Loss During Training",color='#4fc3f7',fontsize=11,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); plot_confusion(ax,fig,yte,yp); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(pd.DataFrame(classification_report(yte,yp,output_dict=True)).T.round(3),
                         use_container_width=True,height=220)

# ══════════════════════════════════════════════════════════════
# ██  AUTOENCODER
# ══════════════════════════════════════════════════════════════
elif algo == "🔁 AutoEncoder":
    st.markdown('<div class="main-header"><h1>🔁 AutoEncoder</h1><p>Compress data to its essence, then reconstruct it — learn without labels</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### AutoEncoder Settings")

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin-bottom:4px;">🗜️ <b>Latent Dimension</b> — the bottleneck size.<br>Smaller = more compression, harder to reconstruct. Larger = easier but less interesting.</div>', unsafe_allow_html=True)
        latent_dim  = st.slider("Latent Dimension", 1, 16, 2, 1)

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin:6px 0 4px;">🧱 <b>Hidden Layer Size</b> — neurons in encoder/decoder layers.<br>More neurons = more capacity to learn complex patterns.</div>', unsafe_allow_html=True)
        hidden_size = st.slider("Hidden Layer Size", 8, 128, 32, 8)

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin:6px 0 4px;">🔊 <b>Input Noise</b> — adds random noise to the input.<br>0 = normal AE. >0 = Denoising AE (must reconstruct clean signal from noisy input).</div>', unsafe_allow_html=True)
        noise_level = st.slider("Input Noise (Denoising)", 0.0, 1.0, 0.0, 0.05)

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin:6px 0 4px;">🔄 <b>Epochs</b> — how many full passes through the data.<br>More = better learning, but slower. Stop when loss flattens.</div>', unsafe_allow_html=True)
        n_epochs    = st.slider("Epochs", 50, 500, 200, 50)

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin:6px 0 4px;">⚡ <b>Learning Rate</b> — how big each weight update step is.<br>Too high = unstable. Too low = very slow. 0.01 is a safe default.</div>', unsafe_allow_html=True)
        lr_ae       = st.select_slider("Learning Rate", [0.0001,0.001,0.01,0.1], value=0.01)

        st.markdown("---")
        st.markdown("### Data")
        ae_dataset  = st.selectbox("Dataset", ["Digits (MNIST-like)", "Iris", "Blobs"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Concept", "🗜️ Compression", "🎨 Reconstruction", "🔍 Latent Space"
    ])

    # ── Pure-numpy AutoEncoder ───────────────────────────────────
    @st.cache_data
    def get_ae_data(name):
        if name == "Digits (MNIST-like)":
            d = datasets.load_digits()
            X = d.data.astype(np.float32) / 16.0   # 64 features
            y = d.target
        elif name == "Iris":
            d = datasets.load_iris()
            X = d.data.astype(np.float32)
            y = d.target
        else:
            X, y = datasets.make_blobs(300, n_features=4, centers=3, random_state=42)
            X = X.astype(np.float32)
        sc = StandardScaler()
        X = sc.fit_transform(X).astype(np.float32)
        return X, y

    def relu(x):   return np.maximum(0, x)
    def relu_d(x): return (x > 0).astype(float)
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def ae_forward(X, W1, b1, W2, b2, W3, b3, W4, b4):
        # Encoder
        z1 = X @ W1 + b1;        a1 = relu(z1)
        z2 = a1 @ W2 + b2;       a2 = relu(z2)       # latent
        # Decoder
        z3 = a2 @ W3 + b3;       a3 = relu(z3)
        z4 = a3 @ W4 + b4;       out = z4             # linear output
        return z1, a1, z2, a2, z3, a3, out

    def ae_loss(X_out, X_target):
        return np.mean((X_out - X_target)**2)

    @st.cache_data
    def train_ae(X, latent_dim, hidden_size, noise_level, n_epochs, lr):
        np.random.seed(42)
        n_in = X.shape[1]
        W1 = np.random.randn(n_in, hidden_size)      * 0.1
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, latent_dim) * 0.1
        b2 = np.zeros((1, latent_dim))
        W3 = np.random.randn(latent_dim, hidden_size) * 0.1
        b3 = np.zeros((1, hidden_size))
        W4 = np.random.randn(hidden_size, n_in)       * 0.1
        b4 = np.zeros((1, n_in))

        losses = []
        batch = min(64, len(X))

        for epoch in range(n_epochs):
            idx = np.random.permutation(len(X))
            epoch_loss = 0
            for i in range(0, len(X), batch):
                Xb = X[idx[i:i+batch]]
                Xb_noisy = Xb + np.random.randn(*Xb.shape) * noise_level

                z1,a1,z2,a2,z3,a3,out = ae_forward(Xb_noisy, W1,b1,W2,b2,W3,b3,W4,b4)
                loss = ae_loss(out, Xb)
                epoch_loss += loss

                # Backprop
                dL = 2*(out - Xb) / Xb.shape[0]
                # Decoder
                dW4 = a3.T @ dL;          db4 = dL.sum(0, keepdims=True)
                da3 = dL @ W4.T;          dz3 = da3 * relu_d(z3)
                dW3 = a2.T @ dz3;         db3 = dz3.sum(0, keepdims=True)
                # Encoder
                da2 = dz3 @ W3.T;         dz2 = da2 * relu_d(z2)
                dW2 = a1.T @ dz2;         db2 = dz2.sum(0, keepdims=True)
                da1 = dz2 @ W2.T;         dz1 = da1 * relu_d(z1)
                dW1 = Xb_noisy.T @ dz1;   db1 = dz1.sum(0, keepdims=True)

                # Gradient clip
                for g in [dW1,dW2,dW3,dW4,db1,db2,db3,db4]:
                    np.clip(g, -1, 1, out=g)

                W1-=lr*dW1; b1-=lr*db1
                W2-=lr*dW2; b2-=lr*db2
                W3-=lr*dW3; b3-=lr*db3
                W4-=lr*dW4; b4-=lr*db4

            losses.append(epoch_loss)

        return (W1,b1,W2,b2,W3,b3,W4,b4), losses

    X_ae, y_ae = get_ae_data(ae_dataset)

    with st.spinner("Training AutoEncoder..."):
        weights, losses = train_ae(X_ae, latent_dim, hidden_size, noise_level, n_epochs, lr_ae)

    W1,b1,W2,b2,W3,b3,W4,b4 = weights
    _,_,_,latent,_,_,reconstructed = ae_forward(X_ae, W1,b1,W2,b2,W3,b3,W4,b4)
    recon_loss = ae_loss(reconstructed, X_ae)

    # ── TAB 1: Concept ───────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
<div class="card"><h3>🔁 What is an AutoEncoder?</h3>
<p>An AutoEncoder is a neural network that learns to <strong>compress data and then reconstruct it</strong>. The goal is: output ≈ input.</p>
<p>It's trained with <strong>no labels</strong> — the input itself is the target. By forcing data through a tiny bottleneck, it must learn the most important patterns to survive compression.</p>
<div class="highlight">💡 Like squeezing a photo into a tiny file and then decompressing it — the network learns what's essential</div></div>

<div class="card"><h3>🏗️ Architecture: Encoder → Bottleneck → Decoder</h3><ul>
<li><strong>Encoder</strong>: takes the original data and compresses it layer by layer into fewer numbers</li>
<li><strong>Latent Space (Bottleneck)</strong>: the compressed representation — the "essence" of the data</li>
<li><strong>Decoder</strong>: takes the compressed code and reconstructs the original data</li></ul>
<div class="highlight">💡 Smaller latent dim = more compression = the network must learn smarter</div></div>

<div class="card"><h3>🎯 What Can AutoEncoders Do?</h3><ul>
<li><strong>Dimensionality Reduction</strong>: compress high-dimensional data (like images) into 2-3 numbers for visualization</li>
<li><strong>Denoising</strong>: input noisy data, output clean data — the model learns to ignore noise</li>
<li><strong>Anomaly Detection</strong>: the model reconstructs normal data well but fails on anomalies — high reconstruction error = anomaly!</li>
<li><strong>Generative Models</strong>: VAEs extend AEs to generate new data (images, text, etc.)</li></ul>
<div class="highlight">💡 If reconstruction error is very high for a sample → it's unusual!</div></div>

<div class="card"><h3>📉 How Does it Learn?</h3>
<p>The loss function is simply: <strong>Reconstruction Error = mean((output - input)²)</strong></p>
<p>Backpropagation adjusts all weights to minimize this error. No labels, no classes — just "make your output look like your input."</p>
<div class="highlight">💡 Self-supervised learning: the data teaches itself</div></div>

<div class="card"><h3>🔊 Denoising AutoEncoder</h3>
<p>Add noise to the input, but train the model to reconstruct the <strong>clean</strong> original. This forces the model to learn robust, noise-resistant features.</p>
<div class="highlight">💡 Try sliding the Noise slider in the sidebar to see the effect!</div></div>
""", unsafe_allow_html=True)

        with c2:
            # Architecture diagram
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0a0f1e'); ax.set_facecolor('#0a0f1e'); ax.axis('off')
            n_in = X_ae.shape[1]
            arch = [min(n_in, 8), hidden_size, latent_dim, hidden_size, min(n_in, 8)]
            labels_arch = ['Input', 'Encoder\nHidden', f'Latent\n(dim={latent_dim})', 'Decoder\nHidden', 'Output']
            colors_arch = ['#4fc3f7', '#1e88e5', '#00e5ff', '#1e88e5', '#4fc3f7']
            xs = np.linspace(0.05, 0.95, 5)
            max_n = max(arch)
            node_pos = {}
            for li, (n, xp) in enumerate(zip(arch, xs)):
                ys = np.linspace(0.5 - (n-1)*0.07, 0.5 + (n-1)*0.07, n)
                for ni, yp in enumerate(ys):
                    node_pos[(li, ni)] = (xp, yp)
                    circ = plt.Circle((xp, yp), 0.022, color=colors_arch[li],
                                      ec='white', lw=0.8, zorder=5, alpha=0.9)
                    ax.add_patch(circ)
            for li in range(4):
                for ni in range(arch[li]):
                    for nj in range(arch[li+1]):
                        x1,y1 = node_pos[(li,ni)]; x2,y2 = node_pos[(li+1,nj)]
                        ax.plot([x1,x2],[y1,y2], color='#1e3a5f', lw=0.5, alpha=0.4, zorder=1)
            for li,(lbl,xp) in enumerate(zip(labels_arch, xs)):
                ax.text(xp, 0.06, lbl, ha='center', va='center',
                        color='#8892b0', fontsize=7.5, fontfamily='monospace')
                ax.text(xp, 0.96, str(arch[li]), ha='center', va='center',
                        color=colors_arch[li], fontsize=9, fontweight='bold', fontfamily='monospace')
            # Bottleneck arrow
            ax.annotate('', xy=(xs[2], 0.5), xytext=(xs[1]+0.04, 0.5),
                        arrowprops=dict(arrowstyle='->', color='#00e5ff', lw=1.5))
            ax.annotate('', xy=(xs[3]-0.04, 0.5), xytext=(xs[2], 0.5),
                        arrowprops=dict(arrowstyle='->', color='#00e5ff', lw=1.5))
            ax.text(xs[2], 0.78, '🔒 Bottleneck', ha='center', color='#00e5ff',
                    fontsize=9, fontfamily='monospace', fontweight='bold')
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_title('AutoEncoder Architecture', color='#4fc3f7', fontsize=12, fontweight='bold')
            st.pyplot(fig); plt.close()

            # Loss curve
            fig, ax = plt.subplots(figsize=(8, 3))
            style_ax(ax, fig)
            ax.plot(losses, color='#4fc3f7', lw=2)
            ax.set_xlabel("Epoch", color='#8892b0')
            ax.set_ylabel("Reconstruction Loss", color='#8892b0')
            ax.set_title("Training Loss Curve", color='#4fc3f7', fontsize=11, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TAB 2: Compression ───────────────────────────────────────
    with tab2:
        m1, m2, m3, m4 = st.columns(4)
        compression = X_ae.shape[1] / latent_dim
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["📥 Input Dim","🗜️ Latent Dim","📉 Recon Loss","📦 Compression"],
                                [str(X_ae.shape[1]), str(latent_dim), f"{recon_loss:.4f}", f"{compression:.1f}x"]):
            with col: st.markdown(metric_box(val, lbl), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Original vs Reconstructed for first 8 samples
        if ae_dataset == "Digits (MNIST-like)":
            st.markdown("### 🖼️ Original vs Reconstructed Digits")
            n_show = 8
            fig, axes = plt.subplots(2, n_show, figsize=(12, 3.5))
            fig.patch.set_facecolor('#0a0f1e')
            for i in range(n_show):
                for row, data, title in zip([0,1],[X_ae[i], reconstructed[i]], ['Original','Reconstructed']):
                    axes[row,i].imshow(data.reshape(8,8), cmap='Blues', vmin=-2, vmax=2)
                    axes[row,i].axis('off')
                    if i == 0: axes[row,i].set_ylabel(title, color='#8892b0', fontsize=9)
            fig.suptitle(f'Original vs Reconstructed  |  Latent dim={latent_dim}  |  Loss={recon_loss:.4f}',
                         color='#4fc3f7', fontsize=11, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            # Feature-level reconstruction comparison
            st.markdown("### 📊 Original vs Reconstructed (Feature Values)")
            n_show = min(30, len(X_ae))
            fig, axes = plt.subplots(1, X_ae.shape[1], figsize=(12, 3.5))
            fig.patch.set_facecolor('#0a0f1e')
            if X_ae.shape[1] == 1: axes = [axes]
            for fi, ax in enumerate(axes):
                style_ax(ax)
                ax.scatter(range(n_show), X_ae[:n_show, fi], c='#4fc3f7', s=20, alpha=0.8, label='Original')
                ax.scatter(range(n_show), reconstructed[:n_show, fi], c='#ef5350', s=20, alpha=0.8, marker='x', label='Reconstructed')
                ax.set_title(f'Feature {fi+1}', color='#4fc3f7', fontsize=9, fontweight='bold')
                if fi == 0: ax.legend(fontsize=7, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Reconstruction error per sample
        st.markdown("### 🔍 Reconstruction Error per Sample (Anomaly Detection)")
        per_sample_err = np.mean((reconstructed - X_ae)**2, axis=1)
        threshold = np.percentile(per_sample_err, 90)
        fig, ax = plt.subplots(figsize=(10, 3))
        style_ax(ax, fig)
        colors_bar = ['#ef5350' if e > threshold else '#4fc3f7' for e in per_sample_err]
        ax.bar(range(len(per_sample_err)), per_sample_err, color=colors_bar, alpha=0.8, width=1.0)
        ax.axhline(threshold, color='#00e5ff', lw=2, ls='--', label=f'90th percentile threshold = {threshold:.4f}')
        ax.set_xlabel("Sample Index", color='#8892b0')
        ax.set_ylabel("Reconstruction Error", color='#8892b0')
        ax.set_title("Per-Sample Reconstruction Error  |  Red = Potential Anomaly",
                     color='#4fc3f7', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.info(f"🔴 **{sum(per_sample_err > threshold)}** samples above threshold — potential anomalies")

    # ── TAB 3: Reconstruction ────────────────────────────────────
    with tab3:
        st.markdown("### 🔊 Denoising AutoEncoder Demo")
        if noise_level == 0:
            st.info("💡 Set **Input Noise > 0** in the sidebar to see denoising in action!")
        else:
            st.success(f"✅ Denoising mode active — noise level: **{noise_level:.2f}**")

        if ae_dataset == "Digits (MNIST-like)":
            n_show = 8
            noisy_X = X_ae[:n_show] + np.random.randn(n_show, X_ae.shape[1]) * noise_level
            _,_,_,_,_,_,denoised = ae_forward(noisy_X, W1,b1,W2,b2,W3,b3,W4,b4)
            fig, axes = plt.subplots(3, n_show, figsize=(12, 5))
            fig.patch.set_facecolor('#0a0f1e')
            for i in range(n_show):
                for row, data, ttl in zip([0,1,2],
                                           [X_ae[i], noisy_X[i], denoised[i]],
                                           ['Clean','Noisy','Denoised']):
                    axes[row,i].imshow(data.reshape(8,8), cmap='Blues', vmin=-2, vmax=2)
                    axes[row,i].axis('off')
                    if i == 0: axes[row,i].set_ylabel(ttl, color='#8892b0', fontsize=9)
            fig.suptitle(f'Denoising AutoEncoder | Noise={noise_level:.2f}',
                         color='#4fc3f7', fontsize=11, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.markdown("**Note:** Switch to **Digits** dataset to see the best denoising visualization.")
            n_show = min(40, len(X_ae))
            noisy_X = X_ae[:n_show] + np.random.randn(n_show, X_ae.shape[1]) * noise_level
            _,_,_,_,_,_,denoised = ae_forward(noisy_X, W1,b1,W2,b2,W3,b3,W4,b4)
            fig, ax = plt.subplots(figsize=(10,4)); style_ax(ax, fig)
            ax.plot(X_ae[:n_show,0], label='Clean', color='#4fc3f7', lw=2)
            ax.plot(noisy_X[:,0], label='Noisy', color='#ef5350', lw=1, alpha=0.7)
            ax.plot(denoised[:,0], label='Denoised', color='#00e5ff', lw=2, ls='--')
            ax.set_title('Feature 0: Clean vs Noisy vs Denoised', color='#4fc3f7', fontsize=11, fontweight='bold')
            ax.legend(facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TAB 4: Latent Space ──────────────────────────────────────
    with tab4:
        st.markdown("### 🔍 Latent Space Visualization")
        st.markdown("The latent space is the compressed representation learned by the encoder. "
                    "Similar data points should cluster together even without any labels!")

        if latent_dim >= 2:
            fig, ax = plt.subplots(figsize=(8, 6)); style_ax(ax, fig)
            n_cls = len(np.unique(y_ae))
            for i in range(n_cls):
                mask = y_ae == i
                ax.scatter(latent[mask, 0], latent[mask, 1],
                           c=COLORS[i % len(COLORS)], s=40, alpha=0.8,
                           edgecolors='white', lw=0.3, label=f'Class {i}')
            ax.set_xlabel("Latent Dim 1", color='#8892b0')
            ax.set_ylabel("Latent Dim 2", color='#8892b0')
            ax.set_title(f'Latent Space (2D projection) — {ae_dataset}',
                         color='#4fc3f7', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
            st.pyplot(fig); plt.close()
            st.info("💡 Even though the AutoEncoder was trained **without class labels**, "
                    "similar classes naturally cluster in latent space!")
        else:
            # 1D latent: show as strip
            fig, ax = plt.subplots(figsize=(10, 3)); style_ax(ax, fig)
            n_cls = len(np.unique(y_ae))
            for i in range(n_cls):
                mask = y_ae == i
                ax.scatter(latent[mask, 0], np.zeros(mask.sum()) + i*0.1,
                           c=COLORS[i % len(COLORS)], s=40, alpha=0.8,
                           edgecolors='white', lw=0.3, label=f'Class {i}')
            ax.set_xlabel("Latent Value", color='#8892b0')
            ax.set_title("1D Latent Space", color='#4fc3f7', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
            st.pyplot(fig); plt.close()
            st.info("💡 Increase Latent Dim to 2+ to see a 2D scatter plot of the latent space.")

# ══════════════════════════════════════════════════════════════
# ██  VAE — Variational AutoEncoder
# ══════════════════════════════════════════════════════════════
elif algo == "🎲 VAE (Variational)":
    st.markdown('<div class="main-header"><h1>🎲 Variational AutoEncoder (VAE)</h1><p>Learn a smooth, structured latent space — and generate brand-new data</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### VAE Settings")

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin-bottom:4px;">🗜️ <b>Latent Dimension</b> — size of the learned distribution.<br>2D is perfect for visualization. Higher = more expressive but harder to visualize.</div>', unsafe_allow_html=True)
        vae_latent = st.slider("Latent Dimension", 1, 8, 2, 1, key="vae_lat")

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin:6px 0 4px;">🧱 <b>Hidden Size</b> — neurons in encoder/decoder hidden layers.<br>Larger = model can learn richer patterns, but risks overfitting small datasets.</div>', unsafe_allow_html=True)
        vae_hidden = st.slider("Hidden Size", 8, 128, 32, 8, key="vae_hid")

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin:6px 0 4px;">⚖️ <b>Beta (KL weight)</b> — balances reconstruction vs. regularity.<br>β=1 is standard VAE. Higher β = smoother/more disentangled latent space, but worse reconstruction. Lower β = sharper outputs but messier latent space.</div>', unsafe_allow_html=True)
        vae_beta   = st.slider("Beta (KL weight) β", 0.0, 5.0, 1.0, 0.1, key="vae_beta")

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin:6px 0 4px;">🔄 <b>Epochs</b> — training passes through all data.<br>Watch the loss curve — stop when it flattens. More data needs more epochs.</div>', unsafe_allow_html=True)
        vae_epochs = st.slider("Epochs", 50, 600, 250, 50, key="vae_ep")

        st.markdown('<div style="color:#8892b0;font-size:0.75rem;margin:6px 0 4px;">⚡ <b>Learning Rate</b> — gradient step size.<br>0.001–0.01 works best for VAEs. Too high → NaN loss. Too low → very slow convergence.</div>', unsafe_allow_html=True)
        vae_lr     = st.select_slider("Learning Rate", [0.0001,0.001,0.005,0.01,0.05], value=0.005, key="vae_lr")

        st.markdown("---")
        st.markdown("### Data")
        vae_ds = st.selectbox("Dataset", ["Digits (MNIST-like)", "Iris", "Blobs"], key="vae_ds")

    # ── Data ──────────────────────────────────────────────────
    @st.cache_data
    def get_vae_data(name):
        if name == "Digits (MNIST-like)":
            d = datasets.load_digits()
            X = d.data.astype(np.float32) / 16.0
            y = d.target
        elif name == "Iris":
            d = datasets.load_iris()
            X = d.data.astype(np.float32); y = d.target
        else:
            X, y = datasets.make_blobs(300, n_features=4, centers=3, random_state=42)
            X = X.astype(np.float32)
        sc = StandardScaler()
        X = sc.fit_transform(X).astype(np.float32)
        return X, y

    # ── Pure-numpy VAE ────────────────────────────────────────
    def relu(x):    return np.maximum(0, x)
    def relu_d(x):  return (x > 0).astype(float)

    @st.cache_data
    def train_vae(X, latent_dim, hidden, beta, epochs, lr):
        np.random.seed(42)
        n = X.shape[1]
        # Encoder weights → μ and log σ²
        We1 = np.random.randn(n,      hidden)     * np.sqrt(2/n)
        be1 = np.zeros((1, hidden))
        Wmu = np.random.randn(hidden, latent_dim) * 0.01
        bmu = np.zeros((1, latent_dim))
        Wlv = np.random.randn(hidden, latent_dim) * 0.01
        blv = np.zeros((1, latent_dim))
        # Decoder weights
        Wd1 = np.random.randn(latent_dim, hidden) * np.sqrt(2/latent_dim)
        bd1 = np.zeros((1, hidden))
        Wo  = np.random.randn(hidden, n)           * 0.01
        bo  = np.zeros((1, n))

        losses, recon_losses, kl_losses = [], [], []
        batch = min(64, len(X))

        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            ep_loss = ep_recon = ep_kl = 0
            for i in range(0, len(X), batch):
                Xb = X[idx[i:i+batch]]
                B  = len(Xb)

                # ── Encode ──
                h1  = relu(Xb @ We1 + be1)         # (B, hidden)
                mu  = h1 @ Wmu + bmu               # (B, latent)
                lv  = np.clip(h1 @ Wlv + blv, -4, 4)  # log variance
                std = np.exp(0.5 * lv)
                eps = np.random.randn(B, latent_dim)
                z   = mu + std * eps               # reparameterization

                # ── Decode ──
                h2  = relu(z @ Wd1 + bd1)
                xr  = h2 @ Wo + bo                 # reconstruction

                # ── Loss ──
                recon = np.mean((xr - Xb)**2)
                kl    = -0.5 * np.mean(1 + lv - mu**2 - np.exp(lv))
                loss  = recon + beta * kl
                ep_loss += loss; ep_recon += recon; ep_kl += kl

                # ── Backprop decoder ──
                dxr  = 2*(xr - Xb) / (B * n)
                dWo  = h2.T @ dxr;          dbo  = dxr.sum(0, keepdims=True)
                dh2  = dxr @ Wo.T * relu_d(z @ Wd1 + bd1)
                dWd1 = z.T @ dh2;           dbd1 = dh2.sum(0, keepdims=True)
                dz   = dh2 @ Wd1.T

                # ── Backprop reparameterization ──
                dmu_recon = dz.copy()
                dlv_recon = dz * eps * 0.5 * std

                # ── KL gradients ──
                dmu_kl = beta * mu / (B * latent_dim)
                dlv_kl = beta * 0.5 * (np.exp(lv) - 1) / (B * latent_dim)

                dmu = dmu_recon + dmu_kl
                dlv = np.clip(dlv_recon + dlv_kl, -1, 1)

                # ── Backprop encoder ──
                dh1_mu  = dmu @ Wmu.T * relu_d(Xb @ We1 + be1)
                dh1_lv  = dlv @ Wlv.T * relu_d(Xb @ We1 + be1)
                dh1     = dh1_mu + dh1_lv

                dWmu = np.clip(h1.T @ dmu,  -1, 1); dbmu = np.clip(dmu.sum(0, keepdims=True),  -1, 1)
                dWlv = np.clip(h1.T @ dlv,  -1, 1); dblv = np.clip(dlv.sum(0, keepdims=True),  -1, 1)
                dWe1 = np.clip(Xb.T @ dh1,  -1, 1); dbe1 = np.clip(dh1.sum(0, keepdims=True),  -1, 1)

                We1-=lr*dWe1; be1-=lr*dbe1
                Wmu-=lr*dWmu; bmu-=lr*dbmu
                Wlv-=lr*dWlv; blv-=lr*dblv
                Wd1-=lr*dWd1; bd1-=lr*dbd1
                Wo -=lr*dWo;  bo -=lr*dbo

            losses.append(ep_loss)
            recon_losses.append(ep_recon)
            kl_losses.append(ep_kl)

        return (We1,be1,Wmu,bmu,Wlv,blv,Wd1,bd1,Wo,bo), losses, recon_losses, kl_losses

    def vae_encode(X, We1,be1,Wmu,bmu,Wlv,blv):
        h1  = relu(X @ We1 + be1)
        mu  = h1 @ Wmu + bmu
        lv  = np.clip(h1 @ Wlv + blv, -4, 4)
        return mu, lv

    def vae_decode(z, Wd1,bd1,Wo,bo):
        h2  = relu(z @ Wd1 + bd1)
        return h2 @ Wo + bo

    X_vae, y_vae = get_vae_data(vae_ds)

    with st.spinner("Training VAE... (pure numpy, no PyTorch needed)"):
        vae_w, losses_total, losses_recon, losses_kl = train_vae(
            X_vae, vae_latent, vae_hidden, vae_beta, vae_epochs, vae_lr
        )

    We1,be1,Wmu,bmu,Wlv,blv,Wd1,bd1,Wo,bo = vae_w
    mu_all, lv_all = vae_encode(X_vae, We1,be1,Wmu,bmu,Wlv,blv)
    recon_all = vae_decode(mu_all, Wd1,bd1,Wo,bo)
    recon_loss_final = np.mean((recon_all - X_vae)**2)
    kl_final         = -0.5 * np.mean(1 + lv_all - mu_all**2 - np.exp(lv_all))

    # ── TABS ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 VAE vs AE", "📉 Training Curves", "🎲 Generate Samples", "🗺️ Latent Space"
    ])

    # ── TAB 1: Concept ────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("""
<div class="card"><h3>🤔 What's Wrong with a Regular AutoEncoder?</h3>
<p>A regular AE learns a latent space, but it can be <strong>messy and full of holes</strong>. If you pick a random point in the latent space and decode it, you often get garbage — because the AE never learned what should live between the clusters.</p>
<div class="highlight">💡 Regular AE: compress → reconstruct. That's it. No structure in the latent space.</div></div>

<div class="card"><h3>🎲 The VAE Idea: Learn a Distribution, Not a Point</h3>
<p>Instead of encoding each input to a single point, a VAE encodes it to a <strong>probability distribution</strong> (a Gaussian bell curve) — described by two values:</p>
<ul>
<li><strong>μ (mu)</strong>: the mean — "where the point should be"</li>
<li><strong>σ (sigma)</strong>: the spread — "how uncertain we are"</li></ul>
<p>Then we sample from that distribution. This forces the latent space to be <strong>smooth and continuous</strong> — every point in it means something!</p>
<div class="highlight">💡 VAE: compress → distribution → sample → reconstruct. The randomness forces structure.</div></div>

<div class="card"><h3>🔀 The Reparameterization Trick</h3>
<p>Sampling is random, so how do we backpropagate through it? Simple trick: instead of sampling z directly, compute <strong>z = μ + σ × ε</strong> where ε is random noise drawn once.</p>
<p>Now gradients can flow through μ and σ, while ε is just a fixed constant for that step.</p>
<div class="highlight">💡 This is the key innovation that makes VAEs trainable!</div></div>

<div class="card"><h3>⚖️ The VAE Loss: Two Terms Fighting Each Other</h3>
<p><strong>Total Loss = Reconstruction Loss + β × KL Divergence</strong></p>
<ul>
<li><strong>Reconstruction Loss</strong>: "make your output look like your input" — same as AE</li>
<li><strong>KL Divergence</strong>: "keep your distributions close to a standard Gaussian" — forces regularity</li>
<li><strong>β (Beta)</strong>: how much you care about the KL term vs reconstruction</li></ul>
<p>These two terms are in tension — reconstruction wants freedom, KL wants order. The balance creates a useful latent space.</p>
<div class="highlight">💡 High β → very organized latent space but blurry outputs. Low β → sharp outputs but messy latent space.</div></div>

<div class="card"><h3>🆚 AE vs VAE — Quick Comparison</h3>
<table style="width:100%;font-size:0.85rem;color:#a8b2d8;border-collapse:collapse;">
<tr><th style="color:#4fc3f7;text-align:left;padding:4px;">Feature</th><th style="color:#4fc3f7;text-align:left;padding:4px;">AutoEncoder</th><th style="color:#4fc3f7;text-align:left;padding:4px;">VAE</th></tr>
<tr><td style="padding:4px;">Latent</td><td style="padding:4px;">Fixed point</td><td style="padding:4px;">Distribution (μ, σ)</td></tr>
<tr><td style="padding:4px;">Can Generate?</td><td style="padding:4px;">❌ No (gaps are noise)</td><td style="padding:4px;">✅ Yes (smooth space)</td></tr>
<tr><td style="padding:4px;">Latent Space</td><td style="padding:4px;">Unstructured</td><td style="padding:4px;">Smooth & continuous</td></tr>
<tr><td style="padding:4px;">Loss</td><td style="padding:4px;">Reconstruction only</td><td style="padding:4px;">Recon + KL</td></tr>
<tr><td style="padding:4px;">Best for</td><td style="padding:4px;">Compression, denoising</td><td style="padding:4px;">Generation, interpolation</td></tr>
</table></div>
""", unsafe_allow_html=True)

        with c2:
            # Architecture diagram VAE
            fig, ax = plt.subplots(figsize=(8,5.5))
            fig.patch.set_facecolor('#0a0f1e'); ax.set_facecolor('#0a0f1e'); ax.axis('off')
            n_in = X_vae.shape[1]
            # Draw blocks instead of circles for clarity
            sections = [
                (0.05, 0.22, '#1e3a5f',  '#4fc3f7', f'Encoder\n({min(n_in,8)}→{vae_hidden})'),
                (0.27, 0.44, '#0d2137',  '#ab47bc', f'μ  &  log σ²\n(dim={vae_latent})'),
                (0.50, 0.50, '#0a0f1e',  '#00e5ff', 'Sample\nz = μ+σε'),
                (0.56, 0.73, '#1e3a5f',  '#4fc3f7', f'Decoder\n({vae_latent}→{vae_hidden})'),
                (0.78, 0.95, '#0d2137',  '#4fc3f7', f'Output\n(dim={min(n_in,8)})'),
            ]
            for x0, x1, fc, ec, label in sections:
                rect = plt.Rectangle((x0, 0.3), x1-x0, 0.4, fc=fc, ec=ec, lw=1.5, zorder=3)
                ax.add_patch(rect)
                ax.text((x0+x1)/2, 0.5, label, ha='center', va='center',
                        color='white', fontsize=8.5, fontfamily='monospace', fontweight='bold', zorder=5)

            # Arrows between sections
            for xs, xe, col in [(0.22,0.27,'#4fc3f7'),(0.44,0.5,'#ab47bc'),
                                  (0.5,0.56,'#00e5ff'),(0.73,0.78,'#4fc3f7')]:
                ax.annotate('', xy=(xe, 0.5), xytext=(xs, 0.5),
                            arrowprops=dict(arrowstyle='->', color=col, lw=2), zorder=6)

            # Labels above/below
            ax.text(0.35, 0.78, 'Bottleneck\n(Distribution)', ha='center', color='#ab47bc',
                    fontsize=8, fontfamily='monospace')
            ax.text(0.5, 0.16, 'ε ~ N(0,1)', ha='center', color='#00e5ff',
                    fontsize=8.5, fontfamily='monospace')
            ax.text(0.05, 0.82, 'Input x', ha='left', color='#8892b0', fontsize=8, fontfamily='monospace')
            ax.text(0.82, 0.82, 'x̂ ≈ x', ha='left', color='#8892b0', fontsize=8, fontfamily='monospace')
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_title('VAE Architecture', color='#4fc3f7', fontsize=12, fontweight='bold')
            st.pyplot(fig); plt.close()

            # Show loss formula
            st.markdown(f"""
<div class="card" style="margin-top:1rem;">
<h3>📐 Current Loss Breakdown</h3>
<table style="width:100%;font-size:0.9rem;color:#a8b2d8;">
<tr><td>🔵 Reconstruction Loss</td><td style="color:#4fc3f7;font-family:monospace;">{losses_recon[-1]:.4f}</td></tr>
<tr><td>🟣 KL Divergence × β={vae_beta}</td><td style="color:#ab47bc;font-family:monospace;">{vae_beta * losses_kl[-1]:.4f}</td></tr>
<tr><td><b>Total Loss</b></td><td style="color:#00e5ff;font-family:monospace;"><b>{losses_total[-1]:.4f}</b></td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    # ── TAB 2: Training Curves ────────────────────────────────
    with tab2:
        m1,m2,m3,m4 = st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
            ["📉 Final Loss","🔵 Recon Loss","🟣 KL Loss","🗜️ Latent Dim"],
            [f"{losses_total[-1]:.4f}", f"{recon_loss_final:.4f}", f"{kl_final:.4f}", str(vae_latent)]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor('#0a0f1e')
        for ax in axes: style_ax(ax)

        axes[0].plot(losses_total,  color='#00e5ff', lw=2, label='Total')
        axes[0].set_title("Total Loss",       color='#4fc3f7', fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Epoch", color='#8892b0')

        axes[1].plot(losses_recon,  color='#4fc3f7', lw=2)
        axes[1].set_title("Reconstruction Loss", color='#4fc3f7', fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Epoch", color='#8892b0')

        axes[2].plot(losses_kl,     color='#ab47bc', lw=2)
        axes[2].set_title("KL Divergence",    color='#4fc3f7', fontsize=11, fontweight='bold')
        axes[2].set_xlabel("Epoch", color='#8892b0')

        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("""
<div class="card">
<h3>📖 How to Read These Curves</h3>
<ul>
<li><strong>Total Loss</strong> should steadily decrease and flatten — if it's still falling sharply, train more epochs</li>
<li><strong>Reconstruction Loss</strong> going down = the decoder is getting better at rebuilding inputs</li>
<li><strong>KL Divergence</strong> going up slightly then stabilizing is normal — it means the latent space is being regularized</li>
<li>If KL collapses to 0 → the model ignores the latent space (KL collapse) → try lowering β or increasing latent dim</li>
</ul>
</div>
""", unsafe_allow_html=True)

    # ── TAB 3: Generation ────────────────────────────────────
    with tab3:
        st.markdown("### 🎲 Generate New Samples from the Latent Space")
        st.markdown("""
<div class="card">
<h3>💡 How Generation Works</h3>
<p>Because the VAE forces the latent space to follow a standard Gaussian distribution N(0,1), we can <strong>sample random points from it</strong> and decode them into realistic new data.</p>
<p>This is impossible with a regular AE — sampling random points there gives garbage. With a VAE, the smooth latent space means every point decodes to something meaningful.</p>
</div>
""", unsafe_allow_html=True)

        n_gen = st.slider("Number of samples to generate", 4, 32, 16, 4)
        temperature = st.slider("Temperature (sampling spread)", 0.1, 2.0, 1.0, 0.1,
                                 help="Higher = more diverse but less accurate. Lower = conservative.")
        st.markdown('<div style="color:#8892b0;font-size:0.78rem;">⚙️ Temperature scales the noise — higher temp samples from a wider distribution → more variety, less fidelity.</div>', unsafe_allow_html=True)

        np.random.seed(st.session_state.get("gen_seed", 0))
        z_random = np.random.randn(n_gen, vae_latent) * temperature
        generated = vae_decode(z_random, Wd1, bd1, Wo, bo)

        if vae_ds == "Digits (MNIST-like)":
            cols_per_row = 8
            rows = (n_gen + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row*1.5, rows*1.8))
            fig.patch.set_facecolor('#0a0f1e')
            axes = np.array(axes).reshape(rows, cols_per_row)
            for i in range(rows * cols_per_row):
                ax = axes[i // cols_per_row, i % cols_per_row]
                ax.set_facecolor('#0a0f1e'); ax.axis('off')
                if i < n_gen:
                    ax.imshow(generated[i].reshape(8,8), cmap='Blues', vmin=-2, vmax=2)
                    ax.set_title(f"gen {i}", color='#8892b0', fontsize=6)
            fig.suptitle(f'VAE Generated Digits — Temperature={temperature:.1f}',
                         color='#4fc3f7', fontsize=11, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            fig, ax = plt.subplots(figsize=(8,4)); style_ax(ax, fig)
            ax.scatter(generated[:,0], generated[:,1], c='#ab47bc', s=80,
                       alpha=0.8, edgecolors='white', lw=0.5, label='Generated', marker='*', zorder=5)
            ax.scatter(X_vae[:,0], X_vae[:,1], c='#4fc3f7', s=20,
                       alpha=0.4, edgecolors='none', label='Real data')
            ax.set_title(f'Generated vs Real Samples | Temperature={temperature:.1f}',
                         color='#4fc3f7', fontsize=11, fontweight='bold')
            ax.legend(facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
            st.pyplot(fig); plt.close()

        if st.button("🎲 Resample"):
            import random
            st.session_state["gen_seed"] = random.randint(0, 9999)
            st.rerun()

    # ── TAB 4: Latent Space ───────────────────────────────────
    with tab4:
        st.markdown("### 🗺️ Latent Space Explorer")
        st.markdown("""
<div class="card">
<h3>🗺️ Why the Latent Space Matters</h3>
<p>The VAE's latent space is <strong>smooth and continuous</strong> — unlike a regular AE. This means:</p>
<ul>
<li>Similar inputs land in nearby regions</li>
<li>You can <strong>interpolate</strong> between two points and get meaningful transitions</li>
<li>The whole space is "filled" — no dead zones</li>
</ul>
<p>Below you can see the encoded positions of all training samples. Even without class labels during training, the VAE naturally clusters similar inputs together!</p>
</div>
""", unsafe_allow_html=True)

        if vae_latent >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)

            # Left: colored by class label
            n_cls = len(np.unique(y_vae))
            for i in range(n_cls):
                mask = y_vae == i
                axes[0].scatter(mu_all[mask,0], mu_all[mask,1],
                                c=COLORS[i%len(COLORS)], s=30, alpha=0.75,
                                edgecolors='white', lw=0.2, label=f'Class {i}')
            axes[0].set_title("Latent Space — colored by class", color='#4fc3f7', fontsize=11, fontweight='bold')
            axes[0].set_xlabel("z₁ (μ)", color='#8892b0'); axes[0].set_ylabel("z₂ (μ)", color='#8892b0')
            axes[0].legend(fontsize=8, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')

            # Right: colored by uncertainty (std)
            std_all = np.exp(0.5 * lv_all)
            uncertainty = std_all.mean(axis=1)
            sc = axes[1].scatter(mu_all[:,0], mu_all[:,1], c=uncertainty,
                                  cmap='plasma', s=30, alpha=0.8, edgecolors='none')
            plt.colorbar(sc, ax=axes[1], label='Mean σ (uncertainty)')
            axes[1].set_title("Latent Space — colored by uncertainty σ", color='#4fc3f7', fontsize=11, fontweight='bold')
            axes[1].set_xlabel("z₁ (μ)", color='#8892b0'); axes[1].set_ylabel("z₂ (μ)", color='#8892b0')

            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.info("💡 **Left**: classes naturally cluster — learned without labels! "
                    "**Right**: bright = uncertain encoding (near class boundaries)")

            # Interpolation between two classes
            st.markdown("### 🔀 Latent Space Interpolation")
            st.markdown("Pick two class centers and walk a straight line between them in latent space:")

            n_steps = 10
            c_avail = list(range(n_cls))
            ic1, ic2 = st.columns(2)
            with ic1: ca = st.selectbox("From class", c_avail, index=0, key="vae_ca")
            with ic2: cb = st.selectbox("To class",   c_avail, index=min(1,n_cls-1), key="vae_cb")

            mu_a = mu_all[y_vae==ca].mean(0)
            mu_b = mu_all[y_vae==cb].mean(0)
            alphas = np.linspace(0, 1, n_steps)
            z_interp = np.array([(1-a)*mu_a + a*mu_b for a in alphas])
            x_interp = vae_decode(z_interp, Wd1, bd1, Wo, bo)

            if vae_ds == "Digits (MNIST-like)":
                fig, axes2 = plt.subplots(1, n_steps, figsize=(14, 2))
                fig.patch.set_facecolor('#0a0f1e')
                for i, (ax, img) in enumerate(zip(axes2, x_interp)):
                    ax.imshow(img.reshape(8,8), cmap='Blues', vmin=-2, vmax=2)
                    ax.axis('off')
                    ax.set_title(f'{alphas[i]:.1f}', color='#8892b0', fontsize=6)
                fig.suptitle(f'Interpolation: Class {ca} → Class {cb}',
                             color='#4fc3f7', fontsize=10, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                fig, ax = plt.subplots(figsize=(8,3)); style_ax(ax, fig)
                ax.scatter(mu_all[y_vae==ca,0], mu_all[y_vae==ca,1], c=COLORS[ca%len(COLORS)], s=20, alpha=0.4)
                ax.scatter(mu_all[y_vae==cb,0], mu_all[y_vae==cb,1], c=COLORS[cb%len(COLORS)], s=20, alpha=0.4)
                ax.plot(z_interp[:,0], z_interp[:,1], 'o-', color='#00e5ff', lw=2.5,
                        markersize=8, label=f'Class {ca} → {cb}', zorder=5)
                ax.legend(facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
                ax.set_title(f'Interpolation path in latent space',
                             color='#4fc3f7', fontsize=11, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            fig, ax = plt.subplots(figsize=(10,3)); style_ax(ax, fig)
            n_cls = len(np.unique(y_vae))
            for i in range(n_cls):
                mask = y_vae == i
                ax.scatter(mu_all[mask,0], np.zeros(mask.sum()) + i*0.15,
                           c=COLORS[i%len(COLORS)], s=40, alpha=0.8, label=f'Class {i}')
            ax.set_title("1D Latent Space — set Latent Dim ≥ 2 for full visualization",
                         color='#4fc3f7', fontsize=11, fontweight='bold')
            ax.legend(facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
            plt.tight_layout(); st.pyplot(fig); plt.close()


st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#8892b0;font-family:'JetBrains Mono',monospace;font-size:0.75rem;padding:0.4rem 0;">
    ML Explorer &nbsp;·&nbsp; SVM · Regression · Decision Tree · K-Means · KNN · MLP · AutoEncoder &nbsp;·&nbsp; 🤖
</div>
""", unsafe_allow_html=True)
