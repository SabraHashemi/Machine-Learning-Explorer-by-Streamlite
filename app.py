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
         "🔁 AutoEncoder", "🎲 VAE (Variational)", "🔗 Contrastive Learning"],
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

elif algo == "🎲 VAE (Variational)":
    st.markdown('<div class="main-header"><h1>🎲 Variational AutoEncoder (VAE)</h1><p>Learn a structured latent space — compress, generate, and detect anomalies</p></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SIDEBAR — all controls here
    # ═══════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("### ⚙️ VAE Settings")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin-bottom:3px;">
        🗜️ <b>Latent Dimension</b><br>
        The bottleneck size. <b>2 is ideal</b> — you can visualize it as a 2D map.
        Smaller = more compression but harder to reconstruct.
        Larger = easier reconstruction but latent space gets crowded.</div>""", unsafe_allow_html=True)
        vae_latent = st.slider("Latent Dimension", 1, 8, 2, 1, key="vae_lat")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin:8px 0 3px;">
        🧱 <b>Hidden Layer Size</b><br>
        Neurons in the encoder/decoder middle layers.
        More neurons = more capacity. Start with 32–64.
        Too large for small datasets → overfitting.</div>""", unsafe_allow_html=True)
        vae_hidden = st.slider("Hidden Size", 8, 128, 32, 8, key="vae_hid")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin:8px 0 3px;">
        ⚖️ <b>Beta (β) — KL Weight</b><br>
        Balances two competing goals:<br>
        • <b>Low β</b>: sharp reconstructions, messy latent space<br>
        • <b>β = 1</b>: standard VAE — good balance<br>
        • <b>High β</b>: organized latent space, blurrier outputs<br>
        For anomaly detection, β=0.5–1 works best.</div>""", unsafe_allow_html=True)
        vae_beta = st.slider("Beta β", 0.0, 5.0, 1.0, 0.1, key="vae_beta")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin:8px 0 3px;">
        🔄 <b>Epochs</b><br>
        Training passes. Stop when the loss curve flattens.
        200–300 is usually enough for these datasets.</div>""", unsafe_allow_html=True)
        vae_epochs = st.slider("Epochs", 50, 500, 200, 50, key="vae_ep")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin:8px 0 3px;">
        ⚡ <b>Learning Rate</b><br>
        Step size per gradient update.<br>
        • Too high (0.05+): loss explodes or oscillates<br>
        • Too low (0.0001): very slow, may not converge<br>
        • <b>0.005 is a safe default</b> for VAEs</div>""", unsafe_allow_html=True)
        vae_lr = st.select_slider("Learning Rate", [0.0001,0.001,0.005,0.01,0.05], value=0.005, key="vae_lr")

        st.markdown("---")
        st.markdown("### 📊 Data Source")
        vae_src = st.radio("Source", ["📦 Built-in", "📂 Upload CSV"], key="vae_src")
        if vae_src == "📦 Built-in":
            vae_ds = st.selectbox("Dataset", ["Digits (MNIST-like)", "Iris", "Blobs"], key="vae_ds")
        else:
            vae_ds = "upload"

    # ═══════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════
    @st.cache_data
    def get_vae_builtin(name):
        if name == "Digits (MNIST-like)":
            d = datasets.load_digits()
            X, y = d.data.astype(np.float32), d.target
        elif name == "Iris":
            d = datasets.load_iris()
            X, y = d.data.astype(np.float32), d.target
        else:
            X, y = datasets.make_blobs(300, n_features=4, centers=3, random_state=42)
            X = X.astype(np.float32)
        sc = StandardScaler()
        X = sc.fit_transform(X).astype(np.float32)
        return X, y

    is_digits = False
    if vae_src == "📂 Upload CSV":
        st.markdown("### 📂 Upload Your CSV")
        st.markdown("""<div class="upload-hint">
        Upload any CSV with numeric columns. Pick 2+ feature columns and optionally a label column.<br>
        <b>For anomaly detection:</b> your label column should mark which rows are "normal" (0) and which are "anomaly" (1),
        OR pick a class column and choose which class = anomaly below.
        </div>""", unsafe_allow_html=True)

        up = st.file_uploader("Drop CSV here", type=["csv"], key="vae_csv")
        if up is None:
            st.info("⬆️ Upload a CSV to continue, or switch to Built-in in the sidebar.")
            st.stop()
        try:
            df_up = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read file: {e}"); st.stop()

        num_cols = df_up.select_dtypes(include=np.number).columns.tolist()
        all_cols  = df_up.columns.tolist()
        if len(num_cols) < 2:
            st.error("Need at least 2 numeric feature columns."); st.stop()

        st.markdown("#### Map your columns")
        pc1,pc2,pc3 = st.columns(3)
        with pc1:
            feat_cols = st.multiselect("Feature columns (pick 2+)", num_cols,
                                        default=num_cols[:min(4,len(num_cols))], key="vae_feats")
        with pc2:
            has_label = st.checkbox("Has label column?", value=True, key="vae_haslbl")
            label_col = st.selectbox("Label column", all_cols, index=len(all_cols)-1, key="vae_lbl") if has_label else None
        with pc3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"📊 {len(df_up)} rows · {len(feat_cols)} features selected")

        if len(feat_cols) < 2:
            st.error("Select at least 2 feature columns."); st.stop()

        valid = df_up[feat_cols + ([label_col] if label_col else [])].dropna()
        X_vae = StandardScaler().fit_transform(valid[feat_cols].values).astype(np.float32)
        if has_label and label_col:
            raw_y = valid[label_col].values
            uniq  = np.unique(raw_y)
            lmap  = {v:i for i,v in enumerate(uniq)}
            y_vae = np.array([lmap[v] for v in raw_y])
        else:
            y_vae = np.zeros(len(X_vae), dtype=int)

        with st.expander("👀 Data preview"):
            st.dataframe(valid.head(10), use_container_width=True)
    else:
        X_vae, y_vae = get_vae_builtin(vae_ds)
        is_digits = (vae_ds == "Digits (MNIST-like)")

    n_cls_vae = len(np.unique(y_vae))

    # ═══════════════════════════════════════════════════════════
    # NUMPY VAE — helpers
    # ═══════════════════════════════════════════════════════════
    def relu(x):    return np.maximum(0, x)
    def relu_d(x):  return (x > 0).astype(float)

    @st.cache_data
    def train_vae(X, latent_dim, hidden, beta, epochs, lr, seed=42):
        np.random.seed(seed)
        n  = X.shape[1]
        k  = latent_dim
        scale_e = np.sqrt(2.0/n); scale_d = np.sqrt(2.0/k)
        We1 = np.random.randn(n, hidden)*scale_e; be1 = np.zeros((1,hidden))
        Wmu = np.random.randn(hidden,k)*0.01;     bmu = np.zeros((1,k))
        Wlv = np.random.randn(hidden,k)*0.01;     blv = np.zeros((1,k))
        Wd1 = np.random.randn(k,hidden)*scale_d;  bd1 = np.zeros((1,hidden))
        Wo  = np.random.randn(hidden,n)*0.01;     bo  = np.zeros((1,n))

        losses, r_losses, kl_losses = [], [], []
        bs = min(64, len(X))

        for ep in range(epochs):
            idx = np.random.permutation(len(X))
            ep_l = ep_r = ep_k = 0.0
            for i in range(0, len(X), bs):
                Xb = X[idx[i:i+bs]]; B = len(Xb)
                # encode
                h1  = relu(Xb@We1+be1)
                mu  = h1@Wmu+bmu
                lv  = np.clip(h1@Wlv+blv, -4, 4)
                std = np.exp(0.5*lv)
                eps = np.random.randn(B,k)
                z   = mu + std*eps
                # decode
                h2  = relu(z@Wd1+bd1)
                xr  = h2@Wo+bo
                # loss
                recon = np.mean((xr-Xb)**2)
                kl    = -0.5*np.mean(1+lv-mu**2-np.exp(lv))
                loss  = recon + beta*kl
                ep_l+=loss; ep_r+=recon; ep_k+=kl
                # backprop decoder
                dxr = 2*(xr-Xb)/(B*n)
                dWo=h2.T@dxr;     dbo=dxr.sum(0,keepdims=True)
                dh2=dxr@Wo.T*relu_d(z@Wd1+bd1)
                dWd1=z.T@dh2;     dbd1=dh2.sum(0,keepdims=True)
                dz=dh2@Wd1.T
                # backprop reparameterisation
                dmu = dz + beta*(mu)/(B*k)
                dlv = np.clip(dz*eps*0.5*std + beta*0.5*(np.exp(lv)-1)/(B*k), -1, 1)
                # backprop encoder
                dh1 = (dmu@Wmu.T + dlv@Wlv.T)*relu_d(Xb@We1+be1)
                for W,b,dW,db in [(Wmu,bmu,np.clip(h1.T@dmu,-1,1),np.clip(dmu.sum(0,keepdims=True),-1,1)),
                                   (Wlv,blv,np.clip(h1.T@dlv,-1,1),np.clip(dlv.sum(0,keepdims=True),-1,1)),
                                   (We1,be1,np.clip(Xb.T@dh1,-1,1),np.clip(dh1.sum(0,keepdims=True),-1,1))]:
                    W-=lr*dW; b-=lr*db
                Wd1-=lr*dWd1; bd1-=lr*dbd1; Wo-=lr*dWo; bo-=lr*dbo
            losses.append(ep_l); r_losses.append(ep_r); kl_losses.append(ep_k)

        return (We1,be1,Wmu,bmu,Wlv,blv,Wd1,bd1,Wo,bo), losses, r_losses, kl_losses

    def encode(X, w):
        We1,be1,Wmu,bmu,Wlv,blv,Wd1,bd1,Wo,bo = w
        h1=relu(X@We1+be1); mu=h1@Wmu+bmu; lv=np.clip(h1@Wlv+blv,-4,4)
        return mu, lv

    def decode(z, w):
        We1,be1,Wmu,bmu,Wlv,blv,Wd1,bd1,Wo,bo = w
        return relu(z@Wd1+bd1)@Wo+bo

    def recon_score(X, w):
        mu, lv = encode(X, w)
        xr     = decode(mu, w)
        mse    = np.mean((xr-X)**2, axis=1)
        kl     = -0.5*np.mean(1+lv-mu**2-np.exp(lv), axis=1)
        return mse, kl, xr, mu

    # ═══════════════════════════════════════════════════════════
    # TRAIN — full dataset for all tabs except anomaly
    # ═══════════════════════════════════════════════════════════
    with st.spinner("Training VAE on full dataset..."):
        w_full, losses_t, losses_r, losses_kl = train_vae(
            X_vae, vae_latent, vae_hidden, vae_beta, vae_epochs, vae_lr)

    mu_all, lv_all = encode(X_vae, w_full)
    xr_all         = decode(mu_all, w_full)
    mse_all,_,_,_  = recon_score(X_vae, w_full)
    final_recon    = float(np.mean(mse_all))
    final_kl       = float(-0.5*np.mean(1+lv_all-mu_all**2-np.exp(lv_all)))

    # ═══════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📖 VAE vs AE", "📉 Training", "🎲 Generate", "🗺️ Latent Space", "🚨 Anomaly Detection"
    ])

    # ────────────────────────────────────────────────────────────
    # TAB 1 — Concept
    # ────────────────────────────────────────────────────────────
    with tab1:
        c1,c2 = st.columns([1,1])
        with c1:
            st.markdown("""
<div class="card"><h3>🤔 What's Wrong with a Regular AutoEncoder?</h3>
<p>A regular AE compresses data to a point. Those points can end up scattered randomly — some regions of latent space are empty "holes". If you sample a random point from a hole and decode it, you get garbage.</p>
<div class="highlight">💡 AE: great at compression. Terrible at generation.</div></div>

<div class="card"><h3>🎲 VAE: Encode to a Distribution, not a Point</h3>
<p>Instead of "this input maps to point z", a VAE says "this input maps to a <b>bell curve</b> centered at μ with spread σ". Then we sample from that bell curve.</p>
<p>This forces the latent space to be smooth and continuous — no holes. Every point in the space decodes to something meaningful.</p>
<div class="highlight">💡 VAE = AE + probability distribution over the latent space</div></div>

<div class="card"><h3>🔀 The Reparameterization Trick</h3>
<p>Sampling is random — you can't backpropagate through randomness. Solution: write <b>z = μ + σ × ε</b> where ε is fixed random noise. Now gradients flow through μ and σ normally.</p>
<div class="highlight">💡 This one trick is what makes the whole VAE trainable</div></div>

<div class="card"><h3>⚖️ The VAE Loss: Two Terms</h3>
<p><b>Loss = Reconstruction + β × KL Divergence</b></p>
<ul>
<li><b>Reconstruction</b>: "output should look like input" — same as AE</li>
<li><b>KL Divergence</b>: "keep distributions close to a standard Gaussian" — fills in the holes</li>
<li><b>β</b>: how hard you push the KL term. β=1 is standard. Higher → smoother but blurrier.</li>
</ul>
<div class="highlight">💡 KL is what separates VAE from AE — it's the regularizer that creates the smooth latent space</div></div>

<div class="card"><h3>🆚 Quick Comparison</h3>
<table style="width:100%;font-size:0.85rem;color:#a8b2d8;border-collapse:collapse;">
<tr><th style="color:#4fc3f7;text-align:left;padding:4px 8px;">Feature</th>
    <th style="color:#4fc3f7;text-align:left;padding:4px 8px;">AutoEncoder</th>
    <th style="color:#4fc3f7;text-align:left;padding:4px 8px;">VAE</th></tr>
<tr style="border-top:1px solid #1e3a5f;"><td style="padding:4px 8px;">Latent</td><td style="padding:4px 8px;">Single point</td><td style="padding:4px 8px;">Distribution (μ, σ)</td></tr>
<tr style="border-top:1px solid #1e3a5f;"><td style="padding:4px 8px;">Can Generate?</td><td style="padding:4px 8px;">❌ Gaps = noise</td><td style="padding:4px 8px;">✅ Smooth space</td></tr>
<tr style="border-top:1px solid #1e3a5f;"><td style="padding:4px 8px;">Loss</td><td style="padding:4px 8px;">Reconstruction only</td><td style="padding:4px 8px;">Recon + KL</td></tr>
<tr style="border-top:1px solid #1e3a5f;"><td style="padding:4px 8px;">Best for</td><td style="padding:4px 8px;">Compression, denoising</td><td style="padding:4px 8px;">Generation, anomaly detection, interpolation</td></tr>
</table></div>
""", unsafe_allow_html=True)

        with c2:
            # Architecture diagram
            fig,ax=plt.subplots(figsize=(8,5.5))
            fig.patch.set_facecolor('#0a0f1e'); ax.set_facecolor('#0a0f1e'); ax.axis('off')
            blocks=[
                (0.02,0.22,'#0d2b45','#4fc3f7',f'Encoder\nInput→{vae_hidden}'),
                (0.27,0.46,'#1a0d30','#ab47bc',f'μ  and  log σ²\ndim = {vae_latent}'),
                (0.47,0.53,'#0a0f1e','#00e5ff','ε~N(0,1)\nSample z'),
                (0.54,0.73,'#0d2b45','#4fc3f7',f'Decoder\n{vae_latent}→{vae_hidden}'),
                (0.78,0.98,'#0d2b45','#4fc3f7',f'Output\n(reconstructed)'),
            ]
            for x0,x1,fc,ec,lbl in blocks:
                ax.add_patch(plt.Rectangle((x0,0.28),x1-x0,0.44,fc=fc,ec=ec,lw=2,zorder=3,alpha=0.9))
                ax.text((x0+x1)/2,0.5,lbl,ha='center',va='center',color='white',
                        fontsize=8,fontfamily='monospace',fontweight='bold',zorder=5)
            arrows=[(0.22,0.27,'#4fc3f7'),(0.46,0.47,'#ab47bc'),(0.53,0.54,'#00e5ff'),(0.73,0.78,'#4fc3f7')]
            for xs,xe,col in arrows:
                ax.annotate('',xy=(xe,0.5),xytext=(xs,0.5),
                            arrowprops=dict(arrowstyle='->',color=col,lw=2.5),zorder=6)
            ax.text(0.365,0.82,'Bottleneck\nDistribution',ha='center',color='#ab47bc',
                    fontsize=8.5,fontfamily='monospace',fontweight='bold')
            ax.text(0.5,0.14,'z = μ + σ · ε',ha='center',color='#00e5ff',
                    fontsize=9,fontfamily='monospace',fontweight='bold')
            ax.text(0.02,0.84,'x (input)',ha='left',color='#8892b0',fontsize=8,fontfamily='monospace')
            ax.text(0.80,0.84,'x̂ ≈ x',ha='left',color='#8892b0',fontsize=8,fontfamily='monospace')
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_title('VAE Architecture',color='#4fc3f7',fontsize=13,fontweight='bold')
            st.pyplot(fig); plt.close()

            st.markdown(f"""<div class="card">
<h3>📊 Current Model Stats</h3>
<table style="width:100%;font-size:0.88rem;color:#a8b2d8;">
<tr><td>Reconstruction Loss</td><td style="color:#4fc3f7;font-family:monospace;text-align:right;">{final_recon:.5f}</td></tr>
<tr><td>KL Divergence</td><td style="color:#ab47bc;font-family:monospace;text-align:right;">{final_kl:.5f}</td></tr>
<tr><td>Total Loss (β={vae_beta})</td><td style="color:#00e5ff;font-family:monospace;text-align:right;"><b>{final_recon+vae_beta*final_kl:.5f}</b></td></tr>
<tr><td>Input Dim → Latent Dim</td><td style="color:#66bb6a;font-family:monospace;text-align:right;">{X_vae.shape[1]} → {vae_latent}</td></tr>
<tr><td>Compression Ratio</td><td style="color:#ffa726;font-family:monospace;text-align:right;">{X_vae.shape[1]/vae_latent:.1f}×</td></tr>
</table></div>""", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────
    # TAB 2 — Training curves
    # ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📉 Training Loss Curves")
        st.markdown("""<div class="card">
<h3>How to Read These Charts</h3>
<ul>
<li><b>Total Loss</b> should steadily drop and then flatten — if still falling at the end, increase Epochs</li>
<li><b>Reconstruction Loss</b> going down = the decoder is getting better at rebuilding inputs</li>
<li><b>KL Divergence</b> rising then stabilizing is <em>normal</em> — it means the latent space is being regularized.
    If it stays at 0 the whole time ("KL collapse"), try increasing β or the hidden size.</li>
</ul>
</div>""", unsafe_allow_html=True)

        fig,axes=plt.subplots(1,3,figsize=(13,4)); fig.patch.set_facecolor('#0a0f1e')
        for ax in axes: style_ax(ax)
        for ax,data,col,ttl in zip(axes,
            [losses_t, losses_r, losses_kl],
            ['#00e5ff','#4fc3f7','#ab47bc'],
            ['Total Loss','Reconstruction Loss','KL Divergence']):
            ax.plot(data,color=col,lw=2)
            ax.set_title(ttl,color='#4fc3f7',fontsize=11,fontweight='bold')
            ax.set_xlabel("Epoch",color='#8892b0')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ────────────────────────────────────────────────────────────
    # TAB 3 — Generate
    # ────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🎲 Generate New Samples")
        st.markdown("""<div class="card">
<h3>Why Can VAEs Generate Data?</h3>
<p>The KL term forces the latent space to follow N(0,1). So we can simply <b>sample random points from N(0,1)</b> and decode them — every point in the space is "filled" with meaning.</p>
<p>A regular AE can't do this: random points in its latent space hit empty gaps → garbled output.</p>
<div class="highlight">💡 Try Temperature: low=conservative outputs, high=more varied but less accurate</div>
</div>""", unsafe_allow_html=True)

        gc1,gc2=st.columns(2)
        with gc1:
            n_gen = st.slider("Samples to generate", 4, 32, 16, 4, key="vae_ngen")
        with gc2:
            temp  = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1, key="vae_temp")
            st.markdown('<div style="color:#8892b0;font-size:0.75rem;">Temperature scales N(0,1) — higher = more spread = more diverse but less accurate</div>',unsafe_allow_html=True)

        if st.button("🎲 New random samples", key="vae_resample"):
            st.session_state["vae_seed"] = int(np.random.randint(0,9999))
        seed_g = st.session_state.get("vae_seed", 0)
        np.random.seed(seed_g)
        z_rand  = np.random.randn(n_gen, vae_latent)*temp
        gen_out = decode(z_rand, w_full)

        if is_digits:
            cols_r = 8
            rows_r = (n_gen+cols_r-1)//cols_r
            fig,axes=plt.subplots(rows_r,cols_r,figsize=(cols_r*1.6,rows_r*1.9))
            fig.patch.set_facecolor('#0a0f1e')
            axes=np.array(axes).reshape(rows_r,cols_r)
            for i in range(rows_r*cols_r):
                ax=axes[i//cols_r,i%cols_r]; ax.axis('off'); ax.set_facecolor('#0a0f1e')
                if i<n_gen:
                    ax.imshow(gen_out[i].reshape(8,8),cmap='Blues',vmin=-2,vmax=2)
            fig.suptitle(f'VAE Generated Digits — Temp={temp:.1f}',color='#4fc3f7',fontsize=11,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            fig,ax=plt.subplots(figsize=(8,4)); style_ax(ax,fig)
            ax.scatter(X_vae[:,0],X_vae[:,1],c='#4fc3f7',s=18,alpha=0.35,label='Real data')
            ax.scatter(gen_out[:,0],gen_out[:,1],c='#ab47bc',s=80,alpha=0.9,
                       edgecolors='white',lw=0.6,marker='*',zorder=5,label='Generated')
            ax.set_title(f'Generated vs Real | Temp={temp:.1f}',color='#4fc3f7',fontsize=11,fontweight='bold')
            ax.legend(facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ────────────────────────────────────────────────────────────
    # TAB 4 — Latent Space
    # ────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🗺️ Latent Space Visualization")
        st.markdown("""<div class="card">
<p>Even though the VAE trained with <b>no class labels</b>, similar inputs naturally cluster together in the latent space.
The map below shows where each training sample ended up after compression.
In a well-trained VAE, you should see smooth transitions between clusters — not sharp isolated islands.</p>
<div class="highlight">💡 If clusters are completely separated with empty space between them, try lowering β or increasing latent dim</div>
</div>""", unsafe_allow_html=True)

        if vae_latent >= 2:
            fig,axes=plt.subplots(1,2,figsize=(13,5)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)

            for i in range(n_cls_vae):
                m=y_vae==i
                axes[0].scatter(mu_all[m,0],mu_all[m,1],c=COLORS[i%len(COLORS)],
                                s=22,alpha=0.6,edgecolors='none',label=f'Class {i}')
            axes[0].set_title("Latent Space — colored by class",color='#4fc3f7',fontsize=11,fontweight='bold')
            axes[0].set_xlabel("z₁ (μ)",color='#8892b0'); axes[0].set_ylabel("z₂ (μ)",color='#8892b0')
            axes[0].legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')

            uncertainty=np.exp(0.5*lv_all).mean(axis=1)
            sc=axes[1].scatter(mu_all[:,0],mu_all[:,1],c=uncertainty,cmap='plasma',s=22,alpha=0.8)
            plt.colorbar(sc,ax=axes[1],label='Mean σ')
            axes[1].set_title("Latent Space — colored by uncertainty σ",color='#4fc3f7',fontsize=11,fontweight='bold')
            axes[1].set_xlabel("z₁ (μ)",color='#8892b0'); axes[1].set_ylabel("z₂ (μ)",color='#8892b0')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.info("💡 Left: clusters emerge without labels. Right: bright = uncertain encoding (near class boundaries)")

            # Interpolation
            st.markdown("### 🔀 Interpolation Between Two Classes")
            st.markdown("""<div class="card">
<p>One of the coolest VAE features: walk a straight line between two class centers in latent space.
Each decoded point along the path is a smooth blend between the two classes.
This only works because the latent space is continuous — no gaps.</p>
</div>""", unsafe_allow_html=True)

            ic1,ic2=st.columns(2)
            with ic1: ca=st.selectbox("From class",list(range(n_cls_vae)),index=0,key="vae_ca")
            with ic2: cb=st.selectbox("To class",  list(range(n_cls_vae)),index=min(1,n_cls_vae-1),key="vae_cb")

            n_steps=10
            mu_a=mu_all[y_vae==ca].mean(0); mu_b=mu_all[y_vae==cb].mean(0)
            z_path=np.array([(1-a)*mu_a+a*mu_b for a in np.linspace(0,1,n_steps)])
            x_path=decode(z_path, w_full)

            if is_digits:
                fig,axes2=plt.subplots(1,n_steps,figsize=(14,2.2)); fig.patch.set_facecolor('#0a0f1e')
                for i,(ax,img) in enumerate(zip(axes2,x_path)):
                    ax.imshow(img.reshape(8,8),cmap='Blues',vmin=-2,vmax=2); ax.axis('off')
                    ax.set_title(f'{i/(n_steps-1):.1f}',color='#8892b0',fontsize=7)
                fig.suptitle(f'Interpolation: Class {ca} → Class {cb}',color='#4fc3f7',fontsize=10,fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                fig,ax=plt.subplots(figsize=(8,3.5)); style_ax(ax,fig)
                for i in range(n_cls_vae):
                    m=y_vae==i
                    ax.scatter(mu_all[m,0],mu_all[m,1],c=COLORS[i%len(COLORS)],s=18,alpha=0.3)
                ax.plot(z_path[:,0],z_path[:,1],'o-',color='#00e5ff',lw=2.5,markersize=9,zorder=5,
                        label=f'Class {ca} → {cb}')
                ax.legend(facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
                ax.set_title('Interpolation path in latent space',color='#4fc3f7',fontsize=11,fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Set Latent Dimension to 2+ to see the 2D latent space map.")

    # ────────────────────────────────────────────────────────────
    # TAB 5 — Anomaly Detection  (the correct way)
    # ────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("## 🚨 Example: VAE for Anomaly Detection")

        st.markdown("""<div class="card">
<h3>🎓 The Story — Why This Works</h3>
<p>Imagine you work at a factory making digit "1". Every day you see thousands of 1s.
You become an expert at recognizing and reconstructing 1s perfectly.</p>
<p>One day someone hands you a "7". You've never learned to reconstruct 7s —
your reconstruction will be distorted and blurry. The <b>error is high</b>.</p>
<p>That's exactly how VAE anomaly detection works:</p>
<ol>
<li><b>Train the VAE on ONLY normal data</b> (e.g., only digit "1")</li>
<li><b>Run all data through the VAE</b> and measure reconstruction error</li>
<li><b>High error = anomaly</b> — the VAE couldn't reconstruct it because it never learned it</li>
<li><b>Set a threshold</b> — anything above = flagged as anomaly</li>
</ol>
<div class="highlight">💡 Key rule: the VAE must be trained on normal data ONLY. If you train on everything, it learns anomalies too and can't detect them.</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Setup: Choose Normal vs Anomaly")

        anom_c1,anom_c2,anom_c3 = st.columns(3)
        with anom_c1:
            st.markdown("""<div style="color:#8892b0;font-size:0.78rem;margin-bottom:4px;">
            🎯 <b>Normal Class</b><br>
            The VAE will be trained <b>only</b> on this class.
            Think of it as "what normal looks like".</div>""", unsafe_allow_html=True)
            normal_cls = st.selectbox("Normal Class (train on this)", list(range(n_cls_vae)),
                                       index=0, key="vae_norm_cls")

        with anom_c2:
            remaining = [c for c in range(n_cls_vae) if c != normal_cls]
            st.markdown("""<div style="color:#8892b0;font-size:0.78rem;margin-bottom:4px;">
            🔴 <b>Anomaly Class</b><br>
            The class we want to <b>detect</b>.
            The VAE has never seen this during training.</div>""", unsafe_allow_html=True)
            anomaly_cls = st.selectbox("Anomaly Class (detect this)", remaining if remaining else [0],
                                        index=0, key="vae_anom_cls2")

        with anom_c3:
            st.markdown("""<div style="color:#8892b0;font-size:0.78rem;margin-bottom:4px;">
            📏 <b>Threshold Percentile</b><br>
            Set to 95 = "flag the top 5% most unusual reconstruction errors as anomalies".<br>
            Lower = more sensitive (more alerts). Higher = stricter (fewer alerts).</div>""", unsafe_allow_html=True)
            thresh_pct = st.slider("Threshold Percentile", 50, 99, 95, 1, key="vae_thr2")

        # ── Train a SEPARATE VAE on normal-only data ──────────
        X_norm_only = X_vae[y_vae == normal_cls]
        X_anom_test = X_vae[y_vae == anomaly_cls]
        # also grab some normal test samples (not used in training)
        np.random.seed(99)
        perm        = np.random.permutation(len(X_norm_only))
        split       = max(1, int(len(X_norm_only)*0.8))
        X_norm_train= X_norm_only[perm[:split]]
        X_norm_test = X_norm_only[perm[split:]]

        if len(X_norm_train) < 4:
            st.warning("Not enough normal samples. Pick a different normal class.")
            st.stop()

        with st.spinner(f"Training VAE on normal class {normal_cls} only ({len(X_norm_train)} samples)..."):
            w_anom, anom_losses, anom_r, anom_kl = train_vae(
                X_norm_train, vae_latent, vae_hidden, vae_beta,
                min(vae_epochs, 200), vae_lr, seed=77)

        # ── Compute scores ────────────────────────────────────
        mse_norm, _, xr_norm, mu_norm = recon_score(X_norm_test,  w_anom)
        mse_anom, _, xr_anom, mu_anom= recon_score(X_anom_test,   w_anom)

        threshold = np.percentile(mse_norm, thresh_pct)

        tp = int(np.sum(mse_anom  > threshold))
        fp = int(np.sum(mse_norm  > threshold))
        tn = int(np.sum(mse_norm  <= threshold))
        fn = int(np.sum(mse_anom  <= threshold))
        precision = tp/(tp+fp+1e-9)
        recall    = tp/(tp+fn+1e-9)
        f1        = 2*precision*recall/(precision+recall+1e-9)

        st.markdown("---")
        st.markdown(f"### 📊 Results: Trained on class **{normal_cls}**, detecting class **{anomaly_cls}**")

        # Metrics
        mc1,mc2,mc3,mc4,mc5 = st.columns(5)
        for col,lbl,val,clr in zip([mc1,mc2,mc3,mc4,mc5],
            ["✅ Detected\n(True Pos)","❌ False Alarms\n(False Pos)","🎯 Precision","📡 Recall","⚖️ F1 Score"],
            [f"{tp}/{tp+fn}",f"{fp}/{fp+tn}",f"{precision:.1%}",f"{recall:.1%}",f"{f1:.3f}"],
            ['#00e5ff','#ef5350','#66bb6a','#ffa726','#ab47bc']):
            with col:
                st.markdown(f'<div class="metric-box"><div class="value" style="color:{clr};">{val}</div><div class="label" style="white-space:pre-line;">{lbl}</div></div>',unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Score distribution histogram ──────────────────────
        st.markdown("#### 📊 Step 1 — Look at the score distributions")
        st.markdown("""<div class="card">
<p>This histogram shows the reconstruction error for normal samples (blue) and anomaly samples (red).
A good detector has <b>little overlap</b> between the two distributions.
The vertical line is our threshold — everything to its right gets flagged as anomaly.</p>
</div>""", unsafe_allow_html=True)

        fig,axes=plt.subplots(1,2,figsize=(13,4.5)); fig.patch.set_facecolor('#0a0f1e')
        for ax in axes: style_ax(ax)

        all_scores = np.concatenate([mse_norm, mse_anom])
        bins = np.linspace(all_scores.min(), np.percentile(all_scores,99), 45)

        axes[0].hist(mse_norm, bins=bins, color='#4fc3f7', alpha=0.75,
                     label=f'Normal (class {normal_cls})', density=True)
        axes[0].hist(mse_anom, bins=bins, color='#ef5350', alpha=0.75,
                     label=f'Anomaly (class {anomaly_cls})', density=True)
        axes[0].axvline(threshold, color='#00e5ff', lw=2.5, ls='--',
                        label=f'Threshold (p={thresh_pct})')
        axes[0].fill_betweenx([0,axes[0].get_ylim()[1]+0.5], threshold, bins[-1],
                               alpha=0.08, color='#ef5350')
        axes[0].set_xlabel("Reconstruction Error (MSE)", color='#8892b0')
        axes[0].set_ylabel("Density", color='#8892b0')
        axes[0].set_title("Score Distribution — Normal vs Anomaly",
                          color='#4fc3f7', fontsize=11, fontweight='bold')
        axes[0].legend(fontsize=9, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')

        # Per-sample bar chart
        ns = min(50, len(mse_norm)); na = min(25, len(mse_anom))
        xs_n=np.arange(ns); xs_a=np.arange(ns+2, ns+2+na)
        bar_n=['#4fc3f7' if s<=threshold else '#ffa726' for s in mse_norm[:ns]]
        bar_a=['#ef5350' if s>threshold  else '#66bb6a'  for s in mse_anom[:na]]
        axes[1].bar(xs_n, mse_norm[:ns], color=bar_n, width=0.9, alpha=0.85)
        axes[1].bar(xs_a, mse_anom[:na], color=bar_a, width=0.9, alpha=0.85)
        axes[1].axhline(threshold, color='#00e5ff', lw=2, ls='--', label=f'Threshold = {threshold:.4f}')
        axes[1].axvline(ns+0.5, color='#ffffff', lw=1, ls=':', alpha=0.3)
        ymax = max(mse_norm[:ns].max(), mse_anom[:na].max())*1.1
        axes[1].text(ns/2, ymax*0.92, f'← Normal (cls {normal_cls})',
                     ha='center', color='#4fc3f7', fontsize=8, fontfamily='monospace')
        axes[1].text(ns+2+na/2, ymax*0.92, f'Anomaly (cls {anomaly_cls}) →',
                     ha='center', color='#ef5350', fontsize=8, fontfamily='monospace')
        axes[1].set_xlabel("Sample index", color='#8892b0')
        axes[1].set_ylabel("Reconstruction Error", color='#8892b0')
        axes[1].set_title("Per-sample Errors  |  🟠=normal flagged  🟢=anomaly missed",
                          color='#4fc3f7', fontsize=10, fontweight='bold')
        axes[1].legend(fontsize=9, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Reconstruction images for digits ──────────────────
        if is_digits:
            st.markdown("#### 🖼️ Step 2 — See Why the Error is High for Anomalies")
            st.markdown(f"""<div class="card">
<p>The VAE was trained only on digit <b>{normal_cls}</b>. Look at what happens when it tries to reconstruct digit <b>{anomaly_cls}</b>:
it forces everything through what it knows — producing something that looks more like a {normal_cls} than a {anomaly_cls}.</p>
<p>The <b>visual distortion</b> is exactly what the high reconstruction error measures numerically.</p>
</div>""", unsafe_allow_html=True)

            n_demo = min(7, len(X_norm_test), len(X_anom_test))
            fig,axes2=plt.subplots(4,n_demo,figsize=(n_demo*2.0,8.5))
            fig.patch.set_facecolor('#0a0f1e')
            rows_data=[
                (X_norm_test[:n_demo], xr_norm[:n_demo], f'Normal Input (digit {normal_cls})',   f'Normal Recon  ✅ err≈{mse_norm[:n_demo].mean():.3f}', '#4fc3f7','#00e5ff'),
                (X_anom_test[:n_demo], xr_anom[:n_demo], f'Anomaly Input (digit {anomaly_cls})', f'Anomaly Recon ❌ err≈{mse_anom[:n_demo].mean():.3f}', '#ef5350','#ffa726'),
            ]
            r=0
            for (Xi,Xr,lbl_in,lbl_rec,cin,crec) in rows_data:
                for c in range(n_demo):
                    ax=axes2[r,c]; ax.imshow(Xi[c].reshape(8,8),cmap='Blues',vmin=-2,vmax=2); ax.axis('off')
                    if c==0: ax.set_ylabel(lbl_in, color=cin, fontsize=8, fontfamily='monospace')
                r+=1
                for c in range(n_demo):
                    ax=axes2[r,c]; ax.imshow(Xr[c].reshape(8,8),cmap='Reds',vmin=-2,vmax=2); ax.axis('off')
                    if c==0: ax.set_ylabel(lbl_rec, color=crec, fontsize=8, fontfamily='monospace')
                r+=1
            fig.suptitle(f'VAE trained on {normal_cls} — reconstructing {normal_cls} (top) vs {anomaly_cls} (bottom)',
                         color='#4fc3f7',fontsize=11,fontweight='bold',y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Latent space with anomalies ───────────────────────
        if vae_latent >= 2:
            st.markdown("#### 🗺️ Step 3 — Where do Anomalies Land in Latent Space?")
            st.markdown(f"""<div class="card">
<p>The VAE was trained on class {normal_cls}, so its latent space is shaped around {normal_cls}s.
When anomaly class {anomaly_cls} is pushed through the encoder, it lands in unusual regions — 
far from the normal cluster, or scattered randomly.</p>
<p><b>Red X</b> = detected anomaly (high error). <b>Orange X</b> = missed anomaly (slipped through).</p>
</div>""", unsafe_allow_html=True)

            fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
            ax.scatter(mu_norm[:,0],mu_norm[:,1], c='#4fc3f7',s=30,alpha=0.5,
                       edgecolors='none', label=f'Normal test (cls {normal_cls})')
            det   = mse_anom > threshold
            undet = ~det
            if det.any():
                ax.scatter(mu_anom[det,0],mu_anom[det,1],c='#ef5350',s=130,
                           marker='X',zorder=6,edgecolors='white',lw=0.8,
                           label=f'Detected anomaly ({det.sum()})')
            if undet.any():
                ax.scatter(mu_anom[undet,0],mu_anom[undet,1],c='#ffa726',s=90,
                           marker='X',zorder=5,edgecolors='white',lw=0.5,
                           label=f'Missed anomaly ({undet.sum()})')
            ax.set_title(f'Latent Space — Normal cls {normal_cls} vs Anomaly cls {anomaly_cls}',
                         color='#4fc3f7',fontsize=11,fontweight='bold')
            ax.set_xlabel("z₁",color='#8892b0'); ax.set_ylabel("z₂",color='#8892b0')
            ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── ROC + threshold sweep ──────────────────────────────
        st.markdown("#### 📈 Step 4 — Find the Best Threshold")
        st.markdown("""<div class="card">
<h3>The Threshold Trade-off</h3>
<p>There's no perfect threshold — you're always trading off between two errors:</p>
<ul>
<li><b>Too low threshold</b> → catch every anomaly (high recall) but also flag many normal samples (high false alarm rate). Good for: medical diagnosis, fraud, security — missing an anomaly is dangerous.</li>
<li><b>Too high threshold</b> → very few false alarms but many anomalies slip through. Good for: situations where false alarms are costly (factory shutdowns, customer complaints).</li>
</ul>
<p>The <b>ROC Curve</b> shows this trade-off at every possible threshold. <b>AUC</b> (area under curve) summarizes overall separability — closer to 1.0 is better, 0.5 = random guessing.</p>
</div>""", unsafe_allow_html=True)

        pcts=list(range(50,100,2))
        f1s,recs,precs,fprs_=[],[],[],[]
        for p in pcts:
            thr=np.percentile(mse_norm,p)
            tp_=np.sum(mse_anom>thr); fp_=np.sum(mse_norm>thr)
            tn_=np.sum(mse_norm<=thr); fn_=np.sum(mse_anom<=thr)
            pr=tp_/(tp_+fp_+1e-9); re=tp_/(tp_+fn_+1e-9)
            f1s.append(2*pr*re/(pr+re+1e-9))
            recs.append(re); precs.append(pr)
            fprs_.append(fp_/(fp_+tn_+1e-9))

        fig,ax2=plt.subplots(1,2,figsize=(13,4.5)); fig.patch.set_facecolor('#0a0f1e')
        for ax in ax2: style_ax(ax)

        ax2[0].plot(pcts,f1s,   color='#ab47bc',lw=2,label='F1')
        ax2[0].plot(pcts,recs,  color='#4fc3f7',lw=2,label='Recall')
        ax2[0].plot(pcts,precs, color='#66bb6a',lw=2,label='Precision')
        ax2[0].axvline(thresh_pct,color='#00e5ff',lw=2,ls='--',label=f'Current={thresh_pct}th')
        ax2[0].set_xlabel("Threshold Percentile",color='#8892b0')
        ax2[0].set_ylabel("Score",color='#8892b0')
        ax2[0].set_title("Metrics vs Threshold",color='#4fc3f7',fontsize=11,fontweight='bold')
        ax2[0].legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        ax2[0].set_ylim(0,1.05)

        sorted_pairs=sorted(zip(fprs_,recs))
        sfprs=[p[0] for p in sorted_pairs]; srecs=[p[1] for p in sorted_pairs]
        try:
            from sklearn.metrics import auc as sk_auc
            roc_auc=sk_auc(sfprs,srecs)
        except:
            roc_auc=0.0
        ax2[1].plot(sfprs,srecs,color='#4fc3f7',lw=2.5,label=f'ROC (AUC={roc_auc:.2f})')
        ax2[1].plot([0,1],[0,1],color='#1e3a5f',lw=1.5,ls='--',label='Random (AUC=0.5)')
        cur_fpr=fp/(fp+tn+1e-9); cur_rec=recall
        ax2[1].scatter([cur_fpr],[cur_rec],c='#00e5ff',s=140,zorder=6,
                       edgecolors='white',lw=1.2,label='Current threshold')
        ax2[1].set_xlabel("False Positive Rate",color='#8892b0')
        ax2[1].set_ylabel("True Positive Rate (Recall)",color='#8892b0')
        ax2[1].set_title(f"ROC Curve | AUC = {roc_auc:.3f}",color='#4fc3f7',fontsize=11,fontweight='bold')
        ax2[1].legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        ax2[1].set_xlim(0,1); ax2[1].set_ylim(0,1.05)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# ██  CONTRASTIVE LEARNING
# ══════════════════════════════════════════════════════════════
elif algo == "🔗 Contrastive Learning":
    st.markdown("""
<div class="main-header">
<h1>🔗 Contrastive Learning</h1>
<p>Learn powerful representations without labels — by teaching the model what's similar and what's different</p>
</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin-bottom:3px;">
        🗜️ <b>Projection Dimension</b><br>
        Size of the embedding space where contrastive loss is applied.
        Too small = not enough room to separate. Too large = overfits noise.
        <b>32–128 is typical.</b></div>""", unsafe_allow_html=True)
        proj_dim = st.slider("Projection Dimension", 2, 128, 32, 2, key="cl_proj")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin:8px 0 3px;">
        🌡️ <b>Temperature τ (tau)</b><br>
        Controls how "sharp" the similarity distribution is.<br>
        • <b>Low τ (0.05–0.1)</b>: very sharp — model focuses on the hardest negatives. Learns faster but can be unstable.<br>
        • <b>High τ (0.5+)</b>: soft — treats all negatives similarly. More stable but slower.<br>
        <b>0.1 is the SimCLR default.</b></div>""", unsafe_allow_html=True)
        temperature = st.slider("Temperature τ", 0.01, 1.0, 0.1, 0.01, key="cl_temp")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin:8px 0 3px;">
        🔄 <b>Epochs</b><br>
        Contrastive learning usually needs more epochs than supervised learning
        because the signal per sample is weaker (no labels).<br>
        100–300 is fine for these small datasets.</div>""", unsafe_allow_html=True)
        cl_epochs = st.slider("Epochs", 50, 400, 150, 50, key="cl_ep")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin:8px 0 3px;">
        ⚡ <b>Learning Rate</b><br>
        Contrastive learning is sensitive to LR.<br>
        0.001–0.01 works well. Too high → embeddings collapse to a single point.</div>""", unsafe_allow_html=True)
        cl_lr = st.select_slider("Learning Rate", [0.0001,0.001,0.005,0.01,0.05], value=0.005, key="cl_lr")

        st.markdown("""<div style="color:#8892b0;font-size:0.75rem;margin:8px 0 3px;">
        🎭 <b>Augmentation Noise</b><br>
        How much random noise to add when creating two "views" of the same sample.
        This is the core of data augmentation in contrastive learning.<br>
        • Too low: views are identical → trivial task, no learning<br>
        • Too high: views are too different → impossible task</div>""", unsafe_allow_html=True)
        aug_noise = st.slider("Augmentation Noise σ", 0.01, 0.8, 0.15, 0.01, key="cl_aug")

        st.markdown("---")
        st.markdown("### 📊 Data")
        cl_ds = st.selectbox("Dataset", ["Digits (MNIST-like)", "Iris", "Blobs", "Moons"], key="cl_ds")

    # ═══════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════
    tab1,tab2,tab3,tab4 = st.tabs([
        "📖 Core Concepts",
        "🎭 Augmentation",
        "📉 Training",
        "🗺️ Embedding Space"
    ])

    # ═══════════════════════════════════════════════════════════
    # DATA
    # ═══════════════════════════════════════════════════════════
    @st.cache_data
    def get_cl_data(name):
        if name == "Digits (MNIST-like)":
            d = datasets.load_digits()
            X, y = d.data.astype(np.float32), d.target
        elif name == "Iris":
            d = datasets.load_iris()
            X, y = d.data.astype(np.float32), d.target
        elif name == "Moons":
            X, y = datasets.make_moons(300, noise=0.15, random_state=42)
            X = X.astype(np.float32)
        else:
            X, y = datasets.make_blobs(300, centers=4, n_features=4, random_state=42)
            X = X.astype(np.float32)
        sc = StandardScaler()
        X = sc.fit_transform(X).astype(np.float32)
        return X, y

    X_cl, y_cl = get_cl_data(cl_ds)
    n_in_cl    = X_cl.shape[1]
    n_cls_cl   = len(np.unique(y_cl))

    # ═══════════════════════════════════════════════════════════
    # NUMPY SimCLR-style encoder
    # ═══════════════════════════════════════════════════════════
    def relu(x):    return np.maximum(0, x)
    def relu_d(x):  return (x > 0).astype(float)

    def cl_forward(X, W1, b1, W2, b2, W3, b3):
        """Encoder (2 layers) + projection head (1 layer)"""
        h1  = relu(X  @ W1 + b1)   # (N, hidden)
        h2  = relu(h1 @ W2 + b2)   # (N, hidden)
        z   = h2 @ W3 + b3          # (N, proj_dim) — projection head output
        # L2-normalise → embeddings live on unit hypersphere
        norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        return h2, z / norm

    def nt_xent_loss(z_i, z_j, tau):
        """
        Normalised Temperature-scaled Cross Entropy Loss (NT-Xent)
        Used in SimCLR.  z_i, z_j are already L2-normalised.
        For batch size N we have 2N samples total.
        The positive pair for sample k is sample k+N (its augmented twin).
        All other 2N-2 samples are negatives.
        """
        N   = len(z_i)
        Z   = np.concatenate([z_i, z_j], axis=0)          # (2N, d)
        sim = Z @ Z.T / tau                                 # (2N, 2N) cosine similarity / tau
        # mask out self-similarity on diagonal
        np.fill_diagonal(sim, -1e9)
        # positives: (i, i+N) and (i+N, i)
        pos_mask = np.zeros((2*N, 2*N), dtype=bool)
        for k in range(N):
            pos_mask[k,   k+N] = True
            pos_mask[k+N, k  ] = True
        # log-softmax trick for numerical stability
        sim_max  = sim.max(axis=1, keepdims=True)
        log_sum  = np.log(np.exp(sim - sim_max).sum(axis=1, keepdims=True)) + sim_max
        log_prob = sim - log_sum
        loss     = -log_prob[pos_mask].mean()
        return loss

    def augment(X, sigma):
        """Simple Gaussian noise augmentation — creates a 'different view' of the same sample"""
        return X + np.random.randn(*X.shape).astype(np.float32) * sigma

    @st.cache_data
    def train_cl(X, proj_dim, tau, epochs, lr, aug_sigma, seed=42):
        np.random.seed(seed)
        n   = X.shape[1]
        hid = 64
        # Xavier init
        W1 = np.random.randn(n,   hid)      * np.sqrt(2/n)
        b1 = np.zeros((1, hid))
        W2 = np.random.randn(hid, hid)      * np.sqrt(2/hid)
        b2 = np.zeros((1, hid))
        W3 = np.random.randn(hid, proj_dim) * np.sqrt(2/hid)
        b3 = np.zeros((1, proj_dim))

        losses = []
        bs = min(128, len(X))

        for ep in range(epochs):
            idx      = np.random.permutation(len(X))
            ep_loss  = 0.0
            n_batches = 0

            for i in range(0, len(X) - bs + 1, bs):
                Xb = X[idx[i:i+bs]]          # (B, n)
                B  = len(Xb)

                # ── Two augmented views ───────────────────────
                Xi = augment(Xb, aug_sigma)   # (B, n)
                Xj = augment(Xb, aug_sigma)   # (B, n)

                # ── Forward pass (manual, keep all intermediates) ──
                # View i
                a1i = relu(Xi @ W1 + b1)      # (B, hid)
                a2i = relu(a1i @ W2 + b2)     # (B, hid)
                p_i = a2i @ W3 + b3           # (B, proj_dim)  — pre-norm
                n_i = np.linalg.norm(p_i, axis=1, keepdims=True) + 1e-8
                zi  = p_i / n_i               # (B, proj_dim)  — L2-normalised

                # View j
                a1j = relu(Xj @ W1 + b1)
                a2j = relu(a1j @ W2 + b2)
                p_j = a2j @ W3 + b3
                n_j = np.linalg.norm(p_j, axis=1, keepdims=True) + 1e-8
                zj  = p_j / n_j

                loss = nt_xent_loss(zi, zj, tau)
                ep_loss += loss; n_batches += 1

                # ── Gradient of NT-Xent w.r.t. zi and zj ─────
                Z   = np.concatenate([zi, zj], axis=0)   # (2B, proj_dim)
                sim = Z @ Z.T / tau                       # (2B, 2B)
                np.fill_diagonal(sim, -1e9)
                sim = sim - sim.max(axis=1, keepdims=True)
                exp_sim  = np.exp(sim)
                softmax  = exp_sim / exp_sim.sum(axis=1, keepdims=True)  # (2B, 2B)

                dL_dZ = softmax.copy()                    # (2B, proj_dim) — wait, this is wrong shape
                # dL_dZ is (2B, 2B); we need to convert to grad w.r.t. Z rows (2B, proj_dim)
                # dL/dZ_k = (1/tau) * sum_l [ (softmax[k,l] - indicator_positive[k,l]) * Z[l] ]
                pos_mask = np.zeros((2*B, 2*B))
                for k in range(B):
                    pos_mask[k,   k+B] = 1.0
                    pos_mask[k+B, k  ] = 1.0
                grad_sim = (softmax - pos_mask) / (2 * B * tau)   # (2B, 2B)
                dL_dZ    = grad_sim @ Z + grad_sim.T @ Z           # (2B, proj_dim)

                dL_dzi = dL_dZ[:B]   # (B, proj_dim)
                dL_dzj = dL_dZ[B:]   # (B, proj_dim)

                # ── Backprop through L2-norm ───────────────────
                # d/dp (p/||p||) = (I - z z^T) / ||p||
                def backprop_norm(dL_dz, p, n_norm):
                    # dL_dz: (B, proj_dim), p: (B, proj_dim), n_norm: (B, 1)
                    z = p / n_norm
                    dot = (dL_dz * z).sum(axis=1, keepdims=True)  # (B,1)
                    return (dL_dz - z * dot) / n_norm              # (B, proj_dim)

                dL_dpi = backprop_norm(dL_dzi, p_i, n_i)  # (B, proj_dim)
                dL_dpj = backprop_norm(dL_dzj, p_j, n_j)  # (B, proj_dim)

                # ── Backprop through projection head W3 ───────
                # p = a2 @ W3 + b3
                dW3 = np.clip(a2i.T @ dL_dpi + a2j.T @ dL_dpj, -1, 1)  # (hid, proj_dim)
                db3 = np.clip((dL_dpi + dL_dpj).sum(0, keepdims=True),  -1, 1)

                da2i = dL_dpi @ W3.T   # (B, hid)
                da2j = dL_dpj @ W3.T

                # ── Backprop through encoder layer 2 ──────────
                # a2 = relu(a1 @ W2 + b2)
                dz2i = da2i * relu_d(a1i @ W2 + b2)  # (B, hid)
                dz2j = da2j * relu_d(a1j @ W2 + b2)

                dW2 = np.clip(a1i.T @ dz2i + a1j.T @ dz2j, -2, 2)  # (hid, hid)
                db2 = np.clip((dz2i + dz2j).sum(0, keepdims=True),  -2, 2)

                da1i = dz2i @ W2.T   # (B, hid)
                da1j = dz2j @ W2.T

                # ── Backprop through encoder layer 1 ──────────
                # a1 = relu(X @ W1 + b1)
                dz1i = da1i * relu_d(Xi @ W1 + b1)   # (B, hid)
                dz1j = da1j * relu_d(Xj @ W1 + b1)

                dW1 = np.clip(Xi.T @ dz1i + Xj.T @ dz1j, -2, 2)  # (n, hid)
                db1 = np.clip((dz1i + dz1j).sum(0, keepdims=True), -2, 2)

                # ── Update ────────────────────────────────────
                W1 -= lr * dW1;  b1 -= lr * db1
                W2 -= lr * dW2;  b2 -= lr * db2
                W3 -= lr * dW3;  b3 -= lr * db3

            losses.append(ep_loss / max(n_batches, 1))

        return (W1, b1, W2, b2, W3, b3), losses

    # ── Get embeddings before and after training ──────────────
    @st.cache_data
    def get_random_weights(n_in, proj_dim, seed=42):
        np.random.seed(seed)
        hid=64
        W1=np.random.randn(n_in,hid)*np.sqrt(2/n_in); b1=np.zeros((1,hid))
        W2=np.random.randn(hid,hid)*np.sqrt(2/hid);   b2=np.zeros((1,hid))
        W3=np.random.randn(hid,proj_dim)*np.sqrt(2/hid);b3=np.zeros((1,proj_dim))
        return (W1,b1,W2,b2,W3,b3)

    w_rand = get_random_weights(n_in_cl, proj_dim)
    _, emb_before = cl_forward(X_cl, *w_rand)

    with st.spinner("Training contrastive model (no labels used!)..."):
        w_cl, cl_losses = train_cl(X_cl, proj_dim, temperature, cl_epochs, cl_lr, aug_noise)
    _, emb_after = cl_forward(X_cl, *w_cl)

    # ── PCA to 2D for visualisation ───────────────────────────
    def pca2d(Z):
        Z = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        return Z @ Vt[:2].T

    eb2 = pca2d(emb_before)
    ea2 = pca2d(emb_after)

    # ─────────────────────────────────────────────────────────
    # TAB 1 — Core Concepts
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("## What is Contrastive Learning?")

        c1,c2 = st.columns([1,1])
        with c1:
            st.markdown("""
<div class="card">
<h3>🤔 The Big Problem: Learning Without Labels</h3>
<p>Labelling data is expensive and slow. A doctor labels 1000 X-rays.
A factory needs to annotate 50,000 defect images.
Supervised learning is hungry for labels.</p>
<p>Contrastive learning says: <b>"What if we could learn useful representations
just by comparing data points to each other — no labels at all?"</b></p>
<div class="highlight">💡 The core insight: you don't need labels to know that two photos of the same dog should be similar,
and a photo of a dog and a cat should be different.</div>
</div>

<div class="card">
<h3>🎓 The Fundamental Idea — Pull & Push</h3>
<p>Contrastive learning works with two forces acting on the embedding space:</p>
<ul>
<li><b>Pull</b>: similar samples → embeddings should be <b>close together</b></li>
<li><b>Push</b>: different samples → embeddings should be <b>far apart</b></li>
</ul>
<p>After training, the embedding space is organized so that semantically similar
things cluster together — even though we never told the model what the classes were.</p>
<div class="highlight">💡 Think of it as learning a "similarity map" of your data from scratch</div>
</div>

<div class="card">
<h3>🎭 What are Positive and Negative Pairs?</h3>
<ul>
<li><b>Positive pair</b>: two versions of the <em>same</em> sample (e.g., the same image with different crops, rotations, or noise). The model should embed these close together.</li>
<li><b>Negative pair</b>: two <em>different</em> samples. The model should push these apart.</li>
</ul>
<p>In SimCLR (the most famous method), every sample in a batch is augmented twice.
Each sample's two augmented versions are a positive pair.
All other samples in the batch are negatives.</p>
<div class="highlight">💡 With batch size 256: each sample has 1 positive and 510 negatives to push away from!</div>
</div>

<div class="card">
<h3>🌡️ What is NT-Xent Loss (the Contrastive Loss)?</h3>
<p>NT-Xent = <b>Normalised Temperature-scaled Cross Entropy</b>. For each sample i:</p>
<ol>
<li>Compute cosine similarity between its embedding and all other 2N-1 embeddings</li>
<li>Divide similarities by temperature τ (sharpening)</li>
<li>Apply softmax → treat as a classification problem: "which one is my positive?"</li>
<li>Maximise the probability assigned to the true positive</li>
</ol>
<p>Minimising NT-Xent pushes positive pairs together and negative pairs apart simultaneously.</p>
<div class="highlight">💡 It's like asking the model: "out of all these samples, which one is your twin?" — and penalising wrong answers</div>
</div>

<div class="card">
<h3>⚠️ The Collapse Problem</h3>
<p>A trivial solution exists: embed everything to the same point.
Then all positives are perfectly close — but so are all negatives. Loss = 0 but the model learned nothing.</p>
<p>This is called <b>representation collapse</b>. Solutions:</p>
<ul>
<li><b>NT-Xent</b>: the negative pairs in the denominator prevent collapse — if everything collapses, negatives get high similarity too, which increases the loss</li>
<li><b>Large batches</b>: more negatives = harder to collapse</li>
<li><b>L2 normalisation</b>: embeddings on a unit sphere — collapse means all points pile up in one spot, which NT-Xent penalises</li>
</ul>
<div class="highlight">💡 The negatives are what prevent the model from cheating — they're the hard constraints</div>
</div>

<div class="card">
<h3>🔗 The Encoder + Projection Head Architecture</h3>
<p>SimCLR uses a two-part network:</p>
<ul>
<li><b>Encoder f(·)</b>: the main network (e.g., ResNet, MLP). Extracts features. <em>This is what you keep after training.</em></li>
<li><b>Projection Head g(·)</b>: a small MLP on top. Contrastive loss is applied here. <em>This is thrown away after training!</em></li>
</ul>
<p>Why discard the projection head? Chen et al. (SimCLR) showed that the encoder representations are more general and transferable than the projection head's output. The projection head "overfits" to the augmentation strategy.</p>
<div class="highlight">💡 You train with the projection head, but you use the encoder for downstream tasks (classification, clustering, etc.)</div>
</div>
""", unsafe_allow_html=True)

        with c2:
            # ── Visual 1: Pull & Push diagram ────────────────
            fig,ax=plt.subplots(figsize=(7,4))
            fig.patch.set_facecolor('#0a0f1e'); ax.set_facecolor('#0d1b2a'); ax.axis('off')
            # Draw two clusters
            np.random.seed(7)
            pts_a = np.random.randn(6,2)*0.25 + [-1.5, 0.8]
            pts_b = np.random.randn(6,2)*0.25 + [ 1.5, 0.8]
            pts_a2= pts_a + np.random.randn(6,2)*0.5  # augmented versions scattered
            pts_b2= pts_b + np.random.randn(6,2)*0.5

            # Before: scattered
            ax.scatter(pts_a2[:,0]-0.3, pts_a2[:,1]-1.8, c='#4fc3f7', s=70, alpha=0.5, marker='o')
            ax.scatter(pts_b2[:,0]+0.3, pts_b2[:,1]-1.8, c='#ef5350', s=70, alpha=0.5, marker='o')
            ax.text(0, -1.0, 'Before training\n(random embeddings)', ha='center',
                    color='#8892b0', fontsize=8.5, fontfamily='monospace')

            # After: clustered
            ax.scatter(pts_a[:,0], pts_a[:,1], c='#4fc3f7', s=70, alpha=0.9, marker='o', zorder=5)
            ax.scatter(pts_b[:,0], pts_b[:,1], c='#ef5350', s=70, alpha=0.9, marker='o', zorder=5)
            # Pull arrows within cluster A
            ctr_a = pts_a.mean(0)
            for p in pts_a[:3]:
                ax.annotate('',xy=ctr_a,xytext=p,
                            arrowprops=dict(arrowstyle='->',color='#00e5ff',lw=1.2,alpha=0.7))
            # Push arrow between clusters
            ax.annotate('',xy=pts_b.mean(0),xytext=pts_a.mean(0),
                        arrowprops=dict(arrowstyle='<->',color='#ffa726',lw=2.5,
                                        connectionstyle='arc3,rad=0.3'))
            ax.text(0, 1.7, '← Push apart →', ha='center', color='#ffa726',
                    fontsize=8.5, fontfamily='monospace', fontweight='bold')
            ax.text(-1.5,0.1,'Pull\ntogether',ha='center',color='#00e5ff',
                    fontsize=8, fontfamily='monospace')
            ax.text(0, -0.2, 'After training\n(organised embeddings)', ha='center',
                    color='#8892b0', fontsize=8.5, fontfamily='monospace')
            ax.set_xlim(-3,3); ax.set_ylim(-2.5,2.5)
            ax.set_title('The Pull & Push Mechanism', color='#4fc3f7',
                         fontsize=11, fontweight='bold')
            st.pyplot(fig); plt.close()

            # ── Visual 2: Architecture diagram ───────────────
            fig,ax=plt.subplots(figsize=(7,4.5))
            fig.patch.set_facecolor('#0a0f1e'); ax.set_facecolor('#0a0f1e'); ax.axis('off')
            boxes = [
                (0.05,0.60,'#0d2b45','#4fc3f7','Sample x'),
                (0.05,0.25,'#0d2b45','#4fc3f7','Augment\nx_i  x_j'),
                (0.35,0.25,'#112240','#66bb6a','Encoder f(·)\n(keep this!)'),
                (0.65,0.25,'#1a0d30','#ab47bc','Proj Head g(·)\n(discard after)'),
            ]
            for x,y,fc,ec,lbl in boxes:
                ax.add_patch(plt.Rectangle((x,y),0.22,0.18,fc=fc,ec=ec,lw=2,zorder=3))
                ax.text(x+0.11,y+0.09,lbl,ha='center',va='center',color='white',
                        fontsize=8,fontfamily='monospace',fontweight='bold',zorder=5)
            # arrows
            arrows_diag = [
                (0.27,0.34,0.35,0.34,'#4fc3f7'),
                (0.57,0.34,0.65,0.34,'#66bb6a'),
                (0.76,0.34,0.82,0.34,'#ab47bc'),
                (0.16,0.60,0.16,0.43,'#4fc3f7'),
            ]
            for x1,y1,x2,y2,col in arrows_diag:
                ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                            arrowprops=dict(arrowstyle='->',color=col,lw=2))
            ax.text(0.87,0.34,'NT-Xent\nLoss',ha='left',va='center',color='#00e5ff',
                    fontsize=8.5,fontfamily='monospace',fontweight='bold')
            ax.text(0.46,0.10,'h (representation)',ha='center',color='#66bb6a',
                    fontsize=8,fontfamily='monospace')
            ax.text(0.75,0.10,'z (projection)',ha='center',color='#ab47bc',
                    fontsize=8,fontfamily='monospace')
            ax.text(0.16,0.82,'Input x',ha='center',color='#8892b0',
                    fontsize=8.5,fontfamily='monospace')
            ax.set_xlim(0,1.1); ax.set_ylim(0,1)
            ax.set_title('SimCLR Architecture', color='#4fc3f7',
                         fontsize=11, fontweight='bold')
            st.pyplot(fig); plt.close()

            # ── Visual 3: NT-Xent explained ──────────────────
            fig,ax=plt.subplots(figsize=(7,3))
            fig.patch.set_facecolor('#0a0f1e'); ax.set_facecolor('#0d1b2a')
            ax.tick_params(colors='#8892b0',labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')
            labels=['z_i (self)','z_j\n(positive)','neg 1','neg 2','neg 3','neg 4','neg 5']
            sims  = [-10, 0.85,  -0.3, 0.1, -0.5,  0.2, -0.1]
            colors= ['#555','#00e5ff','#ef5350','#ef5350','#ef5350','#ef5350','#ef5350']
            ax.bar(labels,sims,color=colors,alpha=0.85,edgecolor='#0a0f1e',linewidth=0.5)
            ax.axhline(0,color='#1e3a5f',lw=1)
            ax.set_ylabel('Cosine similarity / τ',color='#8892b0',fontsize=8)
            ax.set_title('NT-Xent: what the loss sees for one sample',
                         color='#4fc3f7',fontsize=10,fontweight='bold')
            ax.text(1,0.88,'← maximise this',ha='center',color='#00e5ff',
                    fontsize=7.5,fontfamily='monospace')
            st.pyplot(fig); plt.close()

    # ─────────────────────────────────────────────────────────
    # TAB 2 — Augmentation
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("## 🎭 The Role of Data Augmentation")

        st.markdown("""
<div class="card">
<h3>🎭 Augmentation is the Heart of Contrastive Learning</h3>
<p>The choice of augmentation defines what the model considers "the same".
If you augment with brightness changes, the model learns brightness-invariant features.
If you augment with rotations, it learns rotation-invariant features.</p>
<p>In tabular/numerical data (like we have here), augmentation = adding Gaussian noise.
In images, common augmentations are: random crop, colour jitter, gaussian blur, horizontal flip.</p>
<div class="highlight">💡 The augmentation is your domain knowledge encoded implicitly — you're telling the model "these transformations don't change the identity of a sample"</div>
</div>

<div class="card">
<h3>⚖️ The Noise Trade-off</h3>
<p>The augmentation noise σ (sigma) controls how different the two views are:</p>
<ul>
<li><b>σ too small</b>: views are nearly identical → trivial task → model learns nothing meaningful. Like asking "are these two photos of the same person?" when they're pixel-identical.</li>
<li><b>σ just right</b>: views are different enough to be challenging but similar enough to share structure → model must learn what's truly invariant.</li>
<li><b>σ too large</b>: views are so different they no longer share the same structure → impossible task → loss explodes, model fails.</li>
</ul>
<div class="highlight">💡 Current noise level: σ = {:.2f}. Try moving the slider and watching how embeddings change in the Embedding Space tab.</div>
</div>
""".format(aug_noise), unsafe_allow_html=True)

        # Show augmentation effect
        st.markdown("### 👀 Visualising Augmented Pairs")
        if cl_ds == "Digits (MNIST-like)":
            np.random.seed(5)
            n_show = 6
            indices = np.random.choice(len(X_cl), n_show, replace=False)
            fig,axes=plt.subplots(3,n_show,figsize=(n_show*1.8,5.5))
            fig.patch.set_facecolor('#0a0f1e')
            row_lbls=['Original','View 1 (aug)','View 2 (aug)']
            row_cols=['#8892b0','#4fc3f7','#ab47bc']
            for c,idx in enumerate(indices):
                v1 = augment(X_cl[[idx]], aug_noise)[0]
                v2 = augment(X_cl[[idx]], aug_noise)[0]
                for r,(data,cmap) in enumerate([(X_cl[idx],'Blues'),(v1,'Blues'),(v2,'Purples')]):
                    ax=axes[r,c]
                    ax.imshow(data.reshape(8,8),cmap=cmap,vmin=-2,vmax=2)
                    ax.axis('off')
                    if c==0:
                        ax.set_ylabel(row_lbls[r],color=row_cols[r],
                                      fontsize=8.5,fontfamily='monospace')
            fig.suptitle(f'Positive pairs: same digit, two noisy views (σ={aug_noise:.2f})',
                         color='#4fc3f7',fontsize=11,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            np.random.seed(5)
            n_show=8
            idx=np.random.choice(len(X_cl),n_show,replace=False)
            X_show=X_cl[idx]
            V1=augment(X_show,aug_noise); V2=augment(X_show,aug_noise)
            fig,axes=plt.subplots(1,3,figsize=(13,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            for i in range(n_cls_cl):
                m=y_cl[idx]==i
                if m.any():
                    axes[0].scatter(X_show[m,0],X_show[m,1],c=COLORS[i%len(COLORS)],
                                    s=80,alpha=0.9,edgecolors='white',lw=0.5,label=f'cls {i}')
                    axes[1].scatter(V1[m,0],V1[m,1],c=COLORS[i%len(COLORS)],
                                    s=80,alpha=0.9,edgecolors='white',lw=0.5)
                    axes[2].scatter(V2[m,0],V2[m,1],c=COLORS[i%len(COLORS)],
                                    s=80,alpha=0.9,edgecolors='white',lw=0.5)
            for ax,ttl in zip(axes,['Original','View 1 (augmented)','View 2 (augmented)']):
                ax.set_title(ttl,color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[0].legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            fig.suptitle(f'Positive pairs — same colour = same class (σ={aug_noise:.2f})',
                         color='#4fc3f7',fontsize=11,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("""
<div class="card">
<h3>🤔 Why Does This Work For Learning?</h3>
<p>The model sees View 1 and View 2 and must embed them close together.
To do this reliably (across the whole dataset), it can't just memorise pixel values —
the noise makes them different every time.</p>
<p>Instead, it must learn the <b>underlying structure</b> that both views share:
the shape of the digit, the cluster it belongs to, the statistical pattern.</p>
<p>This is exactly the representation we want for downstream tasks!</p>
<div class="highlight">💡 This is the key intuition: augmentation forces the model to be invariant to surface-level noise and focus on deep structure</div>
</div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # TAB 3 — Training
    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📉 Training Progress")

        st.markdown("""<div class="card">
<h3>Reading the Loss Curve</h3>
<ul>
<li>Loss starts high — embeddings are random, the positive pair is no more similar than the negatives</li>
<li>Loss decreases as the model learns to pull positives together and push negatives apart</li>
<li>If loss <b>doesn't decrease at all</b> → try increasing learning rate or reducing temperature τ</li>
<li>If loss <b>spikes or explodes</b> → learning rate too high or augmentation noise too large</li>
<li>A well-trained model typically reaches a loss of 2–4 for small datasets (log(batch_size) is the random baseline)</li>
</ul>
<div class="highlight">💡 Random baseline loss ≈ log(batch_size) = {:.2f}. If your final loss is near this, the model hasn't learned anything.</div>
</div>""".format(np.log(min(128,len(X_cl)))), unsafe_allow_html=True)

        fig,ax=plt.subplots(figsize=(10,4)); style_ax(ax,fig)
        ax.plot(cl_losses,color='#4fc3f7',lw=2.5,label='NT-Xent Loss')
        ax.axhline(np.log(min(128,len(X_cl))),color='#ef5350',lw=1.5,
                   ls='--',alpha=0.7,label=f'Random baseline ≈ {np.log(min(128,len(X_cl))):.2f}')
        ax.set_xlabel("Epoch",color='#8892b0')
        ax.set_ylabel("NT-Xent Loss",color='#8892b0')
        ax.set_title(f"Contrastive Training Loss (τ={temperature}, σ={aug_noise})",
                     color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=10,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
            ["📉 Final Loss","🎲 Random Baseline","📐 Proj Dim","🌡️ Temperature τ"],
            [f"{cl_losses[-1]:.3f}",f"{np.log(min(128,len(X_cl))):.2f}",
             str(proj_dim),str(temperature)]):
            with col: st.markdown(metric_box(val,lbl),unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # TAB 4 — Embedding Space
    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🗺️ Embedding Space — Before vs After Training")

        st.markdown("""<div class="card">
<h3>What to Look For</h3>
<p>The two plots below show a PCA projection of all sample embeddings to 2D.
<b>Remember: no class labels were used during training.</b></p>
<ul>
<li><b>Before</b>: random network → embeddings are scattered randomly, no structure</li>
<li><b>After</b>: contrastive training → similar samples cluster together, even without labels</li>
</ul>
<p>The quality of clustering is a proxy for representation quality.
Good clustering = the model has learned meaningful features.</p>
<div class="highlight">💡 If the before and after look similar, try training longer or reducing temperature τ</div>
</div>""", unsafe_allow_html=True)

        fig,axes=plt.subplots(1,2,figsize=(13,5)); fig.patch.set_facecolor('#0a0f1e')
        for ax in axes: style_ax(ax)
        for ax,emb,ttl in zip(axes,[eb2,ea2],['Before Training (random weights)','After Contrastive Training']):
            for i in range(n_cls_cl):
                m=y_cl==i
                ax.scatter(emb[m,0],emb[m,1],c=COLORS[i%len(COLORS)],
                           s=28,alpha=0.75,edgecolors='none',label=f'Class {i}')
            ax.set_title(ttl,color='#4fc3f7',fontsize=11,fontweight='bold')
            ax.set_xlabel("PC 1",color='#8892b0'); ax.set_ylabel("PC 2",color='#8892b0')
            ax.legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f',
                      loc='upper right')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Similarity heatmap ────────────────────────────────
        st.markdown("### 🔥 Similarity Matrix")
        st.markdown("""<div class="card">
<p>The heatmap below shows the cosine similarity between the embeddings of the first 30 samples.
In a well-trained model, you should see <b>bright blocks along the diagonal</b> — samples of the same class are similar to each other.</p>
<div class="highlight">💡 A block-diagonal pattern means the model has discovered the class structure without any labels</div>
</div>""", unsafe_allow_html=True)

        n_sim = min(30, len(X_cl))
        # Sort by class for cleaner visualisation
        sort_idx = np.argsort(y_cl[:n_sim])
        emb_sorted = emb_after[sort_idx]
        sim_mat = emb_sorted @ emb_sorted.T

        fig,axes=plt.subplots(1,2,figsize=(13,5)); fig.patch.set_facecolor('#0a0f1e')
        for ax,emb,ttl in zip(axes,
            [emb_before[sort_idx] @ emb_before[sort_idx].T, sim_mat],
            ['Before Training','After Training']):
            ax.set_facecolor('#0d1b2a')
            im=ax.imshow(emb,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
            plt.colorbar(im,ax=ax,label='Cosine Similarity')
            ax.set_title(ttl,color='#4fc3f7',fontsize=11,fontweight='bold')
            ax.set_xlabel("Sample (sorted by class)",color='#8892b0')
            ax.set_ylabel("Sample (sorted by class)",color='#8892b0')
            ax.tick_params(colors='#8892b0')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── KNN accuracy on embeddings ────────────────────────
        st.markdown("### 🎯 How Good are the Representations? (Linear Probe)")
        st.markdown("""<div class="card">
<h3>Linear Probe — The Standard Evaluation</h3>
<p>To measure representation quality, we train a simple linear classifier (KNN with K=1)
on top of the frozen embeddings. No fine-tuning, no extra learning — just "can a straight line separate the clusters?"</p>
<p>If the representations are good, even a simple classifier will perform well.
This is called a <b>linear probe</b> or <b>linear evaluation</b>.</p>
<div class="highlight">💡 Good rule of thumb: if linear probe accuracy >> random (1/n_classes), the model has learned something useful</div>
</div>""", unsafe_allow_html=True)

        from sklearn.neighbors import KNeighborsClassifier as KNN_
        from sklearn.model_selection import cross_val_score
        random_baseline = 1.0/n_cls_cl
        try:
            acc_before = cross_val_score(KNN_(n_neighbors=5),emb_before,y_cl,cv=3).mean()
            acc_after  = cross_val_score(KNN_(n_neighbors=5),emb_after, y_cl,cv=3).mean()
        except:
            acc_before = acc_after = 0.0

        pb1,pb2,pb3=st.columns(3)
        for col,lbl,val,clr in zip([pb1,pb2,pb3],
            ["🎲 Random Baseline","📊 Before Training (KNN-5)","✅ After Training (KNN-5)"],
            [f"{random_baseline:.1%}",f"{acc_before:.1%}",f"{acc_after:.1%}"],
            ['#8892b0','#ffa726','#00e5ff']):
            with col:
                st.markdown(f'<div class="metric-box"><div class="value" style="color:{clr};">{val}</div><div class="label">{lbl}</div></div>',unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#8892b0;font-family:'JetBrains Mono',monospace;font-size:0.75rem;padding:0.4rem 0;">
    ML Explorer &nbsp;·&nbsp; SVM · Regression · Decision Tree · K-Means · KNN · MLP · AutoEncoder · VAE · Contrastive &nbsp;·&nbsp; 🤖
</div>
""", unsafe_allow_html=True)
