import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Toleransi Pohon",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a1a 0%, #2d5a27 100%);
    }
    [data-testid="stSidebar"] * { color: #e8f5e3 !important; }
    [data-testid="stSidebar"] .stRadio label { 
        padding: 8px 12px; border-radius: 8px; cursor: pointer;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.1); }

    /* Main bg */
    .main { background: #f4f9f1; }
    .block-container { padding-top: 2rem; }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #2d5a27 0%, #4a7c3f 50%, #6aaa5e 100%);
        border-radius: 16px;
        padding: 28px 36px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 4px 20px rgba(45,90,39,0.3);
    }
    .header-banner h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: white; }
    .header-banner p  { margin: 6px 0 0; opacity: 0.85; font-size: 0.95rem; color: white; }

    /* Metric cards */
    .metric-row { display: flex; gap: 16px; margin-bottom: 24px; }
    .metric-card {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #4a7c3f;
    }
    .metric-card .label { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; color: #2d5a27; line-height: 1.1; }
    .metric-card .sub   { font-size: 0.8rem; color: #888; margin-top: 2px; }

    /* Section card */
    .section-card {
        background: white;
        border-radius: 14px;
        padding: 24px 28px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .section-card h3 { color: #2d5a27; margin-top: 0; border-bottom: 2px solid #e8f5e3; padding-bottom: 10px; margin-bottom: 10px;}

    /* Prediction result */
    .pred-result {
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 16px;
        text-align: center;
    }
    .pred-result.match    { background: #e8f5e3; border: 2px solid #4a7c3f; }
    .pred-result.mismatch { background: #fff3e0; border: 2px solid #e67e22; }
    .pred-label { font-size: 2.2rem; font-weight: 700; margin: 8px 0; }
    .pred-label.Toleran     { color: #2d5a27; }
    .pred-label.Moderat     { color: #4a7c3f; }
    .pred-label.Intermediet { color: #e67e22; }
    .pred-label.Sensitif    { color: #c0392b; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-Toleran     { background: #d4edda; color: #155724; }
    .badge-Moderat     { background: #d1ecf1; color: #0c5460; }
    .badge-Intermediet { background: #fff3cd; color: #856404; }
    .badge-Sensitif    { background: #f8d7da; color: #721c24; }

    /* Param table */
    .param-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 16px 0;
    }
    .param-item {
        background: #f4f9f1;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        border: 1px solid #d4e8cc;
    }
    .param-item .pname { font-size: 0.72rem; color: #666; text-transform: uppercase; }
    .param-item .pval  { font-size: 1.3rem; font-weight: 700; color: #2d5a27; }

    /* Divider */
    hr { border: none; border-top: 1px solid #e0e8dc; margin: 20px 0; }

    /* Streamlit widget overrides */
    .stSelectbox label, .stRadio label { font-weight: 500; color: #333; }
    div[data-testid="stSelectbox"] > div > div {
        border-color: #4a7c3f !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2d5a27, #4a7c3f) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: transform 0.15s !important;
    }
    .stButton > button:hover { transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('DATA AI - Cut.csv')
    df['Polusi_Angka'] = df['Kategori Paparan Polusi'].map({'Rendah': 0, 'Sedang': 1, 'Tinggi': 2})
    feature_names = ['Asam Askorbat (mg/g)', 'Total Klorofil', 'Kadar Ph', 'Kadar Air', 'Polusi_Angka']
    X = df[feature_names].values.astype(np.float64)
    Y = df['Label Toleransi'].astype(str).values
    plant_names  = df['Nama Latin'].astype(str).values
    plant_common = df['Nama Umum'].astype(str).values
    all_indices  = df.index.values.astype(np.int64)
    return df, X, Y, plant_names, plant_common, all_indices, feature_names

def split_data(_X, _Y, _plant_names, _plant_common, _all_indices):
    X  = np.array(_X,            dtype=np.float64)
    Y  = np.array(list(_Y),      dtype=str)
    pn = np.array(list(_plant_names),  dtype=str)
    pc = np.array(list(_plant_common), dtype=str)
    ai = np.array(list(_all_indices),  dtype=np.int64)
    return train_test_split(
        X, Y, pn, pc, ai,
        test_size=0.2, random_state=42, stratify=Y
    )

# ── ID3 functions ─────────────────────────────────────────────────────────────
def entropy(y):
    counts = Counter(y)
    total  = len(y)
    result = 0
    for count in counts.values():
        p = count / total
        result -= p * np.log2(p + 1e-9)
    return result

def information_gain(X_col, y, threshold):
    parent_entropy = entropy(y)
    left_mask  = X_col <= threshold
    right_mask = ~left_mask
    if left_mask.sum() == 0 or right_mask.sum() == 0: return 0
    n = len(y)
    return parent_entropy - (
        (left_mask.sum()  / n) * entropy(y[left_mask]) +
        (right_mask.sum() / n) * entropy(y[right_mask])
    )

def best_split_id3(X, y):
    best_ig = -1; best_feature = None; best_threshold = None
    for fi in range(X.shape[1]):
        thresholds = (np.unique(X[:, fi])[:-1] + np.unique(X[:, fi])[1:]) / 2
        for t in thresholds:
            ig = information_gain(X[:, fi], y, t)
            if ig > best_ig:
                best_ig = ig; best_feature = fi; best_threshold = t
    return best_feature, best_threshold, best_ig

def build_tree_id3(X, y, feature_names, depth=0, max_depth=5):
    if len(set(y)) == 1: return {'leaf': True, 'label': y[0]}
    if depth >= max_depth:
        return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}
    fi, t, ig = best_split_id3(X, y)
    if ig == 0: return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}
    lm = X[:, fi] <= t
    return {'leaf': False, 'feature': fi, 'feature_name': feature_names[fi],
            'threshold': t, 'ig': ig,
            'left':  build_tree_id3(X[lm],  y[lm],  feature_names, depth+1, max_depth),
            'right': build_tree_id3(X[~lm], y[~lm], feature_names, depth+1, max_depth)}

# ── CART functions ────────────────────────────────────────────────────────────
def gini(y):
    counts = Counter(y); total = len(y)
    return 1.0 - sum((c/total)**2 for c in counts.values())

def gini_gain(X_col, y, threshold):
    lm = X_col <= threshold; rm = ~lm
    if lm.sum() == 0 or rm.sum() == 0: return 0
    n = len(y)
    return gini(y) - (lm.sum()/n)*gini(y[lm]) - (rm.sum()/n)*gini(y[rm])

def best_split_cart(X, y):
    best_g = -1; best_feature = None; best_threshold = None
    for fi in range(X.shape[1]):
        thresholds = (np.unique(X[:, fi])[:-1] + np.unique(X[:, fi])[1:]) / 2
        for t in thresholds:
            g = gini_gain(X[:, fi], y, t)
            if g > best_g:
                best_g = g; best_feature = fi; best_threshold = t
    return best_feature, best_threshold, best_g

def build_tree_cart(X, y, feature_names, depth=0, max_depth=5):
    if len(set(y)) == 1: return {'leaf': True, 'label': y[0]}
    if depth >= max_depth:
        return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}
    fi, t, g = best_split_cart(X, y)
    if g == 0: return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}
    lm = X[:, fi] <= t
    return {'leaf': False, 'feature': fi, 'feature_name': feature_names[fi],
            'threshold': t, 'gain': g,
            'left':  build_tree_cart(X[lm],  y[lm],  feature_names, depth+1, max_depth),
            'right': build_tree_cart(X[~lm], y[~lm], feature_names, depth+1, max_depth)}

def predict_one(tree, x):
    if tree['leaf']: return tree['label']
    return predict_one(tree['left'], x) if x[tree['feature']] <= tree['threshold'] else predict_one(tree['right'], x)

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])

# ── Train models ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    df, X, Y, plant_names, plant_common, all_indices, feature_names = load_data()
    X_train, X_test, Y_train, Y_test, names_train, names_test, common_train, common_test, idx_train, idx_test = split_data(
        X, Y, plant_names, plant_common, all_indices
    )
    tree_id3  = build_tree_id3(X_train,  Y_train, feature_names)
    tree_cart = build_tree_cart(X_train, Y_train, feature_names)
    pred_id3  = predict(tree_id3,  X_test)
    pred_cart = predict(tree_cart, X_test)
    return (tree_id3, tree_cart, X_train, X_test, Y_train, Y_test,
            names_test, common_test, idx_test, pred_id3, pred_cart, feature_names, df)

# ── Label descriptions ────────────────────────────────────────────────────────
LABEL_DESC = {
    'Toleran':     '🟢 Spesies ini memiliki ketahanan tinggi terhadap polusi udara. Cocok ditanam di kawasan industri atau jalan raya dengan polusi tinggi.',
    'Moderat':     '🔵 Spesies ini memiliki ketahanan sedang. Dapat ditanam di kawasan perkotaan dengan tingkat polusi menengah.',
    'Intermediet': '🟡 Spesies ini berada di antara toleran dan sensitif. Perlu pertimbangan lebih lanjut sesuai kondisi lokasi.',
    'Sensitif':    '🔴 Spesies ini rentan terhadap polusi udara. Lebih cocok ditanam di kawasan dengan udara bersih.'
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌳 Klasifikasi Toleransi Pohon")
    st.markdown("**Kelompok 4** · Decision Tree")
    st.markdown("---")
    page = st.radio("Navigasi", [
        "🏠  Beranda",
        "🔍  Prediksi",
        "📊  Evaluasi Model",
        "🗂️  Dataset",
        "🌲  Visualisasi Tree"
    ])
    st.markdown("---")
    st.markdown("<small>ID3 (Entropy) vs CART (Gini Impurity)<br>266 sampel · 4 kelas · 5 fitur</small>", unsafe_allow_html=True)

# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner("Memuat dan melatih model..."):
    (tree_id3, tree_cart, X_train, X_test, Y_train, Y_test,
     names_test, common_test, idx_test, pred_id3, pred_cart,
     feature_names, df) = train_models()

acc_id3  = accuracy_score(Y_test, pred_id3)  * 100
acc_cart = accuracy_score(Y_test, pred_cart) * 100

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Beranda":
    st.markdown("""
    <div class="header-banner">
        <h1>🌳 Sistem Pendukung Keputusan Tata Ruang Hijau</h1>
        <p>Klasifikasi Toleransi Spesies Pohon Terhadap Polusi Udara · Algoritma Decision Tree ID3 & CART</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Total Dataset</div>
            <div class="value">266</div>
            <div class="sub">Spesies pohon</div>
        </div>
        <div class="metric-card">
            <div class="label">Akurasi ID3</div>
            <div class="value">{acc_id3:.1f}%</div>
            <div class="sub">Information Gain</div>
        </div>
        <div class="metric-card">
            <div class="label">Akurasi CART</div>
            <div class="value">{acc_cart:.1f}%</div>
            <div class="sub">Gini Impurity</div>
        </div>
        <div class="metric-card">
            <div class="label">Jumlah Kelas</div>
            <div class="value">4</div>
            <div class="sub">Toleran · Moderat · Intermediet · Sensitif</div>
        </div>
        <div class="metric-card">
            <div class="label">Fitur</div>
            <div class="value">5</div>
            <div class="sub">Parameter biokimia daun</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📋 Tentang Penelitian</h3>
            <p style="color: #555;">Penelitian ini membangun sistem klasifikasi otomatis tingkat toleransi spesies pohon terhadap polusi udara menggunakan pendekatan <em>machine learning</em>.</p>
            <p style="color: #555;">Dataset dikompilasi dari <strong>15 jurnal ilmiah</strong> mencakup <strong>20 lokasi</strong> di <strong>5 negara</strong> (Indonesia, India, Pakistan, Filipina, Nigeria).</p>
            <p style="color: #555;">Klasifikasi didasarkan pada empat parameter biokimia daun dari metode <strong>APTI (Air Pollution Tolerance Index)</strong> oleh Singh & Rao (1983), ditambah kategori paparan polusi lokasi.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🎯 Kategori Toleransi</h3>
            <p style="color:#333;"><span class="badge badge-Toleran">Toleran</span> &nbsp; Ketahanan tinggi terhadap polusi</p>
            <p style="color:#333;"><span class="badge badge-Moderat">Moderat</span> &nbsp; Ketahanan sedang</p>
            <p style="color:#333;"><span class="badge badge-Intermediet">Intermediet</span> &nbsp; Di antara moderat & sensitif</p>
            <p style="color:#333;"><span class="badge badge-Sensitif">Sensitif</span> &nbsp; Rentan terhadap polusi</p>
            <br>
            <p style="color:#333;"><small>Kategorisasi mengacu pada Liu et al. (1991)</small></p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Prediksi":
    total_train = len(Y_train)
    total_test = len(Y_test)

    st.markdown(f"""
    <div class="header-banner">
        <h1>🔍 Prediksi Toleransi Pohon</h1>
        <p>
            Dataset dibagi menjadi <strong>{total_train} data training</strong> dan <strong>{total_test} data testing</strong> (80:20).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Build test dataframe
    test_df = pd.DataFrame({
        'nama_latin' : names_test,
        'nama_umum'  : common_test,
        'real_label' : Y_test,
        'pred_id3'   : pred_id3,
        'pred_cart'  : pred_cart,
    }, index=idx_test)
    for i, fn in enumerate(feature_names[:-1]):  # exclude Polusi_Angka
        test_df[fn] = X_test[:, i]
    test_df['Kategori Paparan Polusi'] = df.loc[idx_test, 'Kategori Paparan Polusi'].values

    st.caption(
        f"Pilih salah satu spesies dari data testing di bawah ini untuk melihat hasil prediksi model terhadap label aktual."
    )

    col_sel, col_algo = st.columns([2, 1])
    with col_sel:
        # Dropdown 1: nama umum (sorted)
        nama_umum_list = sorted(test_df['nama_umum'].unique())
        selected_umum  = st.selectbox("🌿 Pilih Nama Umum Tanaman", nama_umum_list)

    # Filter by selected plant
    plant_rows = test_df[test_df['nama_umum'] == selected_umum]

    with col_sel:
        # Dropdown 2: linked polusi
        polusi_options = sorted(plant_rows['Kategori Paparan Polusi'].unique())
        if len(polusi_options) > 1:
            selected_polusi = st.selectbox("🏭 Pilih Kategori Paparan Polusi", polusi_options)
        else:
            selected_polusi = polusi_options[0]
            st.info(f"📍 Data tersedia untuk polusi: **{selected_polusi}**")

    with col_algo:
        algo = st.radio("⚙️ Algoritma", ["ID3", "CART", "Keduanya"], index=2)

    # Get the row
    row = plant_rows[plant_rows['Kategori Paparan Polusi'] == selected_polusi].iloc[0]

    st.markdown("<hr>", unsafe_allow_html=True)

    # Parameter display
    st.markdown(f"""
    <div class="section-card">
        <h3>📐 Parameter Biokimia — <em>{row['nama_latin']}</em></h3>
        <div class="param-grid">
            <div class="param-item">
                <div class="pname">Asam Askorbat</div>
                <div class="pval">{row['Asam Askorbat (mg/g)']:.3f}</div>
                <div class="pname">mg/g</div>
            </div>
            <div class="param-item">
                <div class="pname">Total Klorofil</div>
                <div class="pval">{row['Total Klorofil']:.3f}</div>
                <div class="pname">mg/g</div>
            </div>
            <div class="param-item">
                <div class="pname">Kadar pH</div>
                <div class="pval">{row['Kadar Ph']:.2f}</div>
                <div class="pname">pH</div>
            </div>
            <div class="param-item">
                <div class="pname">Kadar Air</div>
                <div class="pval">{row['Kadar Air']:.2f}</div>
                <div class="pname">%</div>
            </div>
            <div class="param-item">
                <div class="pname">Paparan Polusi</div>
                <div class="pval">{selected_polusi}</div>
                <div class="pname">kategori</div>
            </div>
        </div>
        <p style="color: #70b34e;"><small>📍 Nama Latin: <em>{row['nama_latin']}</em> &nbsp;·&nbsp; Label Aktual: 
        <span class="badge badge-{row['real_label']}">{row['real_label']}</span></small></p>
    </div>
    """, unsafe_allow_html=True)

    # Results
    def show_result(pred_label, real_label, algo_name):
        match    = pred_label == real_label
        css_cls  = "match" if match else "mismatch"
        icon     = "✅" if match else "⚠️"
        verdict  = "Prediksi Benar" if match else "Prediksi Tidak Tepat"
        st.markdown(f"""
        <div class="pred-result {css_cls}">
            <div style="font-size:0.85rem;font-weight:600;color:#555;">{algo_name}</div>
            <div class="pred-label {pred_label}">{pred_label}</div>
            <div style="font-size:0.85rem;color:#555;">{icon} {verdict} &nbsp;·&nbsp; Real: 
            <span class="badge badge-{real_label}">{real_label}</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<p style='margin-top:10px;font-size:0.88rem;color:#444;'>{LABEL_DESC[pred_label]}</p>", unsafe_allow_html=True)

    if algo == "Keduanya":
        c1, c2 = st.columns(2)
        with c1: show_result(row['pred_id3'],  row['real_label'], "🔵 ID3 — Information Gain")
        with c2: show_result(row['pred_cart'], row['real_label'], "🟢 CART — Gini Impurity")
    elif algo == "ID3":
        show_result(row['pred_id3'],  row['real_label'], "🔵 ID3 — Information Gain")
    else:
        show_result(row['pred_cart'], row['real_label'], "🟢 CART — Gini Impurity")

    #
    st.subheader("Data Testing beserta Hasil Prediksi")
    st.dataframe(test_df, use_container_width=True)
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUASI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Evaluasi Model":
    st.markdown("""
    <div class="header-banner">
        <h1>📊 Evaluasi Model</h1>
        <p>Perbandingan performa algoritma ID3 dan CART</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Akurasi ID3</div>
            <div class="value">{acc_id3:.2f}%</div>
            <div class="sub">Test set (54 sampel)</div>
        </div>
        <div class="metric-card">
            <div class="label">Akurasi CART</div>
            <div class="value">{acc_cart:.2f}%</div>
            <div class="sub">Test set (54 sampel)</div>
        </div>
        <div class="metric-card">
            <div class="label">K-Fold ID3</div>
            <div class="value">62.03%</div>
            <div class="sub">k=5, rata-rata</div>
        </div>
        <div class="metric-card">
            <div class="label">K-Fold CART</div>
            <div class="value">69.56%</div>
            <div class="sub">k=5, rata-rata</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Classification Report", "Confusion Matrix", "Perbandingan"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔵 ID3")
            report_id3 = classification_report(Y_test, pred_id3, output_dict=True)

            df_r = pd.DataFrame(report_id3).T
            df_r.loc["accuracy", "precision"] = np.nan
            df_r.loc["accuracy", "recall"] = np.nan
            df_r.loc["accuracy", "support"] = len(Y_test)

            st.dataframe(df_r.round(2))
        with c2:
            st.markdown("#### 🟢 CART")
            report_cart = classification_report(Y_test, pred_cart, output_dict=True)
            
            df_r2 = pd.DataFrame(report_cart).T
            df_r2.loc["accuracy", "precision"] = np.nan
            df_r2.loc["accuracy", "recall"] = np.nan
            df_r2.loc["accuracy", "support"] = len(Y_test)

            st.dataframe(df_r2.round(2))

    with tab2:
        c1, c2 = st.columns(2)
        labels = ['Intermediet', 'Moderat', 'Sensitif', 'Toleran']
        with c1:
            st.markdown("#### 🔵 ID3")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(Y_test, pred_id3, labels=labels)
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title("Confusion Matrix — ID3", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with c2:
            st.markdown("#### 🟢 CART")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(Y_test, pred_cart, labels=labels)
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap='Greens', colorbar=False)
            ax.set_title("Confusion Matrix — CART", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab3:
        compare_data = {
            'Metrik'         : ['Akurasi Test Set', 'K-Fold Average', 'F1 Intermediet', 'F1 Moderat', 'F1 Sensitif', 'F1 Toleran'],
            'ID3'            : [f'{acc_id3:.2f}%', '62.03%', '0.40', '0.62', '0.70', '0.67'],
            'CART'           : [f'{acc_cart:.2f}%', '69.56%', '0.43', '0.64', '0.75', '0.62'],
            'Unggul'         : ['CART', 'CART', 'CART', 'CART', 'CART', 'ID3'],
        }
        st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)
        st.info("💡 CART unggul secara keseluruhan. ID3 sedikit lebih baik pada kelas Toleran. Kelas Intermediet menjadi kelas terlemah di kedua model karena jumlah sampel paling sedikit (42 dari 266).")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️  Dataset":
    st.markdown("""
    <div class="header-banner">
        <h1>🗂️ Dataset</h1>
        <p>266 spesies pohon · 15 jurnal · 20 lokasi · 5 negara</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""<div class="section-card"><h3>📈 Distribusi Kelas</h3>""", unsafe_allow_html=True)
        dist = pd.Series(df['Label Toleransi'].values).value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ['#2d5a27', '#4a7c3f', '#e67e22', '#c0392b']
        bars = ax.bar(dist.index, dist.values, color=colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, dist.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Jumlah Sampel')
        ax.set_facecolor('#f4f9f1')
        fig.patch.set_facecolor('#f4f9f1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="section-card"><h3>🔍 Jelajahi Data</h3>""", unsafe_allow_html=True)
        filter_label  = st.multiselect("Filter Label", options=df['Label Toleransi'].unique(), default=list(df['Label Toleransi'].unique()))
        filter_polusi = st.multiselect("Filter Polusi", options=df['Kategori Paparan Polusi'].unique(), default=list(df['Kategori Paparan Polusi'].unique()))
        display_cols = ['Nama Umum', 'Nama Latin', 'Asam Askorbat (mg/g)', 'Total Klorofil', 'Kadar Ph', 'Kadar Air', 'Kategori Paparan Polusi', 'Label Toleransi']
        filtered = df[
            df['Label Toleransi'].isin(filter_label) &
            df['Kategori Paparan Polusi'].isin(filter_polusi)
        ][display_cols].reset_index(drop=True)
        filtered.insert(0, "No.", range(1, len(filtered) + 1))
        st.dataframe(filtered, use_container_width=True, height=600, hide_index=True)
        st.caption(f"Menampilkan {len(filtered)} dari 266 data")
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALISASI Tree
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌲  Visualisasi Tree":
    st.markdown("""
    <div class="header-banner">
        <h1>🌲 Visualisasi Decision Tree</h1>
        <p>Struktur tree ID3 dan CART dengan max_depth = 5</p>
    </div>
    """, unsafe_allow_html=True)

    algo_vis = st.radio("Pilih Algoritma", ["ID3", "CART"], horizontal=True)
    criterion = 'entropy' if algo_vis == 'ID3' else 'gini'

    sk_tree = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=42)
    sk_tree.fit(X_train, Y_train)

    depth_show = st.slider("Kedalaman yang ditampilkan", 1, 5, 3)

    fig, ax = plt.subplots(figsize=(20, 8), dpi=150)
    plot_tree(
        sk_tree,
        feature_names=feature_names,
        class_names=sk_tree.classes_,
        filled=True, rounded=True, fontsize=9,
        max_depth=depth_show, ax=ax,
        impurity=True
    )
    ax.set_title(f"Decision Tree {algo_vis} — {'Information Gain (Entropy)' if algo_vis == 'ID3' else 'Gini Impurity'}", fontsize=13)
    fig.patch.set_facecolor('#f4f9f1')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    sk_acc = accuracy_score(Y_test, sk_tree.predict(X_test)) * 100
    st.info(f"ℹ️ Tree ini dirender menggunakan sklearn untuk keperluan visualisasi. Model prediksi menggunakan implementasi manual. Akurasi sklearn {algo_vis}: **{sk_acc:.2f}%**")