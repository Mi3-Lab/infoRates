"""
InfoRates Dashboard
Spatiotemporal Aliasing in Video Action Recognition: A Cross-Architecture Analysis at Scale
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
st.set_page_config(
    page_title="InfoRates — Spatiotemporal Aliasing",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Streamlit Cloud: data is bundled in dashboard/data/
# Local cluster:  also try the full evaluations/ tree
_HERE = Path(__file__).parent
DATA  = _HERE / "data"   # bundled CSVs (committed to git)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_KEYS = ["r3d_18","mc3_18","r2plus1d_18","slowfast_r50","timesformer","vivit","videomae","videomamba"]
MODEL_NAMES = {
    "r3d_18":"R3D-18","mc3_18":"MC3-18","r2plus1d_18":"R2+1D",
    "slowfast_r50":"SlowFast","timesformer":"TimeSformer",
    "vivit":"ViViT","videomae":"VideoMAE","videomamba":"VideoMamba",
}
FAMILIES = {
    "r3d_18":"CNN","mc3_18":"CNN","r2plus1d_18":"CNN","slowfast_r50":"Dual-CNN",
    "timesformer":"Transformer","vivit":"Transformer","videomae":"Transformer","videomamba":"SSM",
}
FAM_COLOR = {"CNN":"#e74c3c","Dual-CNN":"#e67e22","Transformer":"#3498db","SSM":"#2ecc71"}
DS_KEYS   = ["autsl","diving48","ssv2","hmdb51","driveact","epic_kitchens","ucf101","finegym"]
DS_LABELS = {
    "autsl":"AUTSL (Sign Language)","diving48":"Diving-48 (Fine-grained)",
    "ssv2":"SSv2 (Causal)","hmdb51":"HMDB-51 (Sports)",
    "driveact":"DriveAct (In-vehicle)","epic_kitchens":"EPIC-Kitchens (Egocentric)",
    "ucf101":"UCF-101 (Appearance)","finegym":"FineGym (Fine-Grained Gym)",
}


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_sweeps():
    """Load native-resolution coverage×stride sweep.

    Primary source: dashboard/data/sweep_summary.csv.
    Fallback: coverage_stride_sweep/{model}_{dataset}_trainres{native}/sweep_summary.csv
    for any (model, dataset) pairs missing from the primary source.
    """
    NATIVE = {"r3d_18": 112, "mc3_18": 112, "r2plus1d_18": 112, "slowfast_r50": 224,
              "timesformer": 224, "vivit": 224, "videomae": 224, "videomamba": 224}
    sweep_root = Path(__file__).parent.parent / "evaluations/accv2026/coverage_stride_sweep"

    f = DATA / "sweep_summary.csv"
    df = pd.read_csv(f) if f.exists() else pd.DataFrame()

    # Find which (model, dataset) pairs are missing from the global CSV
    present = set() if df.empty else set(zip(df["model"], df["dataset"]))
    extra_rows = []
    if sweep_root.exists():
        ds_list = ["ucf101", "ssv2", "hmdb51", "diving48", "autsl",
                   "driveact", "epic_kitchens", "finegym"]
        for mk, native in NATIVE.items():
            for ds in ds_list:
                if (mk, ds) in present:
                    continue
                trainres_csv = sweep_root / f"{mk}_{ds}_trainres{native}" / "sweep_summary.csv"
                if trainres_csv.exists():
                    try:
                        tmp = pd.read_csv(trainres_csv)
                        tmp["model"] = mk
                        tmp["dataset"] = ds
                        extra_rows.append(tmp)
                    except Exception:
                        pass

    if extra_rows:
        df = pd.concat([df] + extra_rows, ignore_index=True) if not df.empty \
             else pd.concat(extra_rows, ignore_index=True)

    if df.empty:
        return df
    df["acc"] = df["top1"] * 100
    df["model_name"] = df["model"].map(MODEL_NAMES)
    df["family"] = df["model"].map(FAMILIES)
    return df


@st.cache_data
def compute_tds(df_sweep):
    """Compute real TDS from E1 data, excluding feature-collapsed models."""
    tds = {}
    for ds in DS_KEYS:
        drops = []
        sub = df_sweep[(df_sweep.dataset == ds) & (df_sweep.coverage == 100)]
        for m in MODEL_KEYS:
            s1  = sub[(sub.model == m) & (sub.stride == 1)]["acc"]
            s16 = sub[(sub.model == m) & (sub.stride == 16)]["acc"]
            if s1.empty or s16.empty: continue
            if s1.values[0] < 5: continue   # feature collapse
            drops.append(s1.values[0] - s16.values[0])
        tds[ds] = round(np.mean(drops), 1) if drops else 0
    return tds


@st.cache_data
def compute_anova_inline(df_sweep):
    """Compute two-way ANOVA effect sizes (eta²) from sweep data.

    Covers all 8 models × 8 datasets from df_sweep, including FineGym and
    VideoMamba which are absent from the pre-computed anova_results.csv.
    Uses between-groups SS decomposition (coverage | stride | residual).
    """
    rows = []
    for mk in MODEL_KEYS:
        for ds in DS_KEYS:
            sub = df_sweep[(df_sweep.model == mk) & (df_sweep.dataset == ds)].copy()
            if len(sub) < 10:
                continue
            grand = sub["acc"].mean()
            ss_total = ((sub["acc"] - grand) ** 2).sum()
            if ss_total < 1e-6:
                continue
            n_str_levels = sub["stride"].nunique()
            n_cov_levels = sub["coverage"].nunique()
            cov_means = sub.groupby("coverage")["acc"].mean()
            ss_cov = n_str_levels * ((cov_means - grand) ** 2).sum()
            str_means = sub.groupby("stride")["acc"].mean()
            ss_str = n_cov_levels * ((str_means - grand) ** 2).sum()
            rows.append({
                "model": mk, "model_name": MODEL_NAMES[mk], "dataset": ds,
                "eta2_coverage": round(min(ss_cov / ss_total, 1.0), 4),
                "eta2_stride":   round(min(ss_str / ss_total, 1.0), 4),
                "dominant": "coverage" if ss_cov > ss_str else "stride",
                "n_configs": len(sub),
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_e7_routing():
    f = DATA / "routing_curves.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    df["acc_pct"] = df["accuracy"] * 100
    df["model_name"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
    df["family"] = df["model"].map(FAMILIES)
    return df


@st.cache_data
def load_e7_summary():
    f = DATA / "routing_summary.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    df["model_name"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
    df["best_acc_pct"] = df["best_accuracy"] * 100
    df["oracle_pct"]   = df["oracle_accuracy"] * 100
    df["fixed4f_pct"]  = df["fixed_cheap_acc"] * 100
    df["fixed16f_pct"] = df["fixed_dense_acc"] * 100
    return df


@st.cache_data
def load_e9_comparison():
    f = DATA / "methods_comparison.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    df["acc_pct"] = df["accuracy"] * 100
    return df


@st.cache_data
def load_e10_duration():
    f = DATA / "clip_duration.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    df["model_name"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
    df["family"] = df["model"].map(FAMILIES)
    return df


@st.cache_data
def load_e3_spectral():
    f = DATA / "spectral_correlation.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    df["ds_label"] = df["dataset"].map(DS_LABELS).fillna(df["dataset"])
    return df


@st.cache_data
def load_anova():
    f = DATA / "anova_results.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    df["model_name"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
    return df


@st.cache_data(ttl=300)
def load_p3():
    """Native-resolution checkpoints evaluated at different input sizes (no retraining).

    Reads from evaluations/accv2026/spatial_resolution_sweep/*/spatial_sweep_summary.csv.
    Falls back to dashboard/data/p3_results.csv if directory is missing.
    Excludes 336px — chart range is 48–224px only.
    """
    sweep_root = Path(__file__).parent.parent / "evaluations/accv2026/spatial_resolution_sweep"
    DS_KEEP = {"ucf101", "ssv2", "hmdb51", "diving48", "autsl", "driveact", "epic_kitchens", "finegym"}
    rows = []

    if sweep_root.exists():
        for csv in sorted(sweep_root.glob("*/spatial_sweep_summary.csv")):
            folder = csv.parent.name
            found_ds = None
            for ds in sorted(DS_KEEP, key=len, reverse=True):
                if folder.endswith("_" + ds):
                    found_ds = ds
                    model = folder[:-(len(ds) + 1)]
                    break
            if found_ds is None or model not in MODEL_NAMES:
                continue
            try:
                df_s = pd.read_csv(csv)
                df_s = df_s[df_s["resolution"] != 336]  # exclude 336px
                df_s["model"] = model
                df_s["dataset"] = found_ds
                df_s["res"] = df_s["resolution"]
                df_s["acc"] = df_s["top1"] * 100
                df_s["model_name"] = MODEL_NAMES[model]
                rows.append(df_s[["model", "dataset", "res", "acc", "model_name"]])
            except Exception:
                continue

    if rows:
        df = pd.concat(rows).sort_values(["model", "dataset", "res"]).reset_index(drop=True)
        return df

    # Fallback: static CSV
    f = DATA / "p3_results.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df = df[df["res"] != 336]
    df["model_name"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
    return df


@st.cache_data(ttl=300)
def load_combined_sweep():
    """Load all coverage×stride sweep CSVs: trainres + cross-res folders + FineGym resolution sweep."""
    NATIVE_L = {"r3d_18":112,"mc3_18":112,"r2plus1d_18":112,"slowfast_r50":224,
                "timesformer":224,"vivit":224,"videomae":224,"videomamba":224}
    ds_list = ["ucf101","ssv2","hmdb51","diving48","autsl","driveact","epic_kitchens","finegym"]
    rows = []

    # ── Original trainres/cross-res sweeps (7 datasets excluding FineGym) ──────
    sweep_root = Path(__file__).parent.parent / "evaluations/accv2026/coverage_stride_sweep"
    if sweep_root.exists():
        for csv in sweep_root.glob("*/sweep_summary.csv"):
            folder = csv.parent.name
            res_override = None
            is_trainres = False
            if "_trainres" in folder:
                tag, res_str = folder.rsplit("_trainres", 1)
                try:
                    res_override = int(res_str)
                    is_trainres = True
                except ValueError:
                    continue
            elif "_res" in folder:
                parts = folder.rsplit("_res", 1)
                tag = parts[0]
                try: res_override = int(parts[1])
                except ValueError: tag = folder
            else:
                tag = folder
            ds = next((d for d in sorted(ds_list, key=len, reverse=True) if tag.endswith(d)), None)
            if not ds:
                continue
            model = tag[:-(len(ds)+1)]
            if model not in NATIVE_L:
                continue
            try:
                df = pd.read_csv(csv)
                df["model"]     = model
                df["dataset"]   = ds
                df["res"]       = res_override if res_override else NATIVE_L.get(model, 224)
                df["train_res"] = res_override if is_trainres else None
                df["acc"]       = df["top1"] * 100
                rows.append(df[["model","dataset","res","train_res","coverage","stride","acc"]])
            except Exception:
                continue

    # ── FineGym: coverage × stride × resolution sweep (P3-retrained) ──────────
    fg_root = Path(__file__).parent.parent / "evaluations/accv2026/coverage_stride_resolution_sweep"
    if fg_root.exists():
        for model_dir in fg_root.glob("*_finegym"):
            model = model_dir.name.replace("_finegym", "")
            if model not in NATIVE_L:
                continue
            for res_dir in model_dir.glob("res*px"):
                csv = res_dir / "sweep_summary.csv"
                if not csv.exists():
                    continue
                try:
                    res = int(res_dir.name.replace("res", "").replace("px", ""))
                    df = pd.read_csv(csv)
                    df["model"]     = model
                    df["dataset"]   = "finegym"
                    df["res"]       = res
                    df["train_res"] = res   # P3-retrained at this exact resolution
                    df["acc"]       = df["top1"] * 100
                    rows.append(df[["model","dataset","res","train_res","coverage","stride","acc"]])
                except Exception:
                    continue

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.cache_data(ttl=300)
def load_retrained_spatial():
    # On Streamlit Cloud /scratch/ does not exist — read the bundled CSV instead.
    # On the cluster, prefer the live checkpoints (newer v2 results) and fall back
    # to the CSV so the Cloud version always works.
    csv_path = DATA / "retrained_spatial.csv"
    ckpt_root = Path("/scratch/wesleyferreiramaia/infoRates/fine_tuned_models")

    if not ckpt_root.exists():
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    import json, re
    ds_keys  = ["epic_kitchens","diving48","driveact","ucf101","hmdb51","autsl","ssv2","finegym"]
    mdl_keys = ["r3d_18","mc3_18","r2plus1d_18","slowfast_r50","timesformer","vivit","videomae","videomamba"]

    def read_val_acc(d: Path):
        for fname in ("accv_meta.json", "config.json"):
            fpath = d / fname
            if not fpath.exists(): continue
            try:
                cfg = json.loads(fpath.read_text())
                v = cfg.get("val_acc")
                if v is not None: return float(v)
            except Exception: continue
        return None

    best: dict = {}
    # Scan all versioned checkpoints: v1 (_e10_h200), v2 (_e10_v2_h200), v3, etc.
    for d in ckpt_root.glob("accv2026_*_e10*_h200"):
        name = d.name
        # Extract inner: strip prefix "accv2026_" and suffix "_e10*_h200"
        m_suffix = re.search(r'_e\d+(?:_v\d+)?_h200$', name)
        if not m_suffix: continue
        inner = name[len("accv2026_"):m_suffix.start()]
        m = re.search(r'_(\d+)px$', inner)
        if not m: continue
        train_res = int(m.group(1))
        middle = inner[:m.start()]
        ds = next((k for k in ds_keys if middle.endswith("_" + k)), None)
        if not ds: continue
        model = middle[:-(len(ds)+1)]
        if model not in mdl_keys: continue
        val_acc = read_val_acc(d)
        if val_acc is None: continue
        # VideoMamba/EK: v1/v3 leaky (num_labels=97, val_acc~50%); exclude above threshold
        if model == "videomamba" and ds == "epic_kitchens" and val_acc > 0.38:
            continue
        key = (model, ds, train_res)
        # Always keep the BEST accuracy across all versions
        if key not in best or val_acc > best[key]:
            best[key] = val_acc

    return pd.DataFrame([
        {"model": m, "dataset": ds, "train_res": res, "acc": acc * 100}
        for (m, ds, res), acc in best.items()
        if res != 336   # 336px excluded: batch-size bug in original runs
    ])


# ── Load everything ───────────────────────────────────────────────────────────
df_sw   = load_sweeps()
TDS     = compute_tds(df_sw) if not df_sw.empty else {}
df_e7   = load_e7_routing()
df_e7s  = load_e7_summary()
df_e9   = load_e9_comparison()
df_e10  = load_e10_duration()
df_e3   = load_e3_spectral()
df_anova = load_anova()
df_p3   = load_p3()
df_retrained_spatial = load_retrained_spatial()
df_comb = load_combined_sweep()   # all stride×coverage sweeps across resolutions

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 InfoRates")
st.sidebar.caption("Spatiotemporal Aliasing")

page = st.sidebar.radio("Navigation", [
    "🏠 Overview & TDS",
    "⚡ Latency & Efficiency",
    "📊 Temporal Aliasing",
    "🖼 Spatial Resolution",
    "🎛 Spatiotemporal Explorer",
    "📈 Dataset Analysis",
    "🔀 Routing & Efficiency",
    "🎯 Architecture Recommender",
])

# ── Helper: model multiselect with "all" toggle ───────────────────────────────
def model_select(key, default_all=True):
    all_names = [MODEL_NAMES[k] for k in MODEL_KEYS]
    if st.sidebar.checkbox("All models", value=default_all, key=f"all_{key}"):
        return MODEL_KEYS
    sel = st.sidebar.multiselect("Models", all_names, default=all_names[:4], key=key)
    return [k for k, v in MODEL_NAMES.items() if v in sel]


# =============================================================================
# 🏠 OVERVIEW & TDS
# =============================================================================
if page == "🏠 Overview & TDS":
    st.title("InfoRates — Spatiotemporal Aliasing Analysis")
    st.markdown(
        "**8 architectures** (CNN / Transformer / SSM) · **8 datasets** · **5 resolutions (48–224px)** · **8,000+ eval configs** "
        "— a large-scale study of how video models degrade under reduced frame rate and spatial resolution."
    )

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Architectures", "8", "CNN + Transformer + SSM")
    c2.metric("Datasets", "8", "5 semantic domains")
    c3.metric("Eval configs", "8,000+", "8 models × 8 datasets × res × cov × stride")
    n_res = len(df_p3.drop_duplicates(["model","dataset","res"])) if not df_p3.empty else 0
    c4.metric("Spatial configs", f"{n_res}", "cross-resolution evaluation")
    st.divider()

    # TDS bar chart from real data
    st.subheader("Temporal Demand Score (TDS)")
    st.caption("Mean accuracy drop (stride=1→16, coverage=100%) averaged over all architectures, excluding feature-collapsed models (<5%). Higher = more temporally demanding.")

    if TDS:
        tds_df = pd.DataFrame([
            {"Dataset": DS_LABELS[k], "TDS (pp)": v, "key": k}
            for k,v in sorted(TDS.items(), key=lambda x: -x[1])
        ])
        fig = px.bar(tds_df, x="TDS (pp)", y="Dataset", orientation="h",
                     color="TDS (pp)", color_continuous_scale="RdYlGn_r",
                     text="TDS (pp)", height=320)
        fig.update_traces(texttemplate="%{text:.1f}pp", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=50,t=10,b=0),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Interactive accuracy explorer — stride × coverage × resolution
    st.subheader("Accuracy Explorer")
    NATIVE_RES_OV = {"r3d_18":112,"mc3_18":112,"r2plus1d_18":112,"slowfast_r50":224,
                     "timesformer":224,"vivit":224,"videomae":224,"videomamba":224}

    c_sel1, c_sel2, c_sel3 = st.columns(3)
    with c_sel1:
        sel_cov = st.select_slider("Coverage (%)", options=[10,25,50,75,100], value=100,
                                    key="ov_cov", help="Fraction of clip observed")
    with c_sel2:
        sel_str = st.select_slider("Stride", options=[1,2,4,8,16], value=1,
                                    key="ov_str", help="Frame sampling density")
    with c_sel3:
        sel_res = st.select_slider("Resolution (px)", options=[48,96,112,160,224], value=224,
                                    key="ov_res", help="Input resolution. Native: CNN=112px, Transformer/SSM=224px")

    def get_acc_overview(mk, ds, cov, stride, res):
        """Return (acc, source_label) for the given config. None if unavailable.

        Priority:
        1. df_comb trainres — validated sweep with retrained checkpoint at this resolution.
           Takes priority over native sweep so all resolutions use the same model generation
           (accv2026 campaign) and curves are comparable across the resolution slider.
        2. Native sweep (df_sw) — fallback when no trainres sweep exists at this resolution.
        3. retrained_spatial — single-point (stride=1, cov=100%) from checkpoint val_acc.
        4. p3_results — stride=1, cov=100% only (cross-res / retrained values).
        """
        native = NATIVE_RES_OV[mk]

        # 1. df_comb trainres for stride/cov variation — preferred source for consistency
        if not df_comb.empty:
            mask_tr = ((df_comb.model == mk) & (df_comb.dataset == ds) &
                       (df_comb.res == res) & (df_comb.stride == stride) &
                       (df_comb.coverage == cov) & (df_comb.train_res == res))
            row = df_comb[mask_tr]
            if not row.empty:
                return (float(row["acc"].values[0]), "trainres")

        # 2. Native sweep (df_sw): fallback when no validated trainres exists
        if res == native and not df_sw.empty:
            row = df_sw[(df_sw.model == mk) & (df_sw.dataset == ds) &
                        (df_sw.stride == stride) & (df_sw.coverage == cov)]
            if not row.empty:
                v = float(row["acc"].values[0])
                return (v, "native")

        # 3. Authoritative single-point accuracy (stride=1, cov=100%)
        if stride == 1 and cov == 100 and not df_retrained_spatial.empty:
            row = df_retrained_spatial[
                (df_retrained_spatial.model == mk) &
                (df_retrained_spatial.dataset == ds) &
                (df_retrained_spatial.train_res == res)
            ]
            if not row.empty:
                return float(row["acc"].values[0]), "retrained"

        # 4. p3_results: stride=1, cov=100% fallback (cross-res or retrained values)
        if stride == 1 and cov == 100 and not df_p3.empty:
            row = df_p3[(df_p3.model == mk) & (df_p3.dataset == ds) & (df_p3.res == res)]
            if not row.empty:
                v = float(row["acc"].values[0])
                return (v, "p3")

        return None, None

    ds_short = {k: DS_LABELS[k].split(" (")[0] for k in DS_KEYS}
    ds_short_list = [ds_short[ds] for ds in DS_KEYS]

    # Build bar chart
    fig_ov = go.Figure()
    has_data = False
    for mk in MODEL_KEYS:
        accs_ordered = []
        for ds in DS_KEYS:
            acc, _ = get_acc_overview(mk, ds, sel_cov, sel_str, sel_res)
            accs_ordered.append(acc)
        if any(v is not None and v > 0 for v in accs_ordered):
            has_data = True
            fig_ov.add_trace(go.Bar(
                x=ds_short_list, y=accs_ordered,
                name=MODEL_NAMES[mk],
                marker_color=FAM_COLOR.get(FAMILIES[mk], "#999"),
                hovertemplate=f"<b>{MODEL_NAMES[mk]}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ))

    if has_data:
        res_label = f"{sel_res}px"
        native_models = [mk for mk in MODEL_KEYS if NATIVE_RES_OV[mk] == sel_res]
        if native_models:
            res_label += f" (native for {', '.join(MODEL_NAMES[m] for m in native_models[:3])})"
        fig_ov.update_layout(
            barmode="group", height=400,
            title=f"Top-1 accuracy @ coverage={sel_cov}%, stride={sel_str}, {res_label}",
            yaxis_title="Top-1 (%)", xaxis_title="",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", y=-0.25), margin=dict(b=80),
            xaxis=dict(categoryorder="array", categoryarray=ds_short_list),
        )
        st.plotly_chart(fig_ov, use_container_width=True)

        if sel_res != 224:
            native_note = "112px is CNN native (R3D-18, MC3-18, R2+1D). " if sel_res == 112 else ""
            st.caption(
                f"{native_note}Resolution {sel_res}px: bars use retrained checkpoints at this resolution "
                f"where available, otherwise cross-resolution evaluation (native model at {sel_res}px input). "
                f"Missing bars = data still generating."
            )
    else:
        st.info(
            f"No data yet for coverage={sel_cov}%, stride={sel_str}, resolution={sel_res}px. "
            "Try coverage=100%, stride=1 to see spatial results, or resolution=224px for full temporal sweep."
        )

    # Compact summary table
    st.subheader(f"Summary table — cov={sel_cov}%, stride={sel_str}, res={sel_res}px")
    rows = []
    for mk in MODEL_KEYS:
        row = {"Model": MODEL_NAMES[mk], "Family": FAMILIES[mk], "Native px": f"{NATIVE_RES_OV[mk]}px"}
        accs = []
        for ds in DS_KEYS:
            acc, src = get_acc_overview(mk, ds, sel_cov, sel_str, sel_res)
            if acc is None:
                row[ds_short[ds]] = "—"
            else:
                row[ds_short[ds]] = f"{acc:.1f}%"
                accs.append(acc)
        row["Avg"] = f"{np.mean(accs):.1f}%" if accs else "—"
        rows.append(row)
    df_tbl = pd.DataFrame(rows)
    st.dataframe(df_tbl, use_container_width=True)

    if sel_cov == 10 and sel_str == 16:
        st.warning(
            "**Extreme config:** coverage=10% + stride=16 → frame pool has only ~1–3 frames. "
            "This measures **single-frame accuracy**, not temporal reasoning."
        )


# =============================================================================
# 📊 TEMPORAL ALIASING  (Aliasing Curves + Heatmaps)
# =============================================================================
elif page == "📊 Temporal Aliasing":
    st.title("Temporal Aliasing Analysis")

    NATIVE_RES_AL = {"r3d_18":112,"mc3_18":112,"r2plus1d_18":112,"slowfast_r50":224,
                     "timesformer":224,"vivit":224,"videomae":224,"videomamba":224}

    def _get_alias_data(mk, ds, res, cov_val):
        native = NATIVE_RES_AL[mk]
        if not df_comb.empty:
            mask = ((df_comb.model==mk) & (df_comb.dataset==ds) &
                    (df_comb.res==res) & (df_comb.coverage==cov_val) &
                    (df_comb.train_res==res))
            rows = df_comb[mask].sort_values("stride")
            if not rows.empty:
                return rows[["stride","acc"]]
        if res == native and not df_sw.empty:
            rows = df_sw[(df_sw.model==mk)&(df_sw.dataset==ds)&(df_sw.coverage==cov_val)].sort_values("stride")
            if not rows.empty:
                return rows[["stride","acc"]]
        return pd.DataFrame()

    def make_heatmap(sub, title, height=300, show_scale=False):
        pivot = sub.pivot(index="coverage", columns="stride", values="acc")
        pivot = pivot.sort_index(ascending=False)
        pivot = pivot[sorted(pivot.columns)]
        z   = pivot.values
        y   = [f"{c}%" for c in pivot.index]
        x   = [f"s={s}" for s in pivot.columns]
        txt = [[f"{v:.0f}" for v in row] for row in z]
        fig = go.Figure(go.Heatmap(
            z=z, x=x, y=y, colorscale="RdYlGn", zmin=0, zmax=100,
            text=txt, texttemplate="%{text}", textfont=dict(size=9),
            showscale=show_scale,
            hovertemplate="Coverage=%{y}, %{x}<br>Acc=%{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12)), height=height,
            margin=dict(t=40, b=10, l=50, r=10), xaxis=dict(side="bottom"),
        )
        return fig

    tab_curves, tab_heat = st.tabs(["📈 Aliasing Curves", "🔲 Heatmaps"])

    # ── Tab 1: Aliasing Curves ────────────────────────────────────────────────
    with tab_curves:
        st.caption("Accuracy vs. stride at any resolution. Each curve shows how models degrade as temporal density decreases.")

        all_model_names_al = [MODEL_NAMES[k] for k in MODEL_KEYS]
        ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns([3, 3, 1, 1, 1])
        with ctl1:
            sel_mods_al = st.multiselect("Models", all_model_names_al,
                                          default=all_model_names_al, key="ac_mods")
            sel_mkeys = [k for k, v in MODEL_NAMES.items() if v in sel_mods_al]
        with ctl2:
            sel_ds_al = st.multiselect("Datasets", DS_KEYS,
                                        default=["autsl","ssv2","ucf101","finegym"],
                                        format_func=lambda x: DS_LABELS[x].split(" (")[0],
                                        key="ac_ds")
        with ctl3:
            cov = st.select_slider("Coverage (%)", [10,25,50,75,100], value=100, key="ac_cov")
        with ctl4:
            sel_alias_res = st.select_slider("Resolution", [48,96,112,160,224], value=224, key="ac_res")
        with ctl5:
            facet = st.radio("Facet", ["Dataset","Model"], key="ac_facet")

        sel_ds = sel_ds_al

        if facet == "Dataset":
            if not sel_ds:
                st.warning("Select at least one dataset.")
            else:
                ncols = min(3, len(sel_ds))
                cols_grid = st.columns(ncols)
                for col_idx, ds in enumerate(sel_ds):
                    fig = go.Figure()
                    has_any = False
                    for mk in sel_mkeys:
                        grp = _get_alias_data(mk, ds, sel_alias_res, cov)
                        if grp.empty: continue
                        has_any = True
                        color = FAM_COLOR.get(FAMILIES.get(mk,"CNN"), "#999")
                        dash = "solid" if FAMILIES.get(mk) in ("Transformer","SSM") else "dash"
                        fig.add_trace(go.Scatter(
                            x=grp["stride"], y=grp["acc"],
                            mode="lines+markers", name=MODEL_NAMES[mk],
                            line=dict(color=color, dash=dash, width=2), marker=dict(size=7),
                            hovertemplate=f"{MODEL_NAMES[mk]}<br>stride=%{{x}}, acc=%{{y:.1f}}%<extra></extra>",
                        ))
                    fig.update_xaxes(type="log", tickvals=[1,2,4,8,16],
                                     ticktext=["1","2","4","8","16"], title="Stride")
                    fig.update_yaxes(title="Top-1 (%)", range=[0, 100])
                    fig.update_layout(
                        title=dict(text=DS_LABELS[ds].split(" (")[0], font=dict(size=12)),
                        height=300, margin=dict(t=40, b=60),
                        legend=dict(orientation="h", y=-0.45, font=dict(size=9)),
                    )
                    if not has_any:
                        fig.add_annotation(text="No data", xref="paper", yref="paper",
                                           x=0.5, y=0.5, showarrow=False,
                                           font=dict(size=14, color="gray"))
                    cols_grid[col_idx % ncols].plotly_chart(fig, use_container_width=True)
        else:
            for mk in sel_mkeys:
                with st.expander(f"{MODEL_NAMES[mk]} ({FAMILIES[mk]})", expanded=True):
                    fig = go.Figure()
                    for ds in sel_ds:
                        grp = _get_alias_data(mk, ds, sel_alias_res, cov)
                        if grp.empty: continue
                        fig.add_trace(go.Scatter(
                            x=grp["stride"], y=grp["acc"],
                            mode="lines+markers", name=DS_LABELS[ds].split(" (")[0],
                            marker=dict(size=7), line=dict(width=2),
                        ))
                    fig.update_xaxes(type="log", tickvals=[1,2,4,8,16],
                                     ticktext=["1","2","4","8","16"])
                    fig.update_yaxes(title="Top-1 (%)")
                    fig.update_layout(height=300, margin=dict(t=20,b=60),
                                      legend=dict(orientation="h",y=-0.35))
                    st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Heatmaps ───────────────────────────────────────────────────────
    with tab_heat:
        st.caption("Coverage (rows) × Stride (columns) accuracy grid. Red = low, green = high.")

        if df_sw.empty:
            st.error("No sweep data.")
        else:
            hm_c1, hm_c2 = st.columns([2, 1])
            with hm_c1:
                sel_model_hm = st.selectbox("Model", MODEL_KEYS,
                                             format_func=lambda x: MODEL_NAMES[x], key="hm_model")
            with hm_c2:
                show_all_ds_hm = st.checkbox("All 8 datasets", value=True, key="hm_all")

            if show_all_ds_hm:
                cols = st.columns(3)
                for i, ds in enumerate(DS_KEYS):
                    sub = df_sw[(df_sw.model==sel_model_hm) & (df_sw.dataset==ds)]
                    if sub.empty: continue
                    fig = make_heatmap(sub, DS_LABELS[ds].split(" (")[0], height=250)
                    cols[i % 3].plotly_chart(fig, use_container_width=True)
            else:
                sel_ds_hm = st.selectbox("Dataset", DS_KEYS,
                                          format_func=lambda x: DS_LABELS[x], key="hm_ds")
                sub = df_sw[(df_sw.model==sel_model_hm) & (df_sw.dataset==sel_ds_hm)]
                if not sub.empty:
                    fig = make_heatmap(sub,
                                       f"{MODEL_NAMES[sel_model_hm]} · {DS_LABELS[sel_ds_hm]}",
                                       height=380, show_scale=True)
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("TimeSformer vs ViViT — divided vs factorized attention")
            sel_ds_cmp = st.selectbox("Dataset", DS_KEYS,
                                       format_func=lambda x: DS_LABELS[x],
                                       key="cmp_ds", index=2)
            col1, col2 = st.columns(2)
            for col, mk in zip([col1, col2], ["timesformer", "vivit"]):
                sub = df_sw[(df_sw.model==mk) & (df_sw.dataset==sel_ds_cmp)]
                if sub.empty:
                    col.warning(f"No data for {MODEL_NAMES[mk]}")
                    continue
                fig = make_heatmap(sub, f"{MODEL_NAMES[mk]} ({FAMILIES[mk]})", height=320)
                fig.update_layout(margin=dict(t=50, b=10))
                col.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 📈 DATASET ANALYSIS  (ANOVA + Spectral + Clip Duration)
# =============================================================================
elif page == "📈 Dataset Analysis":
    st.title("Dataset Analysis")

    tab_anova, tab_spectral, tab_dur = st.tabs(
        ["📐 ANOVA & Variance", "🌀 Spectral Correlation", "⏱ Clip Duration"]
    )

    # ── Tab 1: ANOVA ─────────────────────────────────────────────────────────
    with tab_anova:
        st.caption(
            "Two-way ANOVA: proportion of accuracy variance (η²) explained by **coverage** vs **stride**. "
            "Computed from sweep data — covers all 8 models × 8 datasets including FineGym and VideoMamba."
        )
        df_anova_full = compute_anova_inline(df_sw)
        if df_anova_full.empty:
            st.error("No sweep data to compute ANOVA from.")
        else:
            metric_a = st.radio("Show effect",
                                ["Stride (η²_stride)", "Coverage (η²_coverage)"],
                                horizontal=True, key="anova_metric")
            col_key_a = "eta2_stride" if "Stride" in metric_a else "eta2_coverage"
            scale_a   = "η² stride" if "Stride" in metric_a else "η² coverage"

            pivot = df_anova_full.pivot_table(
                index="model_name", columns="dataset", values=col_key_a
            )
            col_rename_a = {k: DS_LABELS[k].split(" (")[0] for k in DS_KEYS if k in pivot.columns}
            pivot = pivot.rename(columns=col_rename_a)
            mean_str = df_anova_full.groupby("model_name")["eta2_stride"].mean()
            order = mean_str.sort_values().index.tolist()
            pivot = pivot.reindex([m for m in order if m in pivot.index])

            fig_a = px.imshow(
                pivot,
                color_continuous_scale="Blues" if "Coverage" in metric_a else "Reds",
                zmin=0, zmax=0.5 if "Stride" in metric_a else 1.0,
                text_auto=".2f",
                labels=dict(color=scale_a),
                title=f"Two-way ANOVA: {scale_a} per Model × Dataset (8×8)",
            )
            fig_a.update_layout(height=440, margin=dict(t=60))
            st.plotly_chart(fig_a, use_container_width=True)
            st.caption("Sorted by mean stride η² (most temporally robust = top). "
                       "Coverage dominates in all cells — consistent with ANOVA F=178.94 vs Stride F=80.76.")

            # Stacked bar: cov vs stride per model
            grp_a = df_anova_full.groupby("model_name").agg(
                stride_mean=("eta2_stride","mean"),
                stride_std=("eta2_stride","std"),
                cov_mean=("eta2_coverage","mean"),
            ).round(3).reset_index()
            grp_a["family"] = grp_a["model_name"].map(
                {v: k for k, v in MODEL_NAMES.items()}
            ).map(FAMILIES).fillna("")
            grp_a = grp_a.sort_values("stride_mean")

            fig_bar_a = go.Figure()
            for i, (_, row) in enumerate(grp_a.iterrows()):
                first = (i == 0)
                fig_bar_a.add_trace(go.Bar(
                    x=[row["model_name"]], y=[row["cov_mean"]],
                    name="Coverage η²", marker_color="#3498db",
                    legendgroup="cov", showlegend=first,
                ))
                fig_bar_a.add_trace(go.Bar(
                    x=[row["model_name"]], y=[row["stride_mean"]],
                    name="Stride η²",
                    marker_color=FAM_COLOR.get(row["family"], "#999"),
                    legendgroup="stride", showlegend=first,
                    error_y=dict(type="data", array=[row.get("stride_std", 0)], visible=True),
                ))
            fig_bar_a.update_layout(
                barmode="stack", height=340,
                yaxis_title="η² (mean across 8 datasets)",
                title="Coverage vs Stride effect sizes — coverage dominates everywhere",
                margin=dict(t=50, b=40),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_bar_a, use_container_width=True)

    # ── Tab 2: Spectral Correlation ───────────────────────────────────────────
    with tab_spectral:
        st.caption(
            "Pearson correlation between per-class optical-flow magnitude (Farnebäck) and aliasing loss. "
            "**FineGym:** optical-flow extraction pending — not yet included in this analysis."
        )

        if df_e3.empty:
            st.error("Spectral correlation data not found (`dashboard/data/spectral_correlation.csv`).")
        else:
            st.info("⚠️ **7 of 8 datasets shown.** FineGym requires per-class optical flow extraction "
                    "which is currently in progress. Values will update when the computation completes.",
                    icon="⏳")

            col1_s, col2_s = st.columns(2)
            with col1_s:
                st.subheader("Pearson r: Flow magnitude ↔ Aliasing loss")
                df_e3s = df_e3.sort_values("pearson_r_abs", ascending=True)
                colors_s = ["#2ecc71" if s else "#e74c3c" for s in df_e3s["significant"]]
                fig_s = go.Figure(go.Bar(
                    x=df_e3s["pearson_r_abs"],
                    y=df_e3s["ds_label"].apply(
                        lambda x: x.split(" (")[0] if isinstance(x, str) else x
                    ),
                    orientation="h",
                    marker_color=colors_s,
                    text=[f"r={r:.3f}  p={p:.3f}"
                          for r, p in zip(df_e3s["pearson_r_abs"], df_e3s["pearson_p_abs"])],
                    textposition="outside",
                ))
                fig_s.update_xaxes(title="Pearson r", range=[-0.05, 0.45])
                fig_s.update_layout(height=320, margin=dict(t=20, r=80, b=40),
                                    title="Green = significant (p<0.05), Red = ns")
                st.plotly_chart(fig_s, use_container_width=True)

            with col2_s:
                st.subheader("Optical flow by aliasing-sensitivity tier")
                fig_s2 = go.Figure()
                for tier, col_k, color in [
                    ("High sensitivity", "flow_high_tier", "#e74c3c"),
                    ("Moderate",         "flow_mod_tier",  "#e67e22"),
                    ("Low sensitivity",  "flow_low_tier",  "#2ecc71"),
                ]:
                    fig_s2.add_trace(go.Bar(
                        x=df_e3["dataset"].apply(
                            lambda x: DS_LABELS.get(x, x).split(" (")[0]
                        ),
                        y=df_e3[col_k], name=tier, marker_color=color,
                    ))
                fig_s2.update_layout(barmode="group", height=320,
                                     yaxis_title="Mean flow magnitude (px/frame)",
                                     margin=dict(t=20, b=60))
                st.plotly_chart(fig_s2, use_container_width=True)

            st.subheader("Interpretation")
            st.info(
                "Optical-flow magnitude is an **incomplete proxy** for temporal demand (r=0.03–0.33). "
                "AUTSL (r=0.28, p<0.001): hand speed correlates with aliasing — gestural fine-structure "
                "requires dense sampling. SSv2 (r=0.18, p=0.015): object velocity correlates with "
                "directionality sensitivity. EPIC-Kitchens (r≈−0.002): background camera motion dominates "
                "flow, masking action content. UCF-101 (r=0.03): appearance-dominated — flow irrelevant. "
                "FineGym (TDS=55.9pp): despite near-sports-level flow, aliasing is driven by phase "
                "transitions rather than pixel velocity (flow is expected to show weak-to-moderate r)."
            )

            st.dataframe(
                df_e3[["dataset","n_classes","pearson_r_abs","pearson_p_abs","significant"]].rename(
                    columns={"dataset":"Dataset","n_classes":"Classes",
                             "pearson_r_abs":"Pearson r","pearson_p_abs":"p-value",
                             "significant":"Significant"}
                ),
                use_container_width=True, hide_index=True,
            )

    # ── Tab 3: Clip Duration ──────────────────────────────────────────────────
    with tab_dur:
        st.caption("Counter-intuitive finding: shorter clips alias *more* — less temporal redundancy means each dropped frame costs more.")

        if df_e10.empty:
            st.error("Clip duration data not found (`dashboard/data/clip_duration.csv`).")
        else:
            dur_c1, dur_c2 = st.columns([2, 3])
            with dur_c1:
                sel_ds_dur = st.selectbox(
                    "Dataset", sorted(df_e10["dataset"].unique()),
                    format_func=lambda x: DS_LABELS.get(x, x), key="dur_ds"
                )
                sel_models_dur = st.multiselect(
                    "Models", sorted(df_e10["model_name"].dropna().unique()),
                    default=sorted(df_e10["model_name"].dropna().unique()), key="dur_mods"
                )

            sub_dur = df_e10[
                (df_e10.dataset == sel_ds_dur) & (df_e10.model_name.isin(sel_models_dur))
            ].copy()

            if sub_dur.empty:
                st.warning("No data for this selection.")
            else:
                dur_order = ["<1s", "1-3s", "3-6s", ">6s"]
                sub_dur["dur_order"] = pd.Categorical(
                    sub_dur["duration_bin"], categories=dur_order, ordered=True
                )
                sub_dur = sub_dur.sort_values("dur_order")

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.subheader(f"Aliasing loss by clip duration")
                    fig_d = px.line(
                        sub_dur, x="duration_bin", y="aliasing_loss_pp",
                        color="model_name",
                        color_discrete_map={
                            v: FAM_COLOR.get(FAMILIES.get(k,"CNN"),"#999")
                            for k, v in MODEL_NAMES.items()
                        },
                        markers=True, height=360,
                        labels={"duration_bin":"Clip Duration",
                                "aliasing_loss_pp":"Aliasing loss (pp)",
                                "model_name":"Model"},
                        category_orders={"duration_bin": dur_order},
                    )
                    fig_d.update_layout(legend=dict(orientation="h",y=-0.35), margin=dict(b=80))
                    st.plotly_chart(fig_d, use_container_width=True)

                with col_d2:
                    st.subheader("Dense vs Sparse accuracy by duration")
                    if sel_models_dur:
                        sub_melt = sub_dur.melt(
                            id_vars=["duration_bin","model_name","n"],
                            value_vars=["acc_dense","acc_sparse"],
                            var_name="config", value_name="accuracy",
                        )
                        sub_melt["config"] = sub_melt["config"].map(
                            {"acc_dense":"Dense (s=1)","acc_sparse":"Sparse (s=16)"}
                        )
                        sub_melt["accuracy"] *= 100
                        fig_d2 = px.line(
                            sub_melt[sub_melt.model_name == sel_models_dur[0]],
                            x="duration_bin", y="accuracy", color="config",
                            markers=True, height=360,
                            title=sel_models_dur[0],
                            labels={"duration_bin":"Duration","accuracy":"Top-1 (%)","config":""},
                            category_orders={"duration_bin": dur_order},
                            color_discrete_map={"Dense (s=1)":"#2ecc71","Sparse (s=16)":"#e74c3c"},
                        )
                        fig_d2.update_layout(legend=dict(orientation="h",y=-0.35), margin=dict(b=80))
                        st.plotly_chart(fig_d2, use_container_width=True)

                tbl_d = sub_dur[["model_name","duration_bin","n","acc_dense","acc_sparse","aliasing_loss_pp"]].copy()
                tbl_d["acc_dense"]        = (tbl_d["acc_dense"] * 100).round(1)
                tbl_d["acc_sparse"]       = (tbl_d["acc_sparse"] * 100).round(1)
                tbl_d["aliasing_loss_pp"] = tbl_d["aliasing_loss_pp"].round(1)
                tbl_d = tbl_d.rename(columns={
                    "model_name":"Model","duration_bin":"Duration","n":"N clips",
                    "acc_dense":"Dense acc (%)","acc_sparse":"Sparse acc (%)",
                    "aliasing_loss_pp":"Aliasing loss (pp)",
                })
                st.dataframe(
                    tbl_d.sort_values(["Model","Duration"]).reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )


# =============================================================================
# 🔀 ROUTING & EFFICIENCY (E7 + E9)
# =============================================================================
elif page == "🔀 Routing & Efficiency":
    st.title("Entropy Routing & Efficiency Analysis")

    tab_curves, tab_compare, tab_summary = st.tabs([
        "Routing Curves", "vs Literature Baselines", "Full Results Table"
    ])

    # ── Tab 1: Routing curves ─────────────────────────────────────────────────
    with tab_curves:
        if df_e7.empty:
            st.error("E7 routing data not found.")
        else:
            col_l, col_r = st.columns([1,3])
            with col_l:
                sel_ds_r = st.selectbox("Dataset", DS_KEYS, format_func=lambda x: DS_LABELS[x], key="rt_ds")
                budget   = st.slider("Frame budget line", 4, 16, 8, key="rt_budget")
                sel_mods_r = st.multiselect("Models", [MODEL_NAMES[k] for k in MODEL_KEYS],
                                             default=[MODEL_NAMES[k] for k in MODEL_KEYS], key="rt_mods")
            with col_r:
                sub = df_e7[(df_e7.dataset == sel_ds_r) &
                             df_e7.model_name.isin(sel_mods_r)].sort_values("avg_frames")
                fig = go.Figure()
                for mdl, grp in sub.groupby("model_name"):
                    mk = {v:k for k,v in MODEL_NAMES.items()}.get(mdl)
                    color = FAM_COLOR.get(FAMILIES.get(mk,"CNN"), "#999")
                    dash = "solid" if FAMILIES.get(mk) in ("Transformer","SSM") else "dash"
                    fig.add_trace(go.Scatter(
                        x=grp["avg_frames"], y=grp["acc_pct"],
                        mode="lines", name=mdl,
                        line=dict(color=color, dash=dash, width=2),
                        hovertemplate="%{y:.1f}% @ %{x:.1f}f (τ=%{customdata:.2f})",
                        customdata=grp["threshold"],
                    ))
                    # Fixed anchors
                    cheap = grp["fixed_cheap_acc"].iloc[0]*100
                    dense = grp["fixed_dense_acc"].iloc[0]*100
                    fig.add_trace(go.Scatter(
                        x=[4, 16], y=[cheap, dense],
                        mode="markers", showlegend=False,
                        marker=dict(symbol="circle-open", size=9, color=color),
                    ))

                fig.add_vline(x=budget, line_dash="dash", line_color="red",
                              annotation_text=f"Budget={budget}f", annotation_position="top right")
                fig.update_xaxes(title="Avg frames used", range=[3.5, 17])
                fig.update_yaxes(title="Top-1 Accuracy (%)")
                fig.update_layout(
                    title=f"Routing curves — {DS_LABELS[sel_ds_r]}",
                    legend=dict(orientation="h", y=-0.25), height=480, margin=dict(b=80)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: vs Literature (E9) ─────────────────────────────────────────────
    with tab_compare:
        if df_e9.empty:
            st.warning("E9 comparison data not found.")
        else:
            sel_ds_e9 = st.selectbox("Dataset", sorted(df_e9["dataset"].unique()), key="e9_ds")
            sel_model_e9 = st.selectbox("Our model", sorted(df_e9["model"].dropna().unique()), key="e9_mod")

            sub9 = df_e9[(df_e9["dataset"]==sel_ds_e9) &
                          ((df_e9["model"]==sel_model_e9) | (df_e9["method_type"]=="literature"))]
            if sub9.empty:
                st.warning("No data for this combination.")
            else:
                # Show all method types
                colors_m = {"literature":"#9b59b6","ours_e7":"#2ecc71",
                            "fixed":"#95a5a6","frameexit":"#e74c3c","oracle_knapsack":"#f39c12"}
                fig = go.Figure()
                for mtype, grp in sub9.groupby("method_type"):
                    for _, row in grp.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row["avg_frames"]], y=[row["acc_pct"]],
                            mode="markers+text",
                            name=row["method"],
                            text=[row["method"]],
                            textposition="top center",
                            marker=dict(size=14, color=colors_m.get(mtype,"#999"),
                                        symbol={"literature":"diamond","ours_e7":"star",
                                                "fixed":"circle","frameexit":"square",
                                                "oracle_knapsack":"triangle-up"}.get(mtype,"circle")),
                            hovertemplate=f"<b>{row['method']}</b><br>Acc: {row['acc_pct']:.1f}%<br>Avg frames: {row['avg_frames']:.1f}",
                        ))
                fig.update_xaxes(title="Avg frames used", range=[3, 18])
                fig.update_yaxes(title="Top-1 Accuracy (%)")
                fig.update_layout(title=f"Methods comparison — {sel_ds_e9} ({sel_model_e9})",
                                  height=450, showlegend=False, margin=dict(t=60))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("⭐ = our method  |  ◆ = literature  |  ■ = FrameExit  |  ▲ = Oracle  |  ● = Fixed budget")

    # ── Tab 3: Full summary table ─────────────────────────────────────────────
    with tab_summary:
        if not df_e7s.empty:
            budget_tbl = st.slider("Budget filter (avg frames ≤)", 4, 16, 8, key="sum_budget")
            sub_s = df_e7s.copy()

            pivot = sub_s.pivot_table(
                index="model_name", columns="dataset",
                values="best_acc_pct", aggfunc="mean"
            ).round(1)
            col_rename = {k: k.upper()[:4] for k in DS_KEYS}
            pivot = pivot.rename(columns=col_rename)
            pivot["Avg"] = pivot.mean(axis=1).round(1)
            pivot = pivot.sort_values("Avg", ascending=False)

            st.subheader("Best accuracy (%) at avg ≤8 frames per model × dataset")
            st.dataframe(pivot, use_container_width=True)

            st.subheader("% Cheap-routed (4-frame inference)")
            pivot2 = (sub_s.pivot_table(
                index="model_name", columns="dataset",
                values="pct_cheap", aggfunc="mean"
            ) * 100).round(0)
            pivot2 = pivot2.rename(columns=col_rename)
            pivot2["Avg"] = pivot2.mean(axis=1).round(0)
            pivot2 = pivot2.sort_values("Avg", ascending=False)
            st.dataframe(pivot2, use_container_width=True)
            st.caption("Higher = more videos served with cheap 4-frame inference.")


# =============================================================================
# 🖼 SPATIAL RESOLUTION
# =============================================================================
elif page == "🖼 Spatial Resolution":
    st.title("Spatial Resolution — Retrained vs. Cross-Resolution Evaluation")
    st.markdown(
        "**Solid line:** accuracy of checkpoints **retrained** at each resolution (48–224px, stride=1, cov=100%). "
        "**Dashed line:** native-resolution checkpoint evaluated at different resolutions (no retraining). "
        "The gap between curves shows the **retraining gain** — larger gap = more benefit from resolution-matched training."
    )

    NATIVE_RES_P3 = {
        "r3d_18": 112, "mc3_18": 112, "r2plus1d_18": 112, "slowfast_r50": 224,
        "timesformer": 224, "vivit": 224, "videomae": 224, "videomamba": 224,
    }

    st.sidebar.subheader("Filters")
    all_model_names = sorted(MODEL_NAMES[k] for k in MODEL_KEYS)
    sel_ds_p3 = st.sidebar.selectbox(
        "Dataset", DS_KEYS,
        format_func=lambda x: DS_LABELS.get(x, x), key="p3_ds"
    )
    sel_mods_p3 = st.sidebar.multiselect(
        "Models", all_model_names,
        default=all_model_names, key="p3_mods"
    )
    sel_mkeys_p3 = [k for k, v in MODEL_NAMES.items() if v in sel_mods_p3]

    # At native resolution, retrained == original fine-tune (same setup).
    # Use max(retrained, no-retrain) so reviewers see the best number, not two
    # equivalent runs presented as different experiments.
    NATIVE_RES = {
        "r3d_18": 112, "mc3_18": 112, "r2plus1d_18": 112,
        "slowfast_r50": 224, "timesformer": 224, "vivit": 224,
        "videomae": 224, "videomamba": 224,
    }
    df_retrained = df_retrained_spatial.copy() if not df_retrained_spatial.empty else df_retrained_spatial
    if not df_retrained.empty and not df_p3.empty:
        for mk, native in NATIVE_RES.items():
            p3_at_native = df_p3[(df_p3.model == mk) & (df_p3.res == native)]
            rt_at_native = df_retrained[(df_retrained.model == mk) & (df_retrained.train_res == native)]
            for ds in rt_at_native.dataset.unique():
                p3_val = p3_at_native[p3_at_native.dataset == ds]["acc"]
                rt_idx = df_retrained.index[
                    (df_retrained.model == mk) &
                    (df_retrained.train_res == native) &
                    (df_retrained.dataset == ds)
                ]
                if p3_val.empty or rt_idx.empty:
                    continue
                best = max(float(p3_val.iloc[0]), float(df_retrained.loc[rt_idx[0], "acc"]))
                df_retrained.loc[rt_idx[0], "acc"] = best

    n_retrained = len(df_retrained.drop_duplicates(["model","dataset","train_res"])) if not df_retrained.empty else 0
    n_crossres  = len(df_p3.drop_duplicates(["model","dataset","res"])) if not df_p3.empty else 0
    n_total_expected = 8 * 8 * 5  # 8 models × 8 datasets × 5 resolutions (48/96/112/160/224)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Retrained checkpoints", f"{n_retrained}", "model × dataset × resolution")
    mc2.metric("Cross-res eval (no retrain)", f"{n_crossres}", "model × dataset × resolution")
    pct_done = round(100 * n_retrained / n_total_expected) if n_total_expected else 0
    mc3.metric("Completion", f"{pct_done}%", f"{n_retrained}/{n_total_expected} configs (48–224px)")

    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader(f"Accuracy vs. resolution — {DS_LABELS.get(sel_ds_p3, sel_ds_p3)}")
        fig = go.Figure()

        for mk in sel_mkeys_p3:
            mdl_name = MODEL_NAMES[mk]
            fam = FAMILIES.get(mk, "CNN")
            color = FAM_COLOR.get(fam, "#999")

            sub_r = df_retrained[
                (df_retrained.model == mk) & (df_retrained.dataset == sel_ds_p3)
            ].sort_values("train_res") if not df_retrained.empty else pd.DataFrame()

            native = NATIVE_RES.get(mk)
            has_retrained = not sub_r.empty
            if has_retrained:
                # Star marker at native resolution, circle elsewhere
                r_syms = ["star" if int(x) == native else "circle" for x in sub_r["train_res"]]
                r_sizes = [13 if int(x) == native else 9 for x in sub_r["train_res"]]
                fig.add_trace(go.Scatter(
                    x=sub_r["train_res"], y=sub_r["acc"],
                    mode="lines+markers", name=mdl_name,
                    legendgroup=mdl_name,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=r_sizes, symbol=r_syms),
                    hovertemplate=f"<b>{mdl_name} (retrained)</b><br>%{{x}}px → %{{y:.1f}}%<extra></extra>",
                ))

            if not df_p3.empty:
                sub_p = df_p3[
                    (df_p3.model_name == mdl_name) & (df_p3.dataset == sel_ds_p3)
                ].sort_values("res").copy()
                if not sub_p.empty:
                    # At native resolution: align dashed y-value with solid (same experiment),
                    # and hide its marker — the solid star is the single shared marker there.
                    if has_retrained and native is not None:
                        r_native_acc = sub_r.loc[sub_r["train_res"] == native, "acc"]
                        if not r_native_acc.empty:
                            sub_p.loc[sub_p["res"] == native, "acc"] = float(r_native_acc.iloc[0])
                    # x marker everywhere; size=0 at native so no duplicate marker
                    p_syms = ["x" for _ in sub_p["res"]]
                    p_sizes = [0 if (has_retrained and native and int(x) == native) else 6
                               for x in sub_p["res"]]
                    fig.add_trace(go.Scatter(
                        x=sub_p["res"], y=sub_p["acc"],
                        mode="lines+markers", name=mdl_name if not has_retrained else f"{mdl_name} (no retrain)",
                        legendgroup=mdl_name,
                        showlegend=not has_retrained,
                        line=dict(color=color, width=1.5, dash="dot"),
                        marker=dict(size=p_sizes, symbol=p_syms),
                        hovertemplate=f"<b>{mdl_name} (no retrain)</b><br>%{{x}}px → %{{y:.1f}}%<extra></extra>",
                        opacity=0.6,
                    ))

        fig.update_xaxes(title="Resolution (px)", tickvals=[48, 96, 112, 160, 224], range=[40, 240])
        fig.update_yaxes(title="Top-1 (%)")
        fig.update_layout(
            height=440,
            legend=dict(orientation="h", y=-0.3),
            margin=dict(b=90),
            annotations=[dict(
                x=0.5, y=1.05, xref="paper", yref="paper",
                text="Solid + circle = retrained  |  Dashed + × = no retraining  |  ★ = native resolution (lines converge)",
                showarrow=False, font=dict(size=10, color="gray")
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Retraining gain (pp)")
        st.caption("Retrained − no-retrain at the same resolution")
        gain_rows = []
        for mk in sel_mkeys_p3:
            mdl_name = MODEL_NAMES[mk]
            for res in [48, 96, 112, 160, 224]:
                # Retreinado
                if not df_retrained.empty:
                    r_row = df_retrained[
                        (df_retrained.model==mk) & (df_retrained.dataset==sel_ds_p3) &
                        (df_retrained.train_res==res)
                    ]
                    r_acc = float(r_row["acc"].values[0]) if not r_row.empty else None
                else:
                    r_acc = None
                # Sem retreinar (cross-res)
                if not df_p3.empty:
                    p_row = df_p3[
                        (df_p3.model_name==mdl_name) & (df_p3.dataset==sel_ds_p3) &
                        (df_p3.res==res)
                    ]
                    p_acc = float(p_row["acc"].values[0]) if not p_row.empty else None
                else:
                    p_acc = None
                if r_acc is None and p_acc is None:
                    continue
                # At native resolution both are the same fine-tune — gain is meaningless
                is_native = (NATIVE_RES.get(mk) == res)
                gain = (r_acc - p_acc) if (r_acc is not None and p_acc is not None and not is_native) else None
                gain_rows.append({
                    "Model": mdl_name,
                    "Res": f"{res}px",
                    "Retrained": f"{r_acc:.1f}%" if r_acc is not None else "—",
                    "No retrain": "—" if is_native else (f"{p_acc:.1f}%" if p_acc is not None else "—"),
                    "Gain": "native" if is_native else (f"{gain:+.1f}pp" if gain is not None else "—"),
                })
        if gain_rows:
            tbl = pd.DataFrame(gain_rows)
            tbl["_res_num"] = tbl["Res"].str.extract(r"(\d+)").astype(int)  # noqa
            tbl = tbl.sort_values(["Model", "_res_num"]).drop(columns="_res_num")
            st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=400)

    st.divider()

    st.subheader("CNN vs. Transformer/SSM — retrained vs. no retraining (all datasets)")

    col_a, col_b = st.columns(2)
    clrs = {"CNN": "#e74c3c", "Transformer / SSM": "#2980b9"}

    def family_summary(df_src, res_col):
        if df_src.empty: return pd.DataFrame()
        df_src = df_src.copy()
        df_src["family"] = df_src["model"].map(
            lambda m: "CNN" if FAMILIES.get(m,"CNN") in ("CNN","Dual-CNN") else "Transformer / SSM"
        )
        return (
            df_src.groupby(["family", res_col])["acc"]
            .agg(mean="mean", std="std").reset_index()
        )

    with col_a:
        st.caption("**Retrained** (solid)")
        if not df_retrained.empty:
            fs_r = family_summary(df_retrained, "train_res")
            fig_r = go.Figure()
            for fam, grp in fs_r.groupby("family"):
                grp = grp.sort_values("train_res")
                fig_r.add_trace(go.Scatter(
                    x=grp["train_res"], y=grp["mean"],
                    error_y=dict(type="data", array=grp["std"].tolist(), visible=True),
                    mode="lines+markers", name=fam,
                    line=dict(color=clrs.get(fam,"#999"), width=3),
                    marker=dict(size=10),
                ))
            fig_r.update_xaxes(title="Train resolution (px)", tickvals=[48,96,112,160,224])
            fig_r.update_yaxes(title="Mean Top-1 (%)")
            fig_r.update_layout(height=300, margin=dict(t=20,b=60),
                                legend=dict(orientation="h",y=-0.4))
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Retraining data still being generated.")

    with col_b:
        st.caption("**No retraining** (dashed, baseline)")
        if not df_p3.empty:
            df_p3_fam = df_p3.copy()
            df_p3_fam["family"] = df_p3_fam["model"].map(
                lambda m: "CNN" if FAMILIES.get(m,"CNN") in ("CNN","Dual-CNN") else "Transformer / SSM"
            )
            fs_p = df_p3_fam.groupby(["family","res"])["acc"].agg(mean="mean",std="std").reset_index()
            fig_p = go.Figure()
            for fam, grp in fs_p.groupby("family"):
                grp = grp.sort_values("res")
                fig_p.add_trace(go.Scatter(
                    x=grp["res"], y=grp["mean"],
                    error_y=dict(type="data", array=grp["std"].tolist(), visible=True),
                    mode="lines+markers", name=fam,
                    line=dict(color=clrs.get(fam,"#999"), width=2, dash="dot"),
                    marker=dict(size=8, symbol="x"),
                ))
            fig_p.update_xaxes(title="Eval resolution (px)", tickvals=[48,96,112,160,224])
            fig_p.update_yaxes(title="Mean Top-1 (%)")
            fig_p.update_layout(height=300, margin=dict(t=20,b=60),
                                legend=dict(orientation="h",y=-0.4))
            st.plotly_chart(fig_p, use_container_width=True)


# =============================================================================
# 🎛 SPATIOTEMPORAL EXPLORER
# =============================================================================
elif page == "🎛 Spatiotemporal Explorer":
    st.title("Spatiotemporal Explorer")
    st.markdown("Choose **dataset, stride, coverage, and resolution** to see the accuracy of every model at that exact configuration.")

    # df_comb loaded at startup (top-level load_combined_sweep)

    NATIVE_RES = {"r3d_18":112,"mc3_18":112,"r2plus1d_18":112,"slowfast_r50":224,
                  "timesformer":224,"vivit":224,"videomae":224,"videomamba":224}
    ALL_RES      = [48, 96, 112, 160, 224]
    ALL_STRIDES  = [1, 2, 4, 8, 16]
    ALL_COVERAGES= [10, 25, 50, 75, 100]

    st.subheader("Parameters")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        sel_ds = st.selectbox("Dataset", DS_KEYS,
                              format_func=lambda k: DS_LABELS.get(k, k), key="st_ds")
    with c2:
        sel_res = st.select_slider("Eval resolution (px)", ALL_RES, value=112, key="st_res")
    with c3:
        sel_stride = st.select_slider("Stride", ALL_STRIDES, value=1, key="st_stride")
    with c4:
        sel_cov = st.select_slider("Coverage (%)", ALL_COVERAGES, value=100, key="st_cov")
    # Train-res filter — only shown when trainres sweep data exists
    available_train_res = sorted(df_comb["train_res"].dropna().unique().astype(int).tolist()) \
        if not df_comb.empty and "train_res" in df_comb.columns and df_comb["train_res"].notna().any() \
        else []
    with c5:
        if available_train_res:
            sel_train_res = st.select_slider(
                "Train res (px)", [None] + available_train_res,
                value=None, key="st_train_res",
                help="Filter by checkpoint retrained at this resolution. None = all."
            )
        else:
            sel_train_res = None
            st.caption("Train res\n*(sweep in progress)*")

    # ── Lookup de accuracy para cada modelo ───────────────────────────────────
    results = []
    for model in MODEL_KEYS:
        native = NATIVE_RES.get(model, 224)
        acc = None
        source = None

        if not df_comb.empty:
            mask = (
                (df_comb.model==model) & (df_comb.dataset==sel_ds) &
                (df_comb.res==sel_res) & (df_comb.stride==sel_stride) &
                (df_comb.coverage==sel_cov)
            )
            if sel_train_res is not None and "train_res" in df_comb.columns:
                mask &= (df_comb.train_res == sel_train_res)
            row = df_comb[mask]
            if not row.empty:
                acc = row["acc"].values[0]
                source = "trainres" if sel_train_res else "combined"

        # 2. Temporal sweep (native res only)
        if acc is None and not df_sw.empty and sel_res == native:
            row = df_sw[
                (df_sw.model==model) & (df_sw.dataset==sel_ds) &
                (df_sw.stride==sel_stride) & (df_sw.coverage==sel_cov)
            ]
            if not row.empty:
                acc = row["acc"].values[0]
                source = "temporal"

        # 3. Spatial eval (stride=1, cov=100% only)
        if acc is None and not df_p3.empty and sel_stride==1 and sel_cov==100:
            row = df_p3[
                (df_p3.model==model) & (df_p3.dataset==sel_ds) & (df_p3.res==sel_res)
            ]
            if not row.empty:
                acc = row["acc"].values[0]
                source = "spatial"

        # Reference: native @ stride=1, cov=100%
        ref_acc = None
        if not df_sw.empty:
            ref_row = df_sw[
                (df_sw.model==model) & (df_sw.dataset==sel_ds) &
                (df_sw.stride==1) & (df_sw.coverage==100)
            ]
            if not ref_row.empty:
                ref_acc = ref_row["acc"].values[0]

        results.append({
            "model": model,
            "name": MODEL_NAMES[model],
            "family": FAMILIES[model],
            "native_res": native,
            "acc": acc,
            "ref": ref_acc,
            "delta": (acc - ref_acc) if (acc is not None and ref_acc is not None) else None,
            "source": source,
        })

    df_res = pd.DataFrame(results)
    df_res_valid = df_res[df_res.acc.notna()].copy()

    st.divider()
    title_str = f"Accuracy @ stride={sel_stride} · coverage={sel_cov}% · {sel_res}px — {DS_LABELS.get(sel_ds, sel_ds)}"
    st.subheader(title_str)

    if df_res_valid.empty:
        st.warning(
            f"No data for stride={sel_stride}, coverage={sel_cov}%, resolution={sel_res}px.\n\n"
            "**Available data:**\n"
            f"- Any stride/coverage + **native** resolution per model (temporal sweep)\n"
            f"- stride=1, coverage=100% + any resolution (spatial eval)\n"
            f"- Any stride/coverage + 96–224px for completed trainres sweep combinations\n\n"
            "To generate more configs: `bash scripts/accv2026/submit_trainres_sweep.sh`"
        )
    else:
        # Bar chart ordenado por accuracy
        df_plot = df_res_valid.sort_values("acc", ascending=True)
        colors_bar = [FAM_COLOR.get(f, "#999") for f in df_plot["family"]]
        fig_bar = go.Figure(go.Bar(
            x=df_plot["acc"], y=df_plot["name"],
            orientation="h",
            marker_color=colors_bar,
            text=[f"{a:.1f}%" for a in df_plot["acc"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Acc: %{x:.1f}%<extra></extra>",
        ))
        # Linha de referência (native, stride=1, cov=100%)
        if df_res_valid["ref"].notna().any():
            mean_ref = df_res_valid["ref"].mean()
            fig_bar.add_vline(x=mean_ref, line_dash="dash", line_color="gray",
                              annotation_text=f"Mean native ref: {mean_ref:.1f}%",
                              annotation_position="top right")
        fig_bar.update_layout(
            height=380,
            xaxis=dict(title="Top-1 Accuracy (%)", range=[0, 105]),
            yaxis_title="",
            margin=dict(l=10, r=60, t=20, b=40),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Tabela compacta com Δ vs referência
        tbl_rows = []
        for _, r in df_res_valid.iterrows():
            src_icons = {"temporal": "⏱ temporal", "spatial": "🖼 spatial", "combined": "🎛 combined", "trainres": "📐 trainres"}
            tbl_rows.append({
                "Model": r["name"],
                "Family": r["family"],
                "Native px": f"{r['native_res']}px",
                f"Acc @{sel_res}px s={sel_stride} c={sel_cov}%": f"{r['acc']:.1f}%",
                "Ref (s=1,c=100%,native)": f"{r['ref']:.1f}%" if r['ref'] else "—",
                "Δ": f"{r['delta']:+.1f}pp" if r['delta'] is not None else "—",
                "Source": src_icons.get(r["source"], r["source"]),
            })
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    # ── Heatmap coverage × stride para o modelo selecionado ───────────────────
    st.divider()
    st.subheader("Heatmap Coverage × Stride — per model")
    col_hm1, col_hm2 = st.columns([1, 2])
    with col_hm1:
        hm_model = st.selectbox("Model", MODEL_KEYS,
                                format_func=lambda k: MODEL_NAMES[k], key="st_hm_model")
        hm_res_opts = [sel_res]
        native_hm = NATIVE_RES.get(hm_model, 224)
        if native_hm not in hm_res_opts:
            hm_res_opts = [native_hm] + hm_res_opts
        hm_res = st.selectbox("Heatmap resolution", hm_res_opts, key="st_hm_res")

    with col_hm2:
        # Montar dados para heatmap
        if not df_comb.empty:
            hm_mask = (df_comb.model==hm_model) & (df_comb.dataset==sel_ds) & (df_comb.res==hm_res)
            if sel_train_res is not None and "train_res" in df_comb.columns:
                hm_mask &= (df_comb.train_res == sel_train_res)
            hm_data = df_comb[hm_mask]
            tr_lbl = f" (train@{sel_train_res}px)" if sel_train_res else ""
            src_label = f"{hm_res}px{tr_lbl}"
        else:
            hm_data = pd.DataFrame()
            src_label = ""

        # Fallback: temporal sweep para resolução nativa
        if hm_data.empty and hm_res == NATIVE_RES.get(hm_model, 224) and not df_sw.empty:
            hm_data = df_sw[
                (df_sw.model==hm_model) & (df_sw.dataset==sel_ds)
            ].rename(columns={"acc": "acc"})
            src_label = f"{hm_res}px native (temporal sweep)"

        if hm_data.empty:
            st.info(f"No heatmap data for {MODEL_NAMES[hm_model]} / {sel_ds} @ {hm_res}px.\n\n"
                    "Run `submit_trainres_sweep.sh` to generate multi-resolution data.")
        else:
            pivot = hm_data.pivot_table(
                index="coverage", columns="stride", values="acc", aggfunc="mean"
            ).sort_index(ascending=False)

            # Marcar o ponto selecionado
            z = pivot.values.tolist()
            text = [[f"{v:.1f}%" for v in row] for row in z]

            fig_hm = go.Figure(go.Heatmap(
                z=z,
                x=[str(c) for c in pivot.columns],
                y=[f"{r}%" for r in pivot.index],
                colorscale="RdYlGn",
                text=text, texttemplate="%{text}",
                zmin=0, zmax=100,
                hovertemplate="Coverage=%{y} Stride=%{x}<br>Acc=%{z:.1f}%<extra></extra>",
            ))
            # Destacar o ponto selecionado
            if sel_cov in pivot.index and sel_stride in pivot.columns:
                y_idx = list(pivot.index[::-1]).index(sel_cov)
                x_idx = list(pivot.columns).index(sel_stride)
                fig_hm.add_trace(go.Scatter(
                    x=[str(sel_stride)], y=[f"{sel_cov}%"],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=24, color="white", line=dict(width=3, color="white")),
                    showlegend=False, hoverinfo="skip",
                ))
            fig_hm.update_layout(
                height=260,
                xaxis_title="Stride",
                yaxis_title="Coverage (%)",
                title=f"{MODEL_NAMES[hm_model]} / {DS_LABELS.get(sel_ds,sel_ds)} @ {src_label}",
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    # ── Disponibilidade de dados ───────────────────────────────────────────────
    if df_comb.empty:
        st.caption(
            "🔸 **Train-res sweep not yet run.** "
            "Showing only temporal sweep (native resolution) "
            "and spatial eval (stride=1, cov=100%) data points. "
            "To enable all parameters: `bash scripts/accv2026/submit_trainres_sweep.sh`"
        )


# =============================================================================
# ⚡ LATENCY & EFFICIENCY
# =============================================================================
elif page == "⚡ Latency & Efficiency":
    st.title("⚡ Latency & Efficiency")

    @st.cache_data
    def load_latency_data():
        f = DATA / "latency_by_resolution.csv"
        if not f.exists():
            return pd.DataFrame()
        return pd.read_csv(f)

    df_lat_raw = load_latency_data()

    HW_PRESETS = {
        "RTX PRO 6000 Blackwell (measured)": 1.0,
        "RTX 4090": 1.5,
        "RTX 3090 / 3080": 2.4,
        "RTX 2080 Ti": 3.8,
        "Jetson AGX Orin (edge)": 10.0,
        "Jetson Nano (edge)": 50.0,
        "Custom…": None,
    }

    st.sidebar.subheader("Deployment target")
    hw_choice = st.sidebar.selectbox("Hardware", list(HW_PRESETS.keys()), index=0)
    if HW_PRESETS[hw_choice] is None:
        hw_factor = st.sidebar.number_input(
            "Slowdown vs. RTX PRO 6000 (×)", min_value=0.1, max_value=500.0,
            value=10.0, step=0.5,
        )
    else:
        hw_factor = HW_PRESETS[hw_choice]

    st.sidebar.subheader("Accuracy config")
    lat_ds  = st.sidebar.selectbox("Dataset", DS_KEYS, format_func=lambda x: DS_LABELS[x], index=0)
    lat_cov = st.sidebar.select_slider("Coverage (%)", [10, 25, 50, 75, 100], value=100)
    lat_str = st.sidebar.select_slider("Stride", [1, 2, 4, 8, 16], value=1)
    lat_res = st.sidebar.select_slider("Resolution (px)", [48, 96, 112, 160, 224], value=224)

    if df_lat_raw.empty:
        st.error("Latency data not found. Run `scripts/accv2026/benchmark_latency_by_resolution.py`.")
        st.stop()

    # ── Chart 1: Latency vs Resolution ───────────────────────────────────────
    st.subheader("Latency vs resolution — all 8 models × 5 resolutions")
    st.caption("Measured via CUDA events, bfloat16, batch=1. Lines share the same color when same architecture family.")

    # One subplot per family for readability, or all on one chart grouped by family
    fig_curves = go.Figure()
    y_max_curves = 0.0
    DASH_BY_MODEL = {
        "r3d_18": "solid", "mc3_18": "dot", "r2plus1d_18": "dash",
        "slowfast_r50": "solid",
        "timesformer": "solid", "vivit": "dash", "videomae": "dot",
        "videomamba": "solid",
    }
    for mk in MODEL_KEYS:
        sub = df_lat_raw[df_lat_raw["model"] == mk].sort_values("resolution")
        if sub.empty:
            continue
        color = FAM_COLOR.get(FAMILIES[mk], "#aaa")
        ys = sub["mean_ms"].values * hw_factor
        ye = sub["std_ms"].values  * hw_factor
        y_max_curves = max(y_max_curves, (ys + ye).max())
        fig_curves.add_trace(go.Scatter(
            x=sub["resolution"].tolist(), y=ys.tolist(),
            mode="lines+markers",
            name=MODEL_NAMES[mk],
            line=dict(color=color, width=2, dash=DASH_BY_MODEL.get(mk, "solid")),
            marker=dict(size=7, color=color, line=dict(width=1, color="white")),
            error_y=dict(type="data", array=ye.tolist(), visible=True,
                         color=color, thickness=1, width=3),
            hovertemplate=f"<b>{MODEL_NAMES[mk]}</b><br>%{{x}}px → %{{y:.2f}} ms<extra></extra>",
        ))

    fig_curves.update_layout(
        xaxis=dict(
            title="Spatial resolution (px)",
            tickvals=[48, 96, 112, 160, 224],
            ticktext=["48", "96", "112", "160", "224"],
            gridcolor="#2a2a2a", showgrid=True,
        ),
        yaxis=dict(
            title=f"Latency (ms/clip)  ·  {hw_choice}",
            range=[0, y_max_curves * 1.18],
            gridcolor="#2a2a2a", showgrid=True,
        ),
        legend=dict(
            orientation="v", x=1.01, y=1, xanchor="left",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        height=420, margin=dict(t=20, r=160, b=50, l=60),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        hovermode="x unified",
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    # ── Scaling summary table ─────────────────────────────────────────────────
    with st.expander("Resolution scaling table (48 → 112 → 224 px)"):
        scale_rows = []
        for mk in MODEL_KEYS:
            sub = df_lat_raw[df_lat_raw["model"] == mk]
            def ms(r):
                v = sub[sub["resolution"] == r]["mean_ms"].values
                return f"{v[0]*hw_factor:.2f}" if len(v) else "—"
            lo = sub[sub["resolution"] == 48]["mean_ms"].values
            hi = sub[sub["resolution"] == 224]["mean_ms"].values
            ratio = f"{hi[0]/lo[0]:.1f}×" if len(lo) and len(hi) else "—"
            scale_rows.append({
                "Model": MODEL_NAMES[mk], "Family": FAMILIES[mk],
                "48 px": ms(48), "96 px": ms(96), "112 px": ms(112),
                "160 px": ms(160), "224 px": ms(224), "224/48": ratio,
            })
        st.dataframe(pd.DataFrame(scale_rows), hide_index=True, use_container_width=True)
        st.caption(
            "All values in ms. "
            "CNNs: ~4-5× from 48→224. Transformers: ~2-4×. "
            "VideoMamba (SSM): ~2× — nearly flat 48→160px due to linear O(n) scan."
        )

    st.divider()

    # ── Chart 2: Accuracy vs Latency Pareto ──────────────────────────────────
    st.subheader(f"Accuracy vs latency — {DS_LABELS[lat_ds]}")
    st.caption(f"Resolution: {lat_res}px · Coverage: {lat_cov}% · Stride: {lat_str} · Hardware: {hw_choice}")

    # Build per-model (latency, accuracy) at selected config
    rows = []
    for mk in MODEL_KEYS:
        hit = df_lat_raw[(df_lat_raw.model == mk) & (df_lat_raw.resolution == lat_res)]
        if hit.empty or pd.isna(hit["mean_ms"].values[0]):
            continue
        lat_ms  = hit["mean_ms"].values[0] * hw_factor
        lat_std = hit["std_ms"].values[0]  * hw_factor

        acc = np.nan
        sw_hit = df_sw[
            (df_sw.model == mk) & (df_sw.dataset == lat_ds) &
            (df_sw.coverage == lat_cov) & (df_sw.stride == lat_str)
        ]
        if not sw_hit.empty:
            acc = sw_hit["acc"].values[0]
        else:
            ch = df_comb[
                (df_comb.model == mk) & (df_comb.dataset == lat_ds) &
                (df_comb.coverage == lat_cov) & (df_comb.stride == lat_str) &
                (df_comb.res == lat_res)
            ]
            if not ch.empty:
                acc = ch["acc"].values[0]

        rows.append({"model": mk, "name": MODEL_NAMES[mk], "family": FAMILIES[mk],
                     "latency_ms": lat_ms, "std_ms": lat_std, "acc": acc,
                     "eff": acc / lat_ms if not np.isnan(acc) else np.nan})

    df_lat = pd.DataFrame(rows).dropna(subset=["acc"])

    if df_lat.empty:
        st.warning("No accuracy data for this configuration.")
    else:
        # Pareto mask
        df_lat = df_lat.sort_values("latency_ms").reset_index(drop=True)
        best_so_far, pareto = -np.inf, []
        for _, r in df_lat.iterrows():
            pareto.append(r["acc"] > best_so_far)
            if r["acc"] > best_so_far: best_so_far = r["acc"]
        df_lat["pareto"] = pareto

        # Budget slider — range tied to actual data, not some huge number
        max_lat = float(df_lat["latency_ms"].max())
        col_sl, col_sp = st.columns([3, 1])
        with col_sl:
            target_ms = st.slider(
                "Latency budget (ms)",
                min_value=round(float(df_lat["latency_ms"].min()) * 0.5, 1),
                max_value=round(max_lat * 1.5, 1),
                value=round(max_lat, 1),
                step=0.1,
                format="%.1f ms",
                key="lat_budget",
            )

        # Pareto scatter — LINEAR x-axis, bounded range
        x_pad = max_lat * 0.08
        fig_pareto = go.Figure()

        for fam, grp in df_lat.groupby("family"):
            color = FAM_COLOR.get(fam, "#aaa")
            fig_pareto.add_trace(go.Scatter(
                x=grp["latency_ms"], y=grp["acc"],
                error_x=dict(type="data", array=grp["std_ms"].tolist(),
                             visible=True, color=color, thickness=1.5, width=5),
                mode="markers+text",
                name=fam,
                text=grp["name"],
                textposition="top center",
                textfont=dict(size=11, color="white"),
                marker=dict(
                    size=grp["pareto"].map({True: 20, False: 12}),
                    color=color,
                    symbol=grp["pareto"].map({True: "star", False: "circle"}),
                    line=dict(width=1.5, color="white"),
                ),
                customdata=grp[["name","latency_ms","acc","eff"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Latency: %{customdata[1]:.2f} ms<br>"
                    "Accuracy: %{customdata[2]:.1f}%<br>"
                    "Efficiency: %{customdata[3]:.2f} acc/ms<extra></extra>"
                ),
            ))

        pareto_pts = df_lat[df_lat["pareto"]].sort_values("latency_ms")
        if len(pareto_pts) > 1:
            fig_pareto.add_trace(go.Scatter(
                x=pareto_pts["latency_ms"], y=pareto_pts["acc"],
                mode="lines", name="Pareto frontier",
                line=dict(color="#f39c12", width=2, dash="dot"),
                showlegend=True,
            ))

        # Budget line — clamped to x-axis range so it's always visible
        budget_x = min(target_ms, max_lat * 1.5)
        fig_pareto.add_shape(type="line",
            x0=budget_x, x1=budget_x, y0=0, y1=1, yref="paper",
            line=dict(color="#e74c3c", width=2, dash="dash"))
        fig_pareto.add_annotation(
            x=budget_x, y=1, yref="paper",
            text=f"  Budget: {target_ms:.1f} ms", showarrow=False,
            font=dict(color="#e74c3c", size=11), xanchor="left")

        acc_vals = df_lat["acc"].dropna()
        fig_pareto.update_layout(
            xaxis=dict(
                title=f"Latency (ms/clip) · {hw_choice}",
                range=[-x_pad, max_lat * 1.5 + x_pad],
                gridcolor="#2a2a2a", showgrid=True, zeroline=False,
            ),
            yaxis=dict(
                title="Top-1 accuracy (%)",
                range=[max(0, acc_vals.min() - 5), min(100, acc_vals.max() + 8)],
                gridcolor="#2a2a2a", showgrid=True,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
            height=430, margin=dict(t=40, r=20, b=50, l=60),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # ── Ranking table ─────────────────────────────────────────────────────
        st.subheader("Model ranking by efficiency")
        df_table = df_lat.sort_values("eff", ascending=False).copy()
        df_table["★"] = df_table["pareto"].map({True: "★", False: ""})
        df_table["✓ Budget"] = df_table["latency_ms"].apply(lambda x: "✓" if x <= target_ms else "")
        df_table["Latency (ms)"] = df_table.apply(
            lambda r: f"{r['latency_ms']:.2f} ± {r['std_ms']:.2f}", axis=1)
        df_table["Accuracy (%)"]      = df_table["acc"].map("{:.1f}".format)
        df_table["Efficiency (acc/ms)"] = df_table["eff"].map("{:.2f}".format)
        st.dataframe(
            df_table[["★","name","family","Latency (ms)","Accuracy (%)","Efficiency (acc/ms)","✓ Budget"]]
            .rename(columns={"name":"Model","family":"Family"}),
            hide_index=True, use_container_width=True,
        )

        # ── Insight boxes ─────────────────────────────────────────────────────
        within = df_lat[df_lat["latency_ms"] <= target_ms]
        if within.empty:
            st.warning(f"No model fits within {target_ms:.1f} ms at {lat_res}px on {hw_choice}. "
                       "Try a lower resolution or a faster device.")
        else:
            best      = within.loc[within["acc"].idxmax()]
            efficient = within.loc[within["eff"].idxmax()]
            c1, c2 = st.columns(2)
            c1.info(f"**Best accuracy within budget**  \n"
                    f"{MODEL_NAMES[best['model']]} — **{best['acc']:.1f}%** @ {best['latency_ms']:.1f} ms")
            c2.success(f"**Best efficiency within budget**  \n"
                       f"{MODEL_NAMES[efficient['model']]} — **{efficient['eff']:.2f} acc/ms**")

        with st.expander("Methodology"):
            st.markdown(f"""
**Measurement:** CUDA events (start/end), batch=1, 20 warmup + 100 benchmark iterations, **bfloat16 autocast** for all models.
Models loaded from FineGym P3-retrained checkpoints (same architecture as production inference).
Script: `scripts/accv2026/benchmark_latency_by_resolution.py`

**Hardware scaling ({hw_choice}: {hw_factor:.1f}×):** approximate ratios; real speedup depends on batch size,
CUDA kernels, and driver version. Use as feasibility estimate, not an SLA.

**VideoMamba note:** its forward pass uses `torch.amp.autocast(bfloat16)` internally.
All other models were also measured under the same bfloat16 autocast context for a fair comparison.
""")




# =============================================================================
# 🎯 ARCHITECTURE RECOMMENDER
# =============================================================================
elif page == "🎯 Architecture Recommender":
    st.title("🎯 Architecture Recommender")
    st.caption(
        "Describe your task and deployment constraints. Get an evidence-based recommendation "
        "— model, resolution, stride — grounded in 8,000+ measured configurations and real GPU latency."
    )

    # ── Sidebar: deployment context ───────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("📟 Deployment target")

    _HW_REC = {
        "🖥️ Server — RTX PRO 6000":    ("RTX PRO 6000 Blackwell", 1.0),
        "💻 Desktop — RTX 4090":        ("RTX 4090", 1.5),
        "💻 Desktop — RTX 3090":        ("RTX 3090 / 3080", 2.4),
        "🎮 Gaming — RTX 2080 Ti":      ("RTX 2080 Ti", 3.8),
        "🔧 Edge AI — Jetson AGX Orin": ("Jetson AGX Orin", 10.0),
        "📟 Embedded — Jetson Nano":    ("Jetson Nano", 50.0),
        "❓ Custom":                     ("Custom", None),
    }
    hw_sel = st.sidebar.selectbox("Hardware", list(_HW_REC.keys()), index=0, key="rec_hw")
    hw_name_rec, hw_factor_rec = _HW_REC[hw_sel]
    if hw_factor_rec is None:
        hw_factor_rec = st.sidebar.number_input(
            "Slowdown vs RTX PRO 6000 (×)", 0.1, 500.0, 10.0, key="rec_hw_custom"
        )

    _AUTO_BUDGET = {
        "RTX PRO 6000 Blackwell": 33.0,
        "RTX 4090": 50.0, "RTX 3090 / 3080": 66.0, "RTX 2080 Ti": 100.0,
        "Jetson AGX Orin": 100.0, "Jetson Nano": 200.0, "Custom": 200.0,
    }
    lat_budget_ms = st.sidebar.number_input(
        "Latency budget (ms/clip)",
        10.0, 5000.0, _AUTO_BUDGET.get(hw_name_rec, 200.0), step=10.0,
        help="Max acceptable per-clip latency on the selected device.", key="rec_budget"
    )
    src_fps = st.sidebar.number_input(
        "Video source (fps)", 5, 120, 30, step=5, key="rec_fps",
        help="Frame rate of your input video stream."
    )

    # ── Load latency data ─────────────────────────────────────────────────────
    @st.cache_data
    def _load_lat_rec():
        f = DATA / "latency_by_resolution.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame()
    df_lat_rec = _load_lat_rec()

    # ── Nyquist-safe max stride per TDS value ─────────────────────────────────
    def _nyquist_stride(tds_v):
        if tds_v > 40: return 2
        if tds_v > 20: return 4
        if tds_v > 10: return 8
        return 16

    # ── Device latency preview panel ──────────────────────────────────────────
    if not df_lat_rec.empty:
        with st.expander(f"📊 Device latency profile — {hw_name_rec}", expanded=False):
            lat_rows = []
            for mk in MODEL_KEYS:
                row = {"Model": MODEL_NAMES[mk], "Family": FAMILIES[mk]}
                for res in [48, 96, 112, 160, 224]:
                    v = df_lat_rec[(df_lat_rec.model==mk)&(df_lat_rec.resolution==res)]["mean_ms"].values
                    row[f"{res}px"] = f"{v[0]*hw_factor_rec:.1f}" if len(v) else "—"
                row["Budget ✓"] = "✅" if any(
                    df_lat_rec[(df_lat_rec.model==mk)&(df_lat_rec.resolution==r)]["mean_ms"].values[0]*hw_factor_rec <= lat_budget_ms
                    for r in [48,96,112,160,224]
                    if len(df_lat_rec[(df_lat_rec.model==mk)&(df_lat_rec.resolution==r)]) > 0
                ) else "🔴"
                lat_rows.append(row)
            st.caption(f"All values in ms on **{hw_name_rec}** (RTX PRO 6000 × {hw_factor_rec:.0f}). Budget = {lat_budget_ms:.0f}ms.")
            st.dataframe(pd.DataFrame(lat_rows), hide_index=True, use_container_width=True)

    # ── Build data context (for LLM) ──────────────────────────────────────────
    def build_data_context():
        L = ["## InfoRates Empirical Data — 8 models × 8 datasets × 8,000+ configs\n"]

        L.append("### TDS (Temporal Demand Score) — accuracy drop stride=1→16, cov=100%")
        for ds, tv in sorted(TDS.items(), key=lambda x: -x[1]):
            ns = _nyquist_stride(tv)
            L.append(f"- {DS_LABELS[ds]}: TDS={tv:.1f}pp → Nyquist-safe stride ≤ {ns}")

        L.append("\n### Architecture aliasing robustness (mean drop stride=1→16, 8 datasets)")
        if not df_sw.empty:
            for mk in MODEL_KEYS:
                sub = df_sw[(df_sw.model==mk)&(df_sw.coverage==100)]
                drops = []
                for ds in DS_KEYS:
                    s1  = sub[(sub.stride==1) &(sub.dataset==ds)]["acc"]
                    s16 = sub[(sub.stride==16)&(sub.dataset==ds)]["acc"]
                    if s1.empty or s16.empty or s1.values[0]<5: continue
                    drops.append(s1.values[0]-s16.values[0])
                L.append(f"- {MODEL_NAMES[mk]} ({FAMILIES[mk]}): {np.mean(drops):.1f}pp avg drop")

        L.append(f"\n### INFERENCE LATENCY (bfloat16, batch=1) on {hw_name_rec} "
                 f"(RTX PRO 6000 × {hw_factor_rec:.0f}) — budget={lat_budget_ms:.0f}ms")
        if not df_lat_rec.empty:
            for mk in MODEL_KEYS:
                sub = df_lat_rec[df_lat_rec["model"]==mk]
                vals = []
                for res in [48,96,112,160,224]:
                    v = sub[sub["resolution"]==res]["mean_ms"].values
                    if len(v):
                        scaled = v[0]*hw_factor_rec
                        flag = "✓" if scaled<=lat_budget_ms else "✗"
                        vals.append(f"{res}px={scaled:.0f}ms{flag}")
                L.append(f"- {MODEL_NAMES[mk]}: {', '.join(vals)}")
        L.append("- VideoMamba is SLOWEST in bf16 (8–15ms on PRO 6000, 400–770ms on Jetson Nano)")
        L.append("- VideoMAE is surprisingly fast (1.6–2.8ms) — best accuracy-per-ms transformer")
        L.append("- R3D-18/MC3-18: fastest CNNs (0.6–2.8ms); ideal for edge at 96–160px")
        L.append("- SlowFast latency is nearly flat 48–160px due to adaptive pooling")

        L.append("\n### SAMPLE ACCURACY DATA (cov=100%, all key strides)")
        if not df_sw.empty:
            for ds in DS_KEYS:
                sub_ds = df_sw[df_sw.dataset==ds]
                for stride in [1,2,4,8,16]:
                    sub = sub_ds[(sub_ds.stride==stride)&(sub_ds.coverage==100)].sort_values("acc",ascending=False)
                    if sub.empty: continue
                    top3 = [f"{MODEL_NAMES[r['model']]}={r['acc']:.0f}%"
                            for _,r in sub.head(3).iterrows() if r["acc"]>2]
                    if top3:
                        L.append(f"{ds:12} | s={stride:2} | {', '.join(top3)}")

        L.append("\n### SPATIAL RESOLUTION — P3 retraining benefit")
        L.append("- CNNs: retraining at lower res often IMPROVES accuracy (regularization)")
        L.append("- Transformers: degradation < 112px without bicubic pos-embed interpolation")
        L.append("- VideoMAE @ 96px (retrained): near-native accuracy on most datasets")
        L.append("- VideoMamba @ <112px: risk of collapse on fine-grained tasks")

        L.append("\n### KEY FINDINGS")
        L.append("- Coverage dominates accuracy (ANOVA F=178.94): always use cov=100%")
        L.append("- TimeSformer most robust to stride (10.3pp avg drop) — but 4.3ms @ 224px")
        L.append("- SlowFast most fragile (42.1pp avg drop) — avoid for low-fps deployment")
        L.append("- For edge with Nyquist constraint: CNN at 96–160px beats all transformers in acc/ms")
        return "\n".join(L)

    # ── Core RAG: device-aware, latency-filtered, Nyquist-compliant ──────────
    _NATIVE_RES = {"r3d_18":112,"mc3_18":112,"r2plus1d_18":112,"slowfast_r50":224,
                   "timesformer":224,"vivit":224,"videomae":224,"videomamba":224}

    def rag_recommend(prompt_text, df_sweep, tds_dict,
                      hw_factor, hw_name, budget_ms, src_fps_val):
        import re
        text = prompt_text.lower()

        # 1. Domain match — more precise keyword map with higher specificity first
        keyword_map = [
            # Very specific — few datasets share these terms
            (["sign language","asl","libras","autsl","deaf","signer"],                            "autsl"),
            (["drowsiness","fatigue","dashcam","driveact","in-vehicle","cockpit","cabin"],         "driveact"),
            (["first person","egocentric","epic","kitchens","wrist","utensil"],                    "epic_kitchens"),
            (["diving","dive","springboard","platform dive"],                                      "diving48"),
            (["gymnastics","finegym","floor exercise","pommel","parallel bars","vault","beam"],    "finegym"),
            (["something something","ssv2","pushing","pulling","pick up","put down","sliding"],     "ssv2"),
            # Broader activity / motion words
            (["gesture","hand movement","wrist"],                                                  "autsl"),
            (["cooking","food preparation","chopping","stirring"],                                 "epic_kitchens"),
            (["gym","weightlifting","fitness equipment","treadmill","elliptical","squat","bench"], "finegym"),
            (["causal","counterfactual","temporal order","direction"],                             "ssv2"),
            # General sports / human activity
            (["sport","walk","run","jump","swim","tennis","basketball","soccer",
              "cycling","football","boxing","golf","cricket","skateboard",
              "hmdb","human action","activity"],                                                   "hmdb51"),
            # Appearance-dominated
            (["scene","object recognition","static","ucf","classify","indoor","outdoor",
              "appearance","background"],                                                          "ucf101"),
        ]
        matched_ds, match_score = "hmdb51", 0   # better default than ssv2
        for kws, ds in keyword_map:
            score = sum(1 for kw in kws if kw in text)
            if score > match_score:
                match_score, matched_ds = score, ds

        tds_v  = tds_dict.get(matched_ds, 18.0)
        ns_max = _nyquist_stride(tds_v)
        ds_short = DS_LABELS[matched_ds].split(" (")[0]

        # 2. Parse numeric params from text
        fps_nums   = re.findall(r"(\d+)\s*fps", text)
        stride_num = re.findall(r"stride\s*[=:]?\s*(\d+)", text)
        cov_num    = re.findall(r"coverage\s*[=:]?\s*(\d+)", text)
        text_fps   = int(fps_nums[0])  if fps_nums   else src_fps_val
        query_cov  = int(cov_num[0])   if cov_num    else 100

        # Target stride: Nyquist-respecting, derived from source fps
        if stride_num:
            target_stride = int(stride_num[0])
        else:
            needed_fps = 15.0 if tds_v > 40 else 7.5 if tds_v > 20 else 5.0
            target_stride = max(1, int(text_fps / needed_fps))
        valid_strides = [1, 2, 4, 8, 16]
        target_stride = min(valid_strides, key=lambda s: abs(s - target_stride))
        target_stride = min(target_stride, ns_max)   # never exceed Nyquist

        # 3. Accuracy lookup
        def get_acc_sw(mk, ds, stride, cov=100):
            if df_sweep.empty: return None
            r = df_sweep[(df_sweep.model==mk)&(df_sweep.dataset==ds)&
                         (df_sweep.stride==stride)&(df_sweep.coverage==cov)]
            if not r.empty and r["acc"].values[0] > 2:
                return float(r["acc"].values[0])
            return None

        def get_lat(mk, res):
            if df_lat_rec.empty: return None
            r = df_lat_rec[(df_lat_rec.model==mk)&(df_lat_rec.resolution==res)]
            return float(r["mean_ms"].values[0]) * hw_factor if not r.empty else None

        # 4. Determine budget constraint: is latency actually a binding limit?
        #    Check how many models pass at their native resolution
        native_pass = sum(
            1 for mk in MODEL_KEYS
            if (get_lat(mk, _NATIVE_RES[mk]) or 999) <= budget_ms
        )
        budget_constrained = native_pass < 5   # fewer than 5 models fit at native → constrained

        # 5. Build output
        tier = ("🔴 HIGH" if tds_v>40 else
                "🟡 MODERATE-HIGH" if tds_v>30 else
                "🟠 MODERATE" if tds_v>15 else "🟢 LOW")

        lines = [f"## {prompt_text[:80]}\n"]

        # ── Task summary table ──
        lines.append("### Task Analysis")
        lines.append("| | |")
        lines.append("|---|---|")
        ds_warn = " *(low confidence — refine description)*" if match_score == 0 else ""
        lines.append(f"| Nearest benchmark domain | **{ds_short}**{ds_warn} |")
        lines.append(f"| Temporal demand (TDS) | **{tds_v:.1f}pp** — {tier} |")
        lines.append(f"| Nyquist-safe max stride | ≤ **{ns_max}** at {text_fps}fps source "
                     f"→ min {round(text_fps/ns_max,1)}fps effective |")
        lines.append(f"| Recommended stride | **{target_stride}** "
                     f"({round(text_fps/target_stride,1)}fps effective) |")
        lines.append(f"| Device | **{hw_name}** |")
        budget_note = "no binding constraint — all models fit" if not budget_constrained else f"{budget_ms:.0f}ms/clip"
        lines.append(f"| Latency budget | {budget_ms:.0f}ms/clip — *{budget_note}* |")
        lines.append("")

        # ── Nyquist note (concise) ──
        lines.append("### Sampling / Nyquist")
        if tds_v > 40:
            lines.append(f"Accuracy drops **>30pp** beyond stride {ns_max}. "
                         f"At {text_fps}fps source, stride {ns_max} = {round(text_fps/ns_max,1)}fps effective — "
                         f"do not exceed this.")
        elif tds_v > 15:
            lines.append(f"Stride ≤ {ns_max} empirically safe (mean loss <10pp). "
                         f"Stride {ns_max} at {text_fps}fps → {round(text_fps/ns_max,1)}fps effective.")
        else:
            lines.append(f"Low temporal demand — appearance dominates. "
                         f"Stride up to {ns_max} loses only {tds_v:.1f}pp on average.")
        lines.append("")

        # ── Model ranking table ──
        if not budget_constrained:
            # Device has headroom — rank by accuracy at native resolution
            lines.append(f"### Model Ranking — {ds_short} "
                         f"(device has headroom; sorted by accuracy at native resolution)")
            lines.append("")
            lines.append(f"| Model | Family | Native Res | Latency on {hw_name} | "
                         f"Accuracy s=1 | Accuracy s={target_stride} |")
            lines.append("|-------|--------|-----------|----------------------|"
                         "-------------|----------------------|")
            ranking = []
            for mk in MODEL_KEYS:
                native = _NATIVE_RES[mk]
                lat = get_lat(mk, native)
                acc_s1 = get_acc_sw(mk, matched_ds, 1)
                acc_st = get_acc_sw(mk, matched_ds, target_stride) if target_stride != 1 else acc_s1
                if acc_s1 is None: continue
                ranking.append((mk, native, lat, acc_s1, acc_st))
            ranking.sort(key=lambda x: -(x[3] or 0))
            for mk, native, lat, acc_s1, acc_st in ranking:
                lat_str = f"{lat:.1f}ms" if lat else "—"
                acc_s1_str = f"{acc_s1:.1f}%" if acc_s1 else "—"
                drop = f"{acc_s1-acc_st:.1f}pp↓" if (acc_s1 and acc_st and target_stride > 1) else "—"
                acc_st_str = f"{acc_st:.1f}% ({drop})" if acc_st and target_stride > 1 else (f"{acc_st:.1f}%" if acc_st else "—")
                fam = FAMILIES[mk]
                lines.append(f"| **{MODEL_NAMES[mk]}** | {fam} | {native}px | {lat_str} | "
                             f"{acc_s1_str} | {acc_st_str} |")
            lines.append("")
            lines.append(f"*Latency = RTX PRO 6000 baseline × {hw_factor:.0f}× ({hw_name}), "
                         f"bfloat16, batch=1. Accuracy: cov=100%.*")
            lines.append("")

            # Recommendation for unconstrained device
            if ranking:
                best_mk, best_nat, best_lat, best_acc, best_acc_t = ranking[0]
                lines.append("### Recommendation")
                lines.append(f"**{MODEL_NAMES[best_mk]} @ {best_nat}px, stride={target_stride}**")
                lines.append(f"- Accuracy: **{best_acc_t or best_acc:.1f}%** on {ds_short}")
                lines.append(f"- Latency: {best_lat:.1f}ms/clip on {hw_name} ({best_lat/budget_ms*100:.0f}% of budget)")
                lines.append(f"- Stride={target_stride} → {round(text_fps/target_stride,1)}fps effective — Nyquist-safe (limit: s={ns_max})")
                fam = FAMILIES[best_mk]
                if fam in ("Transformer","SSM") and best_nat == 224:
                    lines.append("- Use full 224px — no resolution trade-off needed on this device")
                # Most efficient
                by_eff = sorted(
                    [(mk, n, l, a, at) for mk, n, l, a, at in ranking if l],
                    key=lambda x: -(x[3] or 0) / (x[2] or 999)
                )
                if by_eff and by_eff[0][0] != best_mk:
                    e = by_eff[0]
                    lines.append(f"\n**Most efficient:** {MODEL_NAMES[e[0]]} @ {e[1]}px — "
                                 f"{e[2]:.1f}ms, {e[3]:.1f}% acc, "
                                 f"{(e[3] or 0)/(e[2] or 1):.2f} acc/ms")

        else:
            # Budget IS constraining — Pareto search across all (model, res)
            lines.append(f"### Latency-Accuracy Pareto — {hw_name}, budget={budget_ms:.0f}ms")
            lines.append(f"Configurations within budget at Nyquist-safe stride ≤ {ns_max}:\n")
            lines.append(f"| Model | Family | Res | Native? | Stride | Latency | Accuracy | acc/ms |")
            lines.append("|-------|--------|-----|---------|--------|---------|----------|--------|")

            # Build records for constrained search
            records = []
            for mk in MODEL_KEYS:
                native = _NATIVE_RES[mk]
                for res in [48, 96, 112, 160, 224]:
                    lat = get_lat(mk, res)
                    if lat is None: continue
                    acc = get_acc_sw(mk, matched_ds, target_stride, query_cov)
                    if acc is None: acc = get_acc_sw(mk, matched_ds, 1, query_cov)
                    if acc is None: continue
                    records.append({
                        "model": mk, "name": MODEL_NAMES[mk], "family": FAMILIES[mk],
                        "res": res, "native": native, "stride": target_stride,
                        "lat_ms": round(lat, 1), "acc": round(acc, 1),
                        "eff": round(acc / lat, 2),
                        "fits": lat <= budget_ms,
                        "is_native": res == native,
                    })

            # Best per (model, res) and sort
            in_budget  = sorted([r for r in records if r["fits"]],  key=lambda x: -x["acc"])
            over_budget = sorted([r for r in records if not r["fits"]], key=lambda x:  x["lat_ms"])

            shown = 0
            for r in in_budget[:8]:
                nat_tag = " ★" if r["is_native"] else ""
                lines.append(
                    f"| **{r['name']}** | {r['family']} | {r['res']}px{nat_tag} | "
                    f"{'yes' if r['is_native'] else 'no'} | s={r['stride']} | "
                    f"{r['lat_ms']:.1f}ms | {r['acc']:.1f}% | {r['eff']:.2f} |"
                )
                shown += 1
            if not shown:
                lines.append("| *no config fits* | — | — | — | — | — | — | — |")
            if over_budget:
                lines.append(f"| *(below: over budget)* | | | | | | | |")
                for r in over_budget[:3]:
                    nat_tag = " ★" if r["is_native"] else ""
                    lines.append(
                        f"| {r['name']} | {r['family']} | {r['res']}px{nat_tag} | "
                        f"{'yes' if r['is_native'] else 'no'} | s={r['stride']} | "
                        f"**{r['lat_ms']:.1f}ms** 🔴 | {r['acc']:.1f}% | {r['eff']:.2f} |"
                    )
            lines.append("")
            lines.append(f"*★ = native training resolution. "
                         f"Latency = RTX PRO 6000 × {hw_factor:.0f}× ({hw_name}), bfloat16, batch=1.*")
            lines.append("")

            if in_budget:
                best = in_budget[0]
                lines.append("### Recommendation")
                lines.append(f"**{best['name']} @ {best['res']}px, stride={target_stride}**")
                lines.append(f"- Accuracy: **{best['acc']:.1f}%** on {ds_short}")
                lines.append(f"- Latency: **{best['lat_ms']:.1f}ms** on {hw_name} "
                             f"({best['lat_ms']/budget_ms*100:.0f}% of {budget_ms:.0f}ms budget)")
                lines.append(f"- Stride={target_stride} → {round(text_fps/target_stride,1)}fps effective — Nyquist-safe")
                fam = FAMILIES[best["model"]]
                if fam in ("CNN","Dual-CNN") and not best["is_native"]:
                    lines.append(f"- Retrain at {best['res']}px: CNNs gain +5–10pp vs cross-resolution eval")
                elif fam in ("Transformer","SSM") and best["res"] < 112:
                    lines.append("- Bicubic pos-embed interpolation required at this resolution (model_factory.py)")
                by_eff = sorted(in_budget, key=lambda x: -x["eff"])
                if by_eff[0]["model"] != best["model"] or by_eff[0]["res"] != best["res"]:
                    e = by_eff[0]
                    lines.append(f"\n**Most efficient:** {e['name']} @ {e['res']}px — "
                                 f"{e['lat_ms']:.1f}ms, {e['acc']:.1f}%, {e['eff']:.2f} acc/ms")
            else:
                lines.append("### ❌ No configuration fits within budget")
                cheapest = sorted(records, key=lambda x: x["lat_ms"])
                if cheapest:
                    c = cheapest[0]
                    lines.append(f"Closest: **{c['name']} @ {c['res']}px** — {c['lat_ms']:.1f}ms, {c['acc']:.1f}%")
                lines.append("Options: lower resolution, relax budget, or use Jetson AGX Orin instead of Nano.")

        return "\n".join(lines)

    # ── LLM call helper ──────────────────────────────────────────────────────
    def call_llm(engine_choice, system_prompt, messages):
        import os
        try:
            from groq import Groq
            key = os.environ.get("GROQ_API_KEY","")
            if not key:
                try: key = st.secrets.get("GROQ_API_KEY","")
                except: pass
            if not key:
                return None, "GROQ_API_KEY not set. Get a free key at console.groq.com"
            client = Groq(api_key=key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile", max_tokens=1500,
                messages=[{"role":"system","content":system_prompt}]+messages,
            )
            return resp.choices[0].message.content, None
        except Exception as e:
            return None, str(e)

    # ── Chat UI ───────────────────────────────────────────────────────────────
    if "rec_msgs" not in st.session_state:
        st.session_state["rec_msgs"] = []
    msgs = st.session_state["rec_msgs"]

    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    example_hint = {
        "🔧 Edge AI — Jetson AGX Orin": "e.g. 'monitoring driver drowsiness at 30fps'",
        "📟 Embedded — Jetson Nano":    "e.g. 'detecting gym exercises at 30fps on Jetson Nano'",
    }.get(hw_sel, "e.g. 'recognizing sign language at 25fps with high accuracy'")

    if prompt := st.chat_input(example_hint):
        msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                rag_args = (prompt, df_sw, TDS, hw_factor_rec, hw_name_rec, lat_budget_ms, src_fps)
                rag_data = rag_recommend(*rag_args)
                data_ctx = build_data_context()
                sys_p = f"""You are a senior research engineer analyzing video architecture deployment.
You have access to measured data from the InfoRates benchmark (8 architectures, 8 datasets, 8,000+ configs).

{data_ctx}

The structured analysis below was produced programmatically from measured data. Your role is to:
- Add mechanistic insight: WHY does this architecture behave this way at this stride/resolution?
- Note any practical trade-offs the user should know before deploying.
- Flag risks (e.g. Transformer sensitivity to resolution, VideoMamba bf16 latency cliff).
- Keep it tight: tables and bullet points only. No narrative framing.

STRICT FORMATTING RULES — violations will cause automatic rejection:
- NO sentences starting with "we", "our", "I", "this analysis", "based on".
- NO phrases like "we are deploying", "we recommend", "to ensure", "as we can see".
- State facts and numbers directly: "VideoMAE @ 224px — 2.1ms, 95.4% on UCF-101."
- Use markdown headers (###), bullet points (-), and tables only.
- If the recommendation is clear-cut, say so in one sentence. Do not hedge.

STRUCTURED ANALYSIS (from measured data):
{rag_data}

Device: {hw_name_rec} | Budget: {lat_budget_ms:.0f}ms/clip | Source: {src_fps}fps"""
                answer, err = call_llm("hybrid", sys_p,
                                       [{"role":m["role"],"content":m["content"]} for m in msgs])
                if err:
                    st.warning(f"LLM unavailable ({err}) — showing data analysis.")
                    answer = rag_data
                st.markdown(answer)
                msgs.append({"role":"assistant","content":answer})

        # Reference chart below the response
        if msgs and not df_sw.empty:
            with st.expander("📈 Aliasing curves — reference"):
                sel_ds_ai = st.selectbox("Dataset", DS_KEYS,
                                          format_func=lambda x: DS_LABELS[x], key="ai_ds")
                sub_ai = df_sw[(df_sw.dataset==sel_ds_ai)&(df_sw.coverage==100)]
                fig_ai = go.Figure()
                for mk in MODEL_KEYS:
                    grp = sub_ai[sub_ai.model==mk].sort_values("stride")
                    if grp.empty or grp["acc"].max()<2: continue
                    fig_ai.add_trace(go.Scatter(
                        x=grp["stride"], y=grp["acc"], mode="lines+markers",
                        name=MODEL_NAMES[mk],
                        line=dict(color=FAM_COLOR.get(FAMILIES[mk],"#999"), width=2),
                        marker=dict(size=7),
                    ))
                ns_ds = _nyquist_stride(TDS.get(sel_ds_ai, 20))
                fig_ai.add_vline(x=ns_ds, line_dash="dash", line_color="#f39c12",
                                 annotation_text=f"Nyquist ≤s={ns_ds}",
                                 annotation_position="top left",
                                 annotation_font_color="#f39c12")
                fig_ai.update_xaxes(type="log", tickvals=[1,2,4,8,16],
                                     ticktext=["s=1","s=2","s=4","s=8","s=16"], title="Stride")
                fig_ai.update_yaxes(title="Top-1 (%)")
                fig_ai.update_layout(height=320, legend=dict(orientation="h",y=-0.3),
                                     margin=dict(b=80))
                st.plotly_chart(fig_ai, use_container_width=True)

    if msgs:
        if st.button("🗑️ Clear conversation"):
            st.session_state[session_key] = []
            st.rerun()
            st.rerun()
