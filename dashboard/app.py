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
DS_KEYS   = ["autsl","diving48","ssv2","hmdb51","driveact","epic_kitchens","ucf101"]
DS_LABELS = {
    "autsl":"AUTSL (Sign Language)","diving48":"Diving-48 (Fine-grained)",
    "ssv2":"SSv2 (Causal)","hmdb51":"HMDB-51 (Sports)",
    "driveact":"DriveAct (In-vehicle)","epic_kitchens":"EPIC-Kitchens (Egocentric)",
    "ucf101":"UCF-101 (Appearance)",
}


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_sweeps():
    f = DATA / "sweep_summary.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
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


@st.cache_data
def load_p3():
    f = DATA / "p3_results.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    df["model_name"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
    return df


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

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 InfoRates")
st.sidebar.caption("Spatiotemporal Aliasing")

page = st.sidebar.radio("", [
    "🏠 Overview & TDS",
    "📊 Aliasing Curves",
    "🔲 Heatmaps",
    "📐 ANOVA & Variance",
    "🌀 Spectral Analysis",
    "⏱ Clip Duration",
    "🔀 Routing & Efficiency",
    "🖼 Spatial Resolution",
    "🎛 Spatiotemporal Explorer",
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
    st.markdown("8 architectures · 7 datasets · 1,400 eval configs")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Architectures", "8", "CNN + Transformer + SSM")
    c2.metric("Datasets", "7", "4 semantic domains")
    c3.metric("Eval configs", "1,400", "coverage × stride grid")
    n_res = len(df_p3.drop_duplicates(["model","dataset","res"])) if not df_p3.empty else 0
    c4.metric("Spatial configs", f"{n_res}", "cross-resolution evaluation")
    st.divider()

    # TDS bar chart from real data
    st.subheader("Temporal Demand Score (TDS)")
    st.caption("Mean accuracy drop (stride=1→16, coverage=100%) averaged over all architectures, excluding feature-collapsed models. Higher = more temporally demanding.")

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

    # Interactive accuracy explorer — any stride × coverage
    st.subheader("Accuracy Explorer — any stride × coverage")
    if not df_sw.empty:
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            sel_cov = st.select_slider("Coverage", options=[10,25,50,75,100], value=100,
                                        key="ov_cov", help="Fraction of clip observed")
        with c_sel2:
            sel_str = st.select_slider("Stride", options=[1,2,4,8,16], value=1,
                                        key="ov_str", help="Sampling density")

        # Bar chart: accuracy at chosen config per model, grouped by dataset (fixed order)
        sub_ov = df_sw[(df_sw.coverage==sel_cov)&(df_sw.stride==sel_str)]
        if not sub_ov.empty:
            fig_ov = go.Figure()
            ds_short = {k: DS_LABELS[k].split(" (")[0] for k in DS_KEYS}
            ds_short_list = [ds_short[ds] for ds in DS_KEYS]  # Fixed order
            for mk in MODEL_KEYS:
                sub_m = sub_ov[sub_ov.model==mk].copy()
                sub_m["ds_short"] = sub_m["dataset"].map(ds_short)
                if sub_m.empty: continue
                # Reindex to fixed dataset order, fill missing with NaN
                accs_ordered = []
                for ds_name in ds_short_list:
                    val = sub_m[sub_m["ds_short"]==ds_name]["acc"]
                    accs_ordered.append(val.values[0] if not val.empty and val.values[0] > 1 else None)
                fig_ov.add_trace(go.Bar(
                    x=ds_short_list, y=accs_ordered,
                    name=MODEL_NAMES[mk],
                    marker_color=FAM_COLOR.get(FAMILIES[mk],"#999"),
                    hovertemplate=f"<b>{MODEL_NAMES[mk]}</b><br>%{{x}}: %{{y:.1f}}%",
                ))
            fig_ov.update_layout(
                barmode="group", height=380,
                title=f"Top-1 accuracy @ coverage={sel_cov}%, stride={sel_str}",
                yaxis_title="Top-1 (%)", xaxis_title="",
                legend=dict(orientation="h", y=-0.25), margin=dict(b=80),
                xaxis=dict(categoryorder="array", categoryarray=ds_short_list),  # Lock order
            )
            st.plotly_chart(fig_ov, use_container_width=True)

        # Compact summary table for current config
        st.subheader(f"Summary table — coverage={sel_cov}%, stride={sel_str}")
        rows = []
        for mk in MODEL_KEYS:
            row = {"Model": MODEL_NAMES[mk], "Family": FAMILIES[mk]}
            accs = []
            for ds in DS_KEYS:
                val = df_sw[(df_sw.model==mk)&(df_sw.dataset==ds)&
                            (df_sw.coverage==sel_cov)&(df_sw.stride==sel_str)]["acc"]
                if val.empty: row[ds_short[ds]] = "—"; continue
                v = val.values[0]
                row[ds_short[ds]] = f"—†" if v < 2 else f"{v:.1f}%"
                if v > 2: accs.append(v)
            row["Avg"] = f"{np.mean(accs):.1f}%" if accs else "—"
            rows.append(row)
        df_tbl = pd.DataFrame(rows)
        st.dataframe(df_tbl, use_container_width=True)
        # Warn about degenerate extreme configs
        if sel_cov == 10 and sel_str == 16:
            st.warning(
                "**Extreme config:** coverage=10% + stride=16 → frame pool has only ~1–3 frames, "
                "so the model receives the same frame repeated up to 8×. "
                "This measures **single-frame accuracy**, not temporal reasoning. "
                "High values on UCF-101 confirm appearance dominance; near-chance on SSv2/AUTSL confirms temporal dependency."
            )
        st.caption("†Feature-collapsed (VideoMamba/AUTSL). Change sliders to explore any of the 25 grid configurations.")


# =============================================================================
# 📊 ALIASING CURVES
# =============================================================================
elif page == "📊 Aliasing Curves":
    st.title("Aliasing Curves — Accuracy vs. Stride")

    st.sidebar.subheader("Settings")
    sel_mkeys = model_select("curves")
    sel_ds = st.sidebar.multiselect("Datasets", DS_KEYS,
                                     default=["autsl","ssv2","ucf101"],
                                     format_func=lambda x: DS_LABELS[x])
    cov = st.sidebar.select_slider("Coverage", [10,25,50,75,100], value=100)
    facet = st.sidebar.radio("Facet by", ["Dataset", "Model"])

    if df_sw.empty:
        st.error("No sweep data.")
        st.stop()

    sub = df_sw[df_sw.model.isin(sel_mkeys) & df_sw.dataset.isin(sel_ds) & (df_sw.coverage == cov)]

    if facet == "Dataset":
        ncols = min(3, len(sel_ds))
        nrows = -(-len(sel_ds) // ncols)
        fig = go.Figure()
        for ds in sel_ds:
            for mk in sel_mkeys:
                grp = sub[(sub.dataset==ds)&(sub.model==mk)].sort_values("stride")
                if grp.empty: continue
                color = FAM_COLOR.get(FAMILIES.get(mk,"CNN"), "#999")
                dash = "solid" if FAMILIES.get(mk) in ("Transformer","SSM") else "dash"
                fig.add_trace(go.Scatter(
                    x=grp["stride"], y=grp["acc"],
                    mode="lines+markers",
                    name=f"{MODEL_NAMES[mk]}",
                    legendgroup=MODEL_NAMES[mk],
                    showlegend=(ds == sel_ds[0]),
                    line=dict(color=color, dash=dash, width=2),
                    marker=dict(size=7),
                    hovertemplate=f"<b>{DS_LABELS[ds]}</b><br>{MODEL_NAMES[mk]}<br>stride=%{{x}}, acc=%{{y:.1f}}%",
                ))
        fig.update_xaxes(type="log", tickvals=[1,2,4,8,16], ticktext=["s=1","s=2","s=4","s=8","s=16"],
                         title="Stride")
        fig.update_yaxes(title="Top-1 Accuracy (%)")
        fig.update_layout(title=f"Coverage={cov}% — all selected datasets",
                          legend=dict(orientation="h",y=-0.25), height=500, margin=dict(b=100))
        st.plotly_chart(fig, use_container_width=True)
    else:
        for mk in sel_mkeys:
            with st.expander(f"{MODEL_NAMES[mk]} ({FAMILIES[mk]})", expanded=True):
                fig = go.Figure()
                for ds in sel_ds:
                    grp = sub[(sub.dataset==ds)&(sub.model==mk)].sort_values("stride")
                    if grp.empty: continue
                    fig.add_trace(go.Scatter(
                        x=grp["stride"], y=grp["acc"],
                        mode="lines+markers", name=DS_LABELS[ds].split(" (")[0],
                        marker=dict(size=7), line=dict(width=2),
                        hovertemplate="stride=%{x}, acc=%{y:.1f}%",
                    ))
                fig.update_xaxes(type="log", tickvals=[1,2,4,8,16], ticktext=["1","2","4","8","16"])
                fig.update_yaxes(title="Top-1 (%)")
                fig.update_layout(height=300, margin=dict(t=20,b=60),
                                  legend=dict(orientation="h",y=-0.35))
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 🔲 HEATMAPS
# =============================================================================
elif page == "🔲 Heatmaps":
    st.title("Coverage × Stride Heatmaps")
    st.sidebar.subheader("Settings")
    sel_model = st.sidebar.selectbox("Model", MODEL_KEYS, format_func=lambda x: MODEL_NAMES[x])
    show_all_ds = st.sidebar.checkbox("All datasets in one view", value=True)

    if df_sw.empty:
        st.error("No sweep data.")
        st.stop()

    def make_heatmap(sub, title, height=300, show_scale=False):
        """Build a proper 5×5 coverage×stride heatmap with categorical axes."""
        pivot = sub.pivot(index="coverage", columns="stride", values="acc")
        # Sort axes correctly
        pivot = pivot.sort_index(ascending=False)          # coverage: 100 top, 10 bottom
        pivot = pivot[sorted(pivot.columns)]               # stride: 1 → 16 left to right
        # Convert to string labels so Plotly treats as equal-width categories
        z    = pivot.values
        y    = [f"{c}%" for c in pivot.index]
        x    = [f"s={s}" for s in pivot.columns]
        text = [[f"{v:.0f}" for v in row] for row in z]
        fig  = go.Figure(go.Heatmap(
            z=z, x=x, y=y,
            colorscale="RdYlGn", zmin=0, zmax=100,
            text=text, texttemplate="%{text}",
            textfont=dict(size=9),
            showscale=show_scale,
            hovertemplate="Coverage=%{y}, %{x}<br>Acc=%{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12)),
            height=height,
            margin=dict(t=40, b=10, l=50, r=10),
            xaxis=dict(side="bottom"),
        )
        return fig

    if show_all_ds:
        cols = st.columns(3)
        for i, ds in enumerate(DS_KEYS):
            sub = df_sw[(df_sw.model==sel_model) & (df_sw.dataset==ds)]
            if sub.empty: continue
            fig = make_heatmap(sub, DS_LABELS[ds].split(" (")[0], height=260)
            cols[i % 3].plotly_chart(fig, use_container_width=True)
    else:
        sel_ds = st.sidebar.selectbox("Dataset", DS_KEYS, format_func=lambda x: DS_LABELS[x])
        sub = df_sw[(df_sw.model==sel_model) & (df_sw.dataset==sel_ds)]
        if not sub.empty:
            fig = make_heatmap(sub,
                               f"{MODEL_NAMES[sel_model]} on {DS_LABELS[sel_ds]}",
                               height=400, show_scale=True)
            st.plotly_chart(fig, use_container_width=True)

    # Side-by-side comparison: TimeSformer vs ViViT (key finding)
    st.divider()
    st.subheader("Key Comparison: TimeSformer (divided attn) vs ViViT (factorized attn)")
    sel_ds_cmp = st.selectbox("Dataset", DS_KEYS, format_func=lambda x: DS_LABELS[x],
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
# 📐 ANOVA & VARIANCE
# =============================================================================
elif page == "📐 ANOVA & Variance":
    st.title("Statistical Analysis — ANOVA & Levene")

    if df_anova.empty:
        st.error("ANOVA data not found.")
        st.stop()

    tab1, tab2 = st.tabs(["ANOVA Effect Sizes", "Stride Effect Summary"])

    with tab1:
        metric = st.radio("Effect", ["Stride (η²_stride)", "Coverage (η²_cov)"], horizontal=True)
        col_key = "eta2_stride" if "Stride" in metric else "eta2_coverage"
        scale_label = "η² stride" if "Stride" in metric else "η² coverage"

        pivot = df_anova.pivot_table(index="model_name", columns="dataset", values=col_key)
        col_rename = {k: DS_LABELS[k].split(" (")[0] for k in DS_KEYS if k in pivot.columns}
        pivot = pivot.rename(columns=col_rename)
        # Sort models by stride η² (most robust first)
        mean_stride = df_anova.groupby("model_name")["eta2_stride"].mean()
        order = mean_stride.sort_values().index.tolist()
        pivot = pivot.reindex([m for m in order if m in pivot.index])

        fig = px.imshow(pivot,
                        color_continuous_scale="Blues" if "Cov" in metric else "Reds",
                        zmin=0, zmax=0.5 if "Stride" in metric else 1.0,
                        text_auto=".2f",
                        labels=dict(color=scale_label),
                        title=f"Two-way ANOVA: {scale_label} per Model × Dataset")
        fig.update_layout(height=420, margin=dict(t=60))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("All entries significant at p<0.05. Sorted by mean stride η² (most robust = top).")

    with tab2:
        grp = df_anova.groupby("model_name").agg(
            stride_mean=("eta2_stride","mean"),
            stride_std=("eta2_stride","std"),
            cov_mean=("eta2_coverage","mean"),
        ).round(3).reset_index()
        grp["family"] = grp["model_name"].map({v:k for k,v in MODEL_NAMES.items()}).map(FAMILIES).fillna("")
        grp = grp.sort_values("stride_mean")

        fig = go.Figure()
        for _, row in grp.iterrows():
            color = FAM_COLOR.get(row["family"], "#999")
            fig.add_trace(go.Bar(
                x=[row["model_name"]],
                y=[row["cov_mean"]],
                name="Coverage η²" if _ == 0 else "",
                marker_color="#3498db", legendgroup="cov",
                showlegend=(_ == 0),
            ))
            fig.add_trace(go.Bar(
                x=[row["model_name"]],
                y=[row["stride_mean"]],
                name="Stride η²" if _ == 0 else "",
                marker_color=color, legendgroup="stride",
                showlegend=(_ == 0),
                error_y=dict(type="data", array=[row["stride_std"]], visible=True),
            ))
        fig.update_layout(barmode="stack", height=380,
                          yaxis_title="η² (mean across 7 datasets)",
                          xaxis_title="Model",
                          title="Coverage vs Stride Effect Sizes (coverage dominates everywhere)",
                          margin=dict(t=50,b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grp.rename(columns={
            "model_name":"Model","stride_mean":"η²_stride","stride_std":"±std",
            "cov_mean":"η²_coverage","family":"Family"}),
            use_container_width=True)


# =============================================================================
# 🌀 SPECTRAL ANALYSIS (E3)
# =============================================================================
elif page == "🌀 Spectral Analysis":
    st.title("Spectral Correlation Analysis (E3)")
    st.caption("Pearson correlation between per-class optical-flow magnitude (Farnebäck) and aliasing loss per dataset.")

    if df_e3.empty:
        st.error("E3 spectral data not found.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pearson r: Flow magnitude ↔ Aliasing loss")
        df_e3_sorted = df_e3.sort_values("pearson_r_abs", ascending=True)
        colors = ["#2ecc71" if s else "#e74c3c" for s in df_e3_sorted["significant"]]
        fig = go.Figure(go.Bar(
            x=df_e3_sorted["pearson_r_abs"],
            y=df_e3_sorted["ds_label"].apply(lambda x: x.split(" (")[0] if isinstance(x, str) else x),
            orientation="h",
            marker_color=colors,
            text=[f"r={r:.3f} (p={p:.3f})" for r,p in zip(df_e3_sorted["pearson_r_abs"], df_e3_sorted["pearson_p_abs"])],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_color="black")
        fig.update_xaxes(title="Pearson r", range=[-0.05, 0.45])
        fig.update_layout(height=350, margin=dict(t=20,r=60,b=40),
                          title="Green = significant (p<0.05), Red = not significant")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Optical flow by taxonomy tier")
        st.caption("Mean flow magnitude per aliasing-sensitivity tier. High-sensitivity classes should have higher flow.")
        fig2 = go.Figure()
        for tier, col_key, color in [
            ("High sensitivity", "flow_high_tier", "#e74c3c"),
            ("Moderate",         "flow_mod_tier",  "#e67e22"),
            ("Low sensitivity",  "flow_low_tier",  "#2ecc71"),
        ]:
            fig2.add_trace(go.Bar(
                x=df_e3["dataset"].apply(lambda x: DS_LABELS.get(x, x).split(" (")[0]),
                y=df_e3[col_key],
                name=tier, marker_color=color,
            ))
        fig2.update_layout(barmode="group", height=350,
                           yaxis_title="Mean flow magnitude (px/frame)",
                           margin=dict(t=20,b=60))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Interpretation")
    st.info(
        "**Moderate correlation overall (r=0.03–0.33):** Optical-flow magnitude alone is an incomplete proxy for "
        "temporal demand. AUTSL (r=0.28, p<0.001) and SSv2 (r=0.18, p=0.015) show significant correlation—flow "
        "captures hand speed in sign language and object velocity in causal actions. "
        "EPIC-Kitchens shows near-zero correlation (r=−0.002): background camera motion dominates flow, "
        "decoupling it from action content. UCF-101 (r=0.03): actions are appearance-dominated, "
        "flow does not predict aliasing sensitivity."
    )
    st.dataframe(df_e3[["dataset","n_classes","pearson_r_abs","pearson_p_abs","significant"]].rename(columns={
        "dataset":"Dataset","n_classes":"Classes",
        "pearson_r_abs":"Pearson r","pearson_p_abs":"p-value","significant":"Significant"
    }), use_container_width=True)


# =============================================================================
# ⏱ CLIP DURATION (E10)
# =============================================================================
elif page == "⏱ Clip Duration":
    st.title("Clip Duration vs. Aliasing Loss (E10)")
    st.caption("Counter-intuitive finding: shorter clips alias *more* — less temporal redundancy means each dropped frame costs more.")

    if df_e10.empty:
        st.error("E10 duration data not found.")
        st.stop()

    st.sidebar.subheader("Settings")
    sel_ds_dur = st.sidebar.selectbox("Dataset", sorted(df_e10["dataset"].unique()),
                                      format_func=lambda x: DS_LABELS.get(x, x))
    sel_models_dur = st.sidebar.multiselect("Models",
                                            sorted(df_e10["model_name"].dropna().unique()),
                                            default=sorted(df_e10["model_name"].dropna().unique()))

    sub = df_e10[(df_e10.dataset == sel_ds_dur) & (df_e10.model_name.isin(sel_models_dur))]
    if sub.empty:
        st.warning("No data for this selection.")
        st.stop()

    # Duration order
    dur_order = ["<1s", "1-3s", "3-6s", ">6s"]
    sub["dur_order"] = pd.Categorical(sub["duration_bin"], categories=dur_order, ordered=True)
    sub = sub.sort_values("dur_order")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Aliasing loss by clip duration — {DS_LABELS.get(sel_ds_dur, sel_ds_dur)}")
        fig = px.line(sub, x="duration_bin", y="aliasing_loss_pp",
                      color="model_name",
                      color_discrete_map={v: FAM_COLOR.get(FAMILIES.get(k,"CNN"),"#999")
                                          for k,v in MODEL_NAMES.items()},
                      markers=True, height=380,
                      labels={"duration_bin":"Clip Duration","aliasing_loss_pp":"Aliasing loss (pp)",
                              "model_name":"Model"},
                      category_orders={"duration_bin": dur_order})
        fig.update_layout(legend=dict(orientation="h",y=-0.3), margin=dict(b=80))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Dense vs Sparse accuracy by duration")
        sub_melt = sub.melt(id_vars=["duration_bin","model_name","n"],
                            value_vars=["acc_dense","acc_sparse"],
                            var_name="config", value_name="accuracy")
        sub_melt["config"] = sub_melt["config"].map({"acc_dense":"Dense (s=1)","acc_sparse":"Sparse (s=16)"})
        sub_melt["accuracy"] *= 100
        fig2 = px.line(sub_melt[sub_melt.model_name == sel_models_dur[0]],
                       x="duration_bin", y="accuracy",
                       color="config", markers=True, height=380,
                       labels={"duration_bin":"Clip Duration","accuracy":"Top-1 (%)","config":""},
                       title=f"{sel_models_dur[0] if sel_models_dur else ''}",
                       category_orders={"duration_bin": dur_order},
                       color_discrete_map={"Dense (s=1)":"#2ecc71","Sparse (s=16)":"#e74c3c"})
        fig2.update_layout(legend=dict(orientation="h",y=-0.3), margin=dict(b=80))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Summary table")
    tbl = sub[["model_name","duration_bin","n","acc_dense","acc_sparse","aliasing_loss_pp"]].copy()
    tbl["acc_dense"]   = (tbl["acc_dense"]*100).round(1)
    tbl["acc_sparse"]  = (tbl["acc_sparse"]*100).round(1)
    tbl["aliasing_loss_pp"] = tbl["aliasing_loss_pp"].round(1)
    tbl = tbl.rename(columns={"model_name":"Model","duration_bin":"Duration","n":"N clips",
                               "acc_dense":"Dense acc (%)","acc_sparse":"Sparse acc (%)",
                               "aliasing_loss_pp":"Aliasing loss (pp)"})
    st.dataframe(tbl.sort_values(["Model","Duration"]).reset_index(drop=True), use_container_width=True)


# =============================================================================
# 🔀 ROUTING & EFFICIENCY (E7 + E9)
# =============================================================================
elif page == "🔀 Routing & Efficiency":
    st.title("Entropy Routing & Efficiency Analysis (E7 / E9)")

    tab_curves, tab_compare, tab_summary = st.tabs([
        "Routing Curves", "vs Literature Baselines (E9)", "Full Results Table"
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
    st.title("Spatial Aliasing — Cross-Resolution Evaluation")
    st.markdown(
        "Each model is evaluated at 5 resolutions (96–336px) **without retraining**. "
        "The same attention-type split that governs temporal robustness also governs spatial robustness."
    )

    if df_p3.empty:
        st.warning("Spatial resolution results not found.")
        st.stop()

    # Sidebar filters
    st.sidebar.subheader("Filters")
    sel_ds_p3 = st.sidebar.selectbox(
        "Dataset", sorted(df_p3["dataset"].unique()),
        format_func=lambda x: DS_LABELS.get(x, x), key="p3_ds"
    )
    sel_mods_p3 = st.sidebar.multiselect(
        "Models", sorted(df_p3["model_name"].dropna().unique()),
        default=sorted(df_p3["model_name"].dropna().unique()), key="p3_mods"
    )

    sub = df_p3[(df_p3.dataset == sel_ds_p3) & df_p3.model_name.isin(sel_mods_p3)]

    # Native accuracy = sweep at stride=1, coverage=100% (each model's own training resolution)
    native_acc = {}
    if not df_sw.empty:
        e1_sub = df_sw[(df_sw.dataset == sel_ds_p3) & (df_sw.coverage == 100) & (df_sw.stride == 1)]
        for mk in MODEL_KEYS:
            row = e1_sub[e1_sub.model == mk]
            if not row.empty:
                native_acc[mk] = row["acc"].values[0]
    # Native resolution per model (training resolution)
    NATIVE_RES = {
        "r3d_18": 112, "mc3_18": 112, "r2plus1d_18": 112, "slowfast_r50": 224,
        "timesformer": 224, "vivit": 224, "videomae": 224, "videomamba": 224,
    }

    # ── Main chart: accuracy vs resolution ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy vs. resolution")
        fig = go.Figure()
        for mdl_name, grp in sub.groupby("model_name"):
            mk = {v: k for k, v in MODEL_NAMES.items()}.get(mdl_name)
            fam = FAMILIES.get(mk, "CNN")
            color = FAM_COLOR.get(fam, "#999")
            dash = "solid" if fam in ("Transformer", "SSM") else "dot"
            grp = grp.sort_values("res")
            fig.add_trace(go.Scatter(
                x=grp["res"], y=grp["acc"],
                mode="lines+markers", name=mdl_name,
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=9),
                hovertemplate=f"<b>{mdl_name}</b><br>%{{x}}px → %{{y:.1f}}%<extra></extra>",
            ))
            # Native accuracy reference (horizontal dashed line)
            if mk in native_acc:
                nat = native_acc[mk]
                native_res = NATIVE_RES.get(mk, 224)
                res_range = [grp["res"].min(), grp["res"].max()]
                fig.add_trace(go.Scatter(
                    x=res_range, y=[nat, nat],
                    mode="lines", name=f"{mdl_name} native ({native_res}px)",
                    line=dict(color=color, dash="dash", width=1),
                    showlegend=False,
                ))
                # Star marker at native resolution
                fig.add_trace(go.Scatter(
                    x=[native_res], y=[nat],
                    mode="markers", name=f"native",
                    marker=dict(symbol="star", size=14, color=color),
                    showlegend=False,
                    hovertemplate=f"<b>{mdl_name} native</b><br>{native_res}px → {nat:.1f}%<extra></extra>",
                ))
        fig.update_xaxes(title="Evaluation resolution (px)", tickvals=[96, 112, 160, 224, 336])
        fig.update_yaxes(title="Top-1 (%)")
        fig.update_layout(
            height=420,
            legend=dict(orientation="h", y=-0.35),
            margin=dict(b=90),
            title=DS_LABELS.get(sel_ds_p3, sel_ds_p3),
            annotations=[dict(
                x=0.5, y=1.04, xref="paper", yref="paper",
                text="Dashed = CNN  |  Solid = Transformer/SSM  |  Native ref = horizontal dash",
                showarrow=False, font=dict(size=10, color="gray")
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Delta table ──
    with col2:
        st.subheader("Δ vs. native resolution")
        rows_tbl = []
        for _, r in sub.iterrows():
            mk = {v: k for k, v in MODEL_NAMES.items()}.get(r["model_name"])
            nat = native_acc.get(mk)
            delta = r["acc"] - nat if nat is not None else None
            rows_tbl.append({
                "Model": r["model_name"],
                "Res": f"{r['res']}px",
                "Acc": f"{r['acc']:.1f}%",
                "Native": f"{nat:.1f}%" if nat is not None else "—",
                "Δ": f"{delta:+.1f}pp" if delta is not None else "—",
            })
        if rows_tbl:
            tbl = pd.DataFrame(rows_tbl).sort_values(["Model", "Res"])
            st.dataframe(tbl.reset_index(drop=True), use_container_width=True)

    st.divider()

    # ── Architecture split summary ──
    st.subheader("CNN vs. Transformer/SSM split across all datasets")
    st.caption("Mean accuracy at each resolution, grouped by architecture family.")

    if not df_p3.empty:
        df_p3_fam = df_p3.copy()
        df_p3_fam["family"] = df_p3_fam["model"].map(
            lambda m: FAMILIES.get(m, "CNN")
        ).map(lambda f: "CNN" if f in ("CNN", "Dual-CNN") else "Transformer / SSM")

        fam_summary = (
            df_p3_fam.groupby(["family", "res"])["acc"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        fig2 = go.Figure()
        colors = {"CNN": "#e74c3c", "Transformer / SSM": "#2980b9"}
        for fam, grp in fam_summary.groupby("family"):
            grp = grp.sort_values("res")
            fig2.add_trace(go.Scatter(
                x=grp["res"], y=grp["mean"],
                error_y=dict(type="data", array=grp["std"].tolist(), visible=True),
                mode="lines+markers", name=fam,
                line=dict(color=colors.get(fam, "#999"), width=3),
                marker=dict(size=10),
                hovertemplate=f"<b>{fam}</b><br>%{{x}}px: %{{y:.1f}}% ± %{{error_y.array:.1f}}<extra></extra>",
            ))
        fig2.update_xaxes(title="Evaluation resolution (px)", tickvals=[96, 112, 160, 224, 336])
        fig2.update_yaxes(title="Mean Top-1 (%)")
        fig2.update_layout(
            height=380,
            legend=dict(orientation="h", y=-0.2),
            title="All datasets · All models (mean ± std)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Key stats
        cnn_96 = fam_summary[(fam_summary.family == "CNN") & (fam_summary.res == 96)]["mean"]
        cnn_224 = fam_summary[(fam_summary.family == "CNN") & (fam_summary.res == 224)]["mean"]
        tr_96 = fam_summary[(fam_summary.family == "Transformer / SSM") & (fam_summary.res == 96)]["mean"]
        tr_224 = fam_summary[(fam_summary.family == "Transformer / SSM") & (fam_summary.res == 224)]["mean"]

        mc1, mc2, mc3 = st.columns(3)
        if not cnn_96.empty and not cnn_224.empty:
            mc1.metric("CNN: 96px vs native", f"{cnn_96.values[0]:.1f}% vs {cnn_224.values[0]:.1f}%",
                       f"{cnn_96.values[0]-cnn_224.values[0]:+.1f}pp")
        if not tr_96.empty and not tr_224.empty:
            mc2.metric("Transformer/SSM: 96px vs native", f"{tr_96.values[0]:.1f}% vs {tr_224.values[0]:.1f}%",
                       f"{tr_96.values[0]-tr_224.values[0]:+.1f}pp")
        mc3.metric("Configs evaluated", f"{len(df_p3.drop_duplicates(['model','dataset','res']))}",
                   "8 models × 7 datasets × 5 res")


# =============================================================================
# 🎛 SPATIOTEMPORAL EXPLORER
# =============================================================================
elif page == "🎛 Spatiotemporal Explorer":
    st.title("Spatiotemporal Explorer")
    st.markdown("Escolha **dataset, stride, coverage e resolução** e veja o accuracy de cada modelo naquele ponto exato.")

    # ── Carregar dados combinados (coverage × stride × resolução) ─────────────
    @st.cache_data
    def load_combined_sweep():
        sweep_root = Path(__file__).parent.parent / "evaluations/accv2026/coverage_stride_sweep"
        if not sweep_root.exists():
            return pd.DataFrame()
        rows = []
        ds_list = ["ucf101","ssv2","hmdb51","diving48","autsl","driveact","epic_kitchens"]
        NATIVE = {"r3d_18":112,"mc3_18":112,"r2plus1d_18":112,"slowfast_r50":224,
                  "timesformer":224,"vivit":224,"videomae":224,"videomamba":224}
        for csv in sweep_root.glob("*/sweep_summary.csv"):
            folder = csv.parent.name
            parts = folder.split("_res")
            tag = parts[0]
            res_override = int(parts[1]) if len(parts) > 1 else None
            ds = next((d for d in sorted(ds_list, key=len, reverse=True) if tag.endswith(d)), None)
            if not ds:
                continue
            model = tag[:-(len(ds)+1)]
            try:
                df = pd.read_csv(csv)
                df["model"]   = model
                df["dataset"] = ds
                df["res"]     = res_override if res_override else NATIVE.get(model, 224)
                df["acc"]     = df["top1"] * 100
                rows.append(df[["model","dataset","res","coverage","stride","acc"]])
            except Exception:
                continue
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    df_comb = load_combined_sweep()

    # Dados que temos:
    # df_sw  = temporal sweep  → (model, dataset, coverage, stride) @ native res
    # df_p3  = spatial eval    → (model, dataset, res) @ stride=1, cov=100%
    # df_comb = combined sweep → (model, dataset, res, coverage, stride)  ← em geração

    NATIVE_RES = {"r3d_18":112,"mc3_18":112,"r2plus1d_18":112,"slowfast_r50":224,
                  "timesformer":224,"vivit":224,"videomae":224,"videomamba":224}
    ALL_RES      = [96, 112, 160, 224, 336]
    ALL_STRIDES  = [1, 2, 4, 8, 16]
    ALL_COVERAGES= [10, 25, 50, 75, 100]

    # ── 3 sliders no topo da página ───────────────────────────────────────────
    st.subheader("Parâmetros")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        sel_ds = st.selectbox("Dataset", DS_KEYS,
                              format_func=lambda k: DS_LABELS.get(k, k), key="st_ds")
    with c2:
        sel_res = st.select_slider("Resolução (px)", ALL_RES, value=112, key="st_res")
    with c3:
        sel_stride = st.select_slider("Stride", ALL_STRIDES, value=1, key="st_stride")
    with c4:
        sel_cov = st.select_slider("Coverage (%)", ALL_COVERAGES, value=100, key="st_cov")

    # ── Lookup de accuracy para cada modelo ───────────────────────────────────
    results = []
    for model in MODEL_KEYS:
        native = NATIVE_RES.get(model, 224)
        acc = None
        source = None

        # 1. Combined sweep (cobertura total)
        if not df_comb.empty:
            row = df_comb[
                (df_comb.model==model) & (df_comb.dataset==sel_ds) &
                (df_comb.res==sel_res) & (df_comb.stride==sel_stride) &
                (df_comb.coverage==sel_cov)
            ]
            if not row.empty:
                acc = row["acc"].values[0]
                source = "combined"

        # 2. Temporal sweep (só para res=native)
        if acc is None and not df_sw.empty and sel_res == native:
            row = df_sw[
                (df_sw.model==model) & (df_sw.dataset==sel_ds) &
                (df_sw.stride==sel_stride) & (df_sw.coverage==sel_cov)
            ]
            if not row.empty:
                acc = row["acc"].values[0]
                source = "temporal"

        # 3. Spatial eval (só para stride=1, cov=100%)
        if acc is None and not df_p3.empty and sel_stride==1 and sel_cov==100:
            row = df_p3[
                (df_p3.model==model) & (df_p3.dataset==sel_ds) & (df_p3.res==sel_res)
            ]
            if not row.empty:
                acc = row["acc"].values[0]
                source = "spatial"

        # Referência: native @ stride=1, cov=100%
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

    # ── Resultado principal: bar chart de todos os modelos ────────────────────
    st.divider()
    title_str = f"Accuracy @ stride={sel_stride} · coverage={sel_cov}% · {sel_res}px — {DS_LABELS.get(sel_ds, sel_ds)}"
    st.subheader(title_str)

    if df_res_valid.empty:
        st.warning(
            f"Sem dados para stride={sel_stride}, coverage={sel_cov}%, resolução={sel_res}px.\n\n"
            "**Dados disponíveis agora:**\n"
            f"- Qualquer stride/coverage + resolução **nativa** de cada modelo (do temporal sweep)\n"
            f"- stride=1, coverage=100% + qualquer resolução para SSv2 (do spatial eval)\n\n"
            "Para habilitar todos os parâmetros, execute o combined sweep:\n"
            "```bash\nbash scripts/accv2026/submit_combined_sweep.sh\n```"
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
                              annotation_text=f"Ref nativa média: {mean_ref:.1f}%",
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
            src_icons = {"temporal": "⏱ temporal", "spatial": "🖼 spatial", "combined": "🎛 combined"}
            tbl_rows.append({
                "Modelo": r["name"],
                "Família": r["family"],
                "Native px": f"{r['native_res']}px",
                f"Acc @{sel_res}px s={sel_stride} c={sel_cov}%": f"{r['acc']:.1f}%",
                "Ref (s=1,c=100%,native)": f"{r['ref']:.1f}%" if r['ref'] else "—",
                "Δ": f"{r['delta']:+.1f}pp" if r['delta'] is not None else "—",
                "Fonte": src_icons.get(r["source"], r["source"]),
            })
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    # ── Heatmap coverage × stride para o modelo selecionado ───────────────────
    st.divider()
    st.subheader("Heatmap Coverage × Stride — por modelo")
    col_hm1, col_hm2 = st.columns([1, 2])
    with col_hm1:
        hm_model = st.selectbox("Modelo", MODEL_KEYS,
                                format_func=lambda k: MODEL_NAMES[k], key="st_hm_model")
        hm_res_opts = [sel_res]
        native_hm = NATIVE_RES.get(hm_model, 224)
        if native_hm not in hm_res_opts:
            hm_res_opts = [native_hm] + hm_res_opts
        hm_res = st.selectbox("Resolução heatmap", hm_res_opts, key="st_hm_res")

    with col_hm2:
        # Montar dados para heatmap
        if not df_comb.empty:
            hm_data = df_comb[
                (df_comb.model==hm_model) & (df_comb.dataset==sel_ds) & (df_comb.res==hm_res)
            ]
            src_label = f"{hm_res}px (combined)"
        else:
            hm_data = pd.DataFrame()
            src_label = ""

        # Fallback: temporal sweep para resolução nativa
        if hm_data.empty and hm_res == NATIVE_RES.get(hm_model, 224) and not df_sw.empty:
            hm_data = df_sw[
                (df_sw.model==hm_model) & (df_sw.dataset==sel_ds)
            ].rename(columns={"acc": "acc"})
            src_label = f"{hm_res}px nativa (temporal sweep)"

        if hm_data.empty:
            st.info(f"Sem dados de heatmap para {MODEL_NAMES[hm_model]} / {sel_ds} @ {hm_res}px.\n\n"
                    "Execute `submit_combined_sweep.sh` para gerar dados multi-resolução.")
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
            "🔸 **Combined sweep ainda não foi executado.** "
            "Só são mostrados pontos existentes no temporal sweep (resolução nativa) "
            "e spatial eval (stride=1, cov=100%). "
            "Para habilitar todos os parâmetros: `bash scripts/accv2026/submit_combined_sweep.sh`"
        )


# =============================================================================
# 🎯 ARCHITECTURE RECOMMENDER
# =============================================================================
elif page == "🎯 Architecture Recommender":
    st.title("🎯 Architecture Recommender")
    st.caption(
        "Describe your activity recognition task in plain English. "
        "Get a recommendation for architecture, frame rate, and observation window "
        "— backed by 1,400 empirical measurement configurations."
    )

    engine = st.sidebar.radio("Engine", [
        "🦙 Groq (Llama-3.3-70B) — free",
        "⚙️ RAG (no API, instant)",
        "🔬 Hybrid (RAG + Groq) — rich analysis",
    ], index=2)
    st.sidebar.caption("Groq: fast. RAG: offline. Hybrid: combines both for depth.")

    # ── Build data context for the AI ────────────────────────────────────────
    def build_data_context():
        lines = ["## InfoRates COMPLETE Empirical Data\n"]

        # TDS
        lines.append("### TDS (Temporal Demand Score) per dataset")
        for ds, tds_v in sorted(TDS.items(), key=lambda x: -x[1]):
            lines.append(f"- {DS_LABELS[ds]}: TDS={tds_v:.1f}pp")

        # Avg drops
        lines.append("\n### Architecture aliasing robustness (avg accuracy drop stride=1→16)")
        if not df_sw.empty:
            for mk in MODEL_KEYS:
                sub = df_sw[(df_sw.model==mk)&(df_sw.coverage==100)]
                drops = []
                for ds in DS_KEYS:
                    s1  = sub[(sub.stride==1)&(sub.dataset==ds)]["acc"]
                    s16 = sub[(sub.stride==16)&(sub.dataset==ds)]["acc"]
                    if s1.empty or s16.empty or s1.values[0]<5: continue
                    drops.append(s1.values[0]-s16.values[0])
                avg = np.mean(drops) if drops else 0
                lines.append(f"- {MODEL_NAMES[mk]} ({FAMILIES[mk]}): {avg:.1f}pp")

        # KEY accuracy configs (not complete table — that's too large)
        lines.append("\n### SAMPLE ACCURACY DATA (stride & coverage combinations)")
        lines.append("Best 3 models per config. Format: dataset | stride | coverage | top models")
        if not df_sw.empty:
            for ds in ["autsl", "ssv2", "ucf101"]:  # Just top 3 datasets
                sub_ds = df_sw[df_sw.dataset==ds]
                for stride in [1, 4, 8, 16]:
                    for cov in [10, 100]:  # Just 10% and 100%
                        sub = sub_ds[(sub_ds.stride==stride)&(sub_ds.coverage==cov)].sort_values("acc", ascending=False)
                        if sub.empty: continue
                        top3 = [f"{MODEL_NAMES[row['model']]}={row['acc']:.0f}%" for _, row in sub.head(3).iterrows() if row['acc']>2]
                        if top3:
                            lines.append(f"{ds:10} | s={stride:2} | c={cov:3}% | {', '.join(top3)}")

        # P3 spatial retraining summary
        lines.append("\n### P3 SPATIAL RETRAINING (CNN resolution ablation)")
        try:
            df_p3 = pd.read_csv(DATA / "p3_results.csv")
            for model in ["r3d_18", "slowfast_r50"]:
                avg_96 = df_p3[(df_p3.model==model)&(df_p3.res==96)]["acc"].mean()
                avg_224 = df_p3[(df_p3.model==model)&(df_p3.res==224)]["acc"].mean()
                lines.append(f"- {MODEL_NAMES[model]}: @96px avg={avg_96:.1f}%, @224px avg={avg_224:.1f}%")
        except: pass

        # Routing summary
        lines.append("\n### E7 ENTROPY ROUTING")
        lines.append("- Adaptive frame allocation: ~77% of videos route to cheap 4-frame inference")
        lines.append("- Saves computation while maintaining accuracy")

        # ANOVA summary
        lines.append("\n### ANOVA KEY INSIGHTS")
        lines.append("- Stride effect: higher on high-TDS datasets (sign language, causal)")
        lines.append("- Coverage effect: larger on temporally demanding tasks")

        lines.append("\n### KEY FINDINGS")
        lines.append("- VideoMamba (SSM) and TimeSformer: ~6-8pp avg drop — most robust to aliasing")
        lines.append("- ViViT (factorized attention): ~34pp drop — anomaly for Transformer")
        lines.append("- SlowFast: ~42pp drop — most fragile")
        lines.append("- CNNs: 18-28pp drop, benefit from lower-res retraining (77% improvement rate)")
        lines.append("- Transformers: sensitive to spatial resolution below 112px (patch token loss)")
        lines.append("- EPIC-Kitchens: exception — even Transformers improve at lower res (egocentric noise)")
        lines.append("- E7 entropy routing: ~77% of videos route to cheap 4-frame inference")

        return "\n".join(lines)

    # ── RAG engine: pure data-driven, no API ─────────────────────────────────
    def rag_recommend(prompt_text, df_sweep, tds_dict):
        """Keyword-based dataset matching + structured recommendation from REAL DATA."""
        text = prompt_text.lower()

        # Map keywords → dataset
        keyword_map = [
            (["sign language","sign","gesture","hand","deaf","asl","libras","autsl"],  "autsl"),
            (["driving","driver","vehicle","car","dashcam","drowsiness","fatigue","driveact"], "driveact"),
            (["kitchen","cooking","food","eat","chef","egocentric","first person","epic"], "epic_kitchens"),
            (["diving","swimming","gymnastics","sport","fine.grained","precise"],        "diving48"),
            (["something","manipulation","push","pull","pick","causal","physics","ssv2"],"ssv2"),
            (["sport","action","human","general","hmdb","exercise","workout"],           "hmdb51"),
            (["appearance","object","scene","recognition","classify","ucf"],             "ucf101"),
        ]
        matched_ds = "ssv2"  # default
        match_score = 0
        for keywords, ds in keyword_map:
            score = sum(1 for kw in keywords if kw in text)
            if score > match_score:
                match_score, matched_ds = score, ds

        tds_v = tds_dict.get(matched_ds, 20)

        # Extract stride, coverage, fps from text
        import re
        stride_match = None
        coverage_match = None
        fps_match = None

        stride_nums = re.findall(r"stride\s*[=:]?\s*(\d+)", text)
        if stride_nums: stride_match = int(stride_nums[0])

        coverage_nums = re.findall(r"coverage\s*[=:]?\s*(\d+)", text)
        if coverage_nums: coverage_match = int(coverage_nums[0])

        fps_nums = re.findall(r"(\d+)\s*fps", text)
        if fps_nums: fps_match = int(fps_nums[0])

        # Default to stride=1, coverage=100 if not specified; use user values if they are
        query_stride = stride_match if stride_match else 1
        query_coverage = coverage_match if coverage_match else 100

        # Get REAL accuracies from data at specified stride/coverage
        real_accs_user_config = {}
        best_s1 = []
        robust = []

        if not df_sweep.empty:
            # At user-specified config (stride/coverage)
            sub_user = df_sweep[(df_sweep.dataset==matched_ds)&
                               (df_sweep.coverage==query_coverage)&
                               (df_sweep.stride==query_stride)]
            if not sub_user.empty:
                for mk in MODEL_KEYS:
                    v = sub_user[sub_user.model==mk]["acc"]
                    if not v.empty and v.values[0] > 2:
                        real_accs_user_config[mk] = v.values[0]
                best_user = sorted(real_accs_user_config.items(), key=lambda x: -x[1])[:3]

            # Also get stride=1 and stride=8 for comparisons
            sub_s1 = df_sweep[(df_sweep.dataset==matched_ds)&(df_sweep.coverage==100)&(df_sweep.stride==1)]
            sub_s8 = df_sweep[(df_sweep.dataset==matched_ds)&(df_sweep.coverage==100)&(df_sweep.stride==8)]

            acc_s1  = {mk: sub_s1[sub_s1.model==mk]["acc"].values[0]
                       for mk in MODEL_KEYS
                       if not sub_s1[sub_s1.model==mk]["acc"].empty
                       and sub_s1[sub_s1.model==mk]["acc"].values[0]>2}
            acc_s8  = {mk: sub_s8[sub_s8.model==mk]["acc"].values[0]
                       for mk in MODEL_KEYS
                       if not sub_s8[sub_s8.model==mk]["acc"].empty
                       and sub_s8[sub_s8.model==mk]["acc"].values[0]>2}

            best_s1 = sorted(acc_s1.items(), key=lambda x: -x[1])[:3] if acc_s1 else []
            drops   = {mk: acc_s1.get(mk,0) - acc_s8.get(mk,0) for mk in acc_s1 if mk in acc_s8}
            robust  = sorted(drops.items(), key=lambda x: x[1])[:2]

        # Determine stride recommendation
        if tds_v > 35:
            rec_stride, rec_fps_note = 2, "≤ 15fps (stride≤2 to stay above Nyquist)"
        elif tds_v > 18:
            rec_stride, rec_fps_note = 4, "8–15fps (stride 4 is borderline safe)"
        else:
            rec_stride, rec_fps_note = 8, "4–8fps sufficient (appearance-dominated)"

        labels_short = {k: v.split(" (")[0] for k,v in {
            "autsl":"AUTSL (Sign Language)","diving48":"Diving-48 (Fine-grained)",
            "ssv2":"SSv2 (Causal)","hmdb51":"HMDB-51 (Sports)",
            "driveact":"DriveAct (In-vehicle)","epic_kitchens":"EPIC-Kitchens (Egocentric)",
            "ucf101":"UCF-101 (Appearance)"}.items()}

        lines = []
        lines.append(f"## Recommendation for: *{prompt_text[:80]}*\n")
        lines.append(f"**Matched domain:** {labels_short.get(matched_ds, matched_ds)} (TDS = {tds_v:.1f}pp)")
        if stride_match or coverage_match:
            lines.append(f"**Your config:** stride={query_stride}, coverage={query_coverage}%")
        if match_score == 0:
            lines.append("*⚠️ No strong keyword match — defaulted to SSv2 (causal actions). Refine your description.*")
        lines.append("")

        tier = "🔴 HIGH" if tds_v>35 else "🟡 MODERATE" if tds_v>18 else "🟢 LOW"
        lines.append(f"### {tier} temporal demand")
        if tds_v > 35:
            lines.append("Dense sampling is **critical**. Sparse frames cause >30pp accuracy loss.")
        elif tds_v > 18:
            lines.append("Moderate aliasing risk. Stride > 8 causes noticeable degradation.")
        else:
            lines.append("Low aliasing risk. Appearance features are sufficient; temporal order matters less.")

        lines.append(f"\n### Recommended sampling: **{rec_fps_note}**")
        lines.append(f"Target stride ≤ {rec_stride} to stay above the Nyquist rate for this domain.")
        if fps_match:
            eff = max(1, fps_match // 8)
            lines.append(f"At your {fps_match}fps source → sample every ~{eff} frame (stride≈{eff}).")

        lines.append("\n### Architecture ranking for YOUR config (stride={}, coverage={}%)".format(query_stride, query_coverage))
        if real_accs_user_config:
            lines.append(f"**MEASURED ACCURACY** from empirical data:")
            for mk, acc in sorted(real_accs_user_config.items(), key=lambda x: -x[1]):
                lines.append(f"- {MODEL_NAMES[mk]} ({FAMILIES[mk]}): **{acc:.1f}%**")
            lines.append("")

        if best_s1:
            lines.append(f"**Best absolute accuracy** (stride=1, coverage=100% — dense baseline):")
            for mk, acc in best_s1:
                lines.append(f"- {MODEL_NAMES[mk]} ({FAMILIES[mk]}): {acc:.1f}%")
        if robust:
            lines.append(f"\n**Most robust to sparse sampling** (smallest accuracy drop stride=1→8):")
            for mk, drop in robust:
                lines.append(f"- {MODEL_NAMES[mk]} ({FAMILIES[mk]}): only {drop:.1f}pp drop")

        lines.append("\n### Spatial resolution")
        lines.append("- **CNNs**: retrain at your deployment resolution for best results (lower res often HELPS)")
        lines.append("- **Transformers/SSMs**: robust across 96–336px without retraining")
        if matched_ds == "autsl":
            lines.append("- ⚠️ AUTSL exception: handshapes need ≥112px — do not downsample below this")

        return "\n".join(lines)

    # ── LLM call helper ──────────────────────────────────────────────────────
    def call_llm(engine_choice, system_prompt, messages):
        import os
        try:
            if "Groq" in engine_choice:
                from groq import Groq
                key = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY",""))
                if not key: return None, "GROQ_API_KEY not set. Get a free key at console.groq.com"
                client = Groq(api_key=key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", max_tokens=1024,
                    messages=[{"role":"system","content":system_prompt}]+messages,
                )
                return resp.choices[0].message.content, None

        except Exception as e:
            return None, str(e)
        return None, "Unknown engine"

    # ── Chat interface ────────────────────────────────────────────────────────
    session_key = f"msgs_{engine.split()[0]}"
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    msgs = st.session_state[session_key]
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Describe your task… e.g. 'monitoring driver drowsiness at 10fps'"):
        msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                if engine.startswith("⚙️"):  # Pure RAG
                    answer = rag_recommend(prompt, df_sw, TDS)
                    st.markdown(answer)
                    msgs.append({"role": "assistant", "content": answer})

                elif engine.startswith("🔬"):  # Hybrid: RAG + Groq
                    # Step 1: Get raw data from RAG
                    rag_data = rag_recommend(prompt, df_sw, TDS)

                    # Step 2: Enrich with Groq interpretation
                    data_ctx = build_data_context()
                    system_prompt = f"""You are an expert AI analyst for InfoRates (spatiotemporal aliasing in video action recognition).

{data_ctx}

A user has asked a recommendation question. Below is a data-driven analysis with MEASURED ACCURACIES.
Your job: enhance this analysis with deeper insights about WHY these results occur, trade-offs, and practical implications.

DATA ANALYSIS FROM MEASUREMENTS:
{rag_data}

Now provide a RICH ANALYSIS that:
1. Confirms/contextualizes the measured data
2. Explains the mechanisms behind the results
3. Discusses trade-offs (speed vs accuracy, cost vs quality)
4. Provides implementation guidance
Use markdown, be specific, cite the data above."""

                    answer, err = call_llm("🦙 Groq (Llama-3.3-70B) — free", system_prompt,  # Force Groq for Hybrid
                                           [{"role":m["role"],"content":m["content"]} for m in msgs])
                    if err:
                        st.error(f"**Groq error:** {err}")
                        msg_out = rag_data  # Fallback to RAG data
                    else:
                        st.markdown(answer)
                        msg_out = answer
                    msgs.append({"role": "assistant", "content": msg_out})

                else:  # Pure Groq
                    data_ctx = build_data_context()
                    system_prompt = f"""You are an expert AI assistant for the InfoRates research project on spatiotemporal aliasing in video action recognition.
{data_ctx}
Given the user's activity recognition task, recommend:
1. **Architecture** (cite TDS and avg drop data)
2. **Frame rate / stride** (Nyquist reasoning)
3. **Observation window / clip duration**
4. **Spatial resolution** (CNN vs Transformer behavior)
5. **Expected accuracy** from empirical measurements
Be specific, cite actual numbers, use markdown. Keep it concise."""
                    answer, err = call_llm(engine, system_prompt,
                                           [{"role":m["role"],"content":m["content"]} for m in msgs])
                    if err:
                        st.error(f"**Groq error:** {err}")
                        msg_out = f"[Error: {err}]"
                    else:
                        st.markdown(answer)
                        msg_out = answer
                    msgs.append({"role": "assistant", "content": msg_out})

        # Empirical chart after every response
        if msgs and not df_sw.empty:
            with st.expander("📊 Aliasing curves for reference"):
                sel_ds_ai = st.selectbox("Dataset", DS_KEYS, format_func=lambda x: DS_LABELS[x], key="ai_ds")
                sub_ai = df_sw[(df_sw.dataset==sel_ds_ai)&(df_sw.coverage==100)]
                fig_ai = go.Figure()
                for mk in MODEL_KEYS:
                    grp = sub_ai[sub_ai.model==mk].sort_values("stride")
                    if grp.empty or grp["acc"].max()<2: continue
                    fig_ai.add_trace(go.Scatter(
                        x=grp["stride"], y=grp["acc"], mode="lines+markers",
                        name=MODEL_NAMES[mk],
                        line=dict(color=FAM_COLOR.get(FAMILIES[mk],"#999"),width=2),
                        marker=dict(size=7),
                    ))
                fig_ai.update_xaxes(type="log", tickvals=[1,2,4,8,16],
                                     ticktext=["s=1","s=2","s=4","s=8","s=16"])
                fig_ai.update_yaxes(title="Top-1 (%)")
                fig_ai.update_layout(height=320, legend=dict(orientation="h",y=-0.3), margin=dict(b=80))
                st.plotly_chart(fig_ai, use_container_width=True)

    if msgs:
        if st.button("🗑️ Clear conversation"):
            st.session_state[session_key] = []
            st.rerun()
