"""
InfoRates Dashboard — ACCV 2026
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
st.sidebar.caption("ACCV 2026 · Spatiotemporal Aliasing")

page = st.sidebar.radio("", [
    "🏠 Overview & TDS",
    "📊 Aliasing Curves",
    "🔲 Heatmaps",
    "📐 ANOVA & Variance",
    "🌀 Spectral Analysis",
    "⏱ Clip Duration",
    "🔀 Routing & Efficiency",
    "🖼 Spatial Aliasing (P3)",
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
    st.markdown("**ACCV 2026** · 8 architectures · 7 datasets · 1,400 eval configs")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Architectures", "8", "CNN + Transformer + SSM")
    c2.metric("Datasets", "7", "4 semantic domains")
    c3.metric("Eval configs", "1,400", "coverage × stride grid")
    c4.metric("P3 checkpoints", f"{len(df_p3)}/224", "resolution retraining")
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

    # Architecture summary table at stride=16, cov=100%
    st.subheader("Cross-Architecture Aliasing Summary (stride=1 → stride=16, coverage=100%)")
    if not df_sw.empty:
        rows = []
        sub = df_sw[df_sw.coverage == 100]
        for mk in MODEL_KEYS:
            row = {"Model": MODEL_NAMES[mk], "Family": FAMILIES[mk]}
            drops, accs = [], []
            for ds in DS_KEYS:
                s1  = sub[(sub.model==mk)&(sub.dataset==ds)&(sub.stride==1)]["acc"]
                s16 = sub[(sub.model==mk)&(sub.dataset==ds)&(sub.stride==16)]["acc"]
                if s1.empty or s16.empty: continue
                if s1.values[0] > 5:
                    drops.append(s1.values[0] - s16.values[0])
                    accs.append(s1.values[0])
                row[DS_LABELS[ds].split(" (")[0]] = f"{s1.values[0]:.0f}/{-(s1.values[0]-s16.values[0]):.0f}"
            row["Avg base acc"] = f"{np.mean(accs):.1f}%" if accs else "—"
            row["Avg drop (pp)"] = f"{np.mean(drops):.1f}" if drops else "—"
            rows.append(row)
        df_tbl = pd.DataFrame(rows)
        st.dataframe(df_tbl, use_container_width=True)
        st.caption("Format: base@s1 / drop(pp). Feature-collapsed cells excluded.")


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

    if show_all_ds:
        cols = st.columns(3)
        for i, ds in enumerate(DS_KEYS):
            sub = df_sw[(df_sw.model==sel_model) & (df_sw.dataset==ds)]
            if sub.empty: continue
            pivot = sub.pivot(index="coverage", columns="stride", values="acc")
            fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                            text_auto=".0f",
                            labels=dict(x="Stride", y="Coverage (%)"),
                            title=DS_LABELS[ds].split(" (")[0])
            fig.update_layout(height=280, margin=dict(t=40,b=10,l=0,r=0),
                              coloraxis_showscale=False)
            cols[i % 3].plotly_chart(fig, use_container_width=True)
    else:
        sel_ds = st.sidebar.selectbox("Dataset", DS_KEYS, format_func=lambda x: DS_LABELS[x])
        sub = df_sw[(df_sw.model==sel_model) & (df_sw.dataset==sel_ds)]
        if not sub.empty:
            pivot = sub.pivot(index="coverage", columns="stride", values="acc")
            fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                            text_auto=".1f",
                            labels=dict(x="Stride", y="Coverage (%)", color="Top-1 (%)"),
                            title=f"{MODEL_NAMES[sel_model]} on {DS_LABELS[sel_ds]}")
            fig.update_layout(height=420, margin=dict(t=60))
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
        pivot = sub.pivot(index="coverage", columns="stride", values="acc")
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                        text_auto=".0f", title=f"{MODEL_NAMES[mk]} ({FAMILIES[mk]})")
        fig.update_layout(height=300, margin=dict(t=50,b=10), coloraxis_showscale=False)
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
# 🖼 SPATIAL ALIASING (P3)
# =============================================================================
elif page == "🖼 Spatial Aliasing (P3)":
    st.title("Spatial Aliasing & Resolution Retraining (P3)")

    if df_p3.empty:
        st.warning("P3 logs not found or no completed checkpoints.")
        st.stop()

    st.sidebar.subheader("Settings")
    sel_ds_p3 = st.sidebar.selectbox("Dataset", sorted(df_p3["dataset"].unique()),
                                      format_func=lambda x: DS_LABELS.get(x, x), key="p3_ds")
    sel_mods_p3 = st.sidebar.multiselect("Models", sorted(df_p3["model_name"].dropna().unique()),
                                          default=sorted(df_p3["model_name"].dropna().unique()), key="p3_mods")

    sub = df_p3[(df_p3.dataset == sel_ds_p3) & df_p3.model_name.isin(sel_mods_p3)]

    # Get native E1 accuracy for each model
    native_acc = {}
    if not df_sw.empty:
        e1_sub = df_sw[(df_sw.dataset == sel_ds_p3) & (df_sw.coverage == 100) & (df_sw.stride == 1)]
        for mk in MODEL_KEYS:
            row = e1_sub[e1_sub.model == mk]
            if not row.empty:
                native_acc[mk] = row["acc"].values[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Retrained accuracy vs resolution")
        fig = go.Figure()
        for mdl_name, grp in sub.groupby("model_name"):
            mk = {v:k for k,v in MODEL_NAMES.items()}.get(mdl_name)
            color = FAM_COLOR.get(FAMILIES.get(mk,"CNN"), "#999")
            grp = grp.sort_values("res")
            fig.add_trace(go.Scatter(
                x=grp["res"], y=grp["acc"],
                mode="lines+markers", name=f"{mdl_name} (retrained)",
                line=dict(color=color, width=2), marker=dict(size=9),
                hovertemplate=f"<b>{mdl_name}</b><br>%{{x}}px → %{{y:.1f}}%",
            ))
            if mk in native_acc:
                nat = native_acc[mk]
                res_range = [grp["res"].min(), grp["res"].max()]
                fig.add_trace(go.Scatter(
                    x=res_range, y=[nat, nat],
                    mode="lines", name=f"{mdl_name} (native)",
                    line=dict(color=color, dash="dot", width=1),
                    showlegend=True,
                ))
        fig.update_xaxes(title="Training resolution (px)", tickvals=[96,112,160,224,336])
        fig.update_yaxes(title="Top-1 (%)")
        fig.update_layout(height=400, legend=dict(orientation="h",y=-0.3),
                          margin=dict(b=80), title=f"{DS_LABELS.get(sel_ds_p3,'')}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Δ vs native resolution")
        rows_tbl = []
        for _, r in sub.iterrows():
            mk = {v:k for k,v in MODEL_NAMES.items()}.get(r["model_name"])
            nat = native_acc.get(mk)
            delta = r["acc"] - nat if nat else None
            rows_tbl.append({
                "Model": r["model_name"], "Res": f"{r['res']}px",
                "Retrained": f"{r['acc']:.1f}%",
                "Native": f"{nat:.1f}%" if nat else "—",
                "Δ": f"{delta:+.1f}pp" if delta else "—",
                "✓": "✅" if (delta and delta > 0) else ("❌" if delta else "—"),
            })
        if rows_tbl:
            tbl = pd.DataFrame(rows_tbl).sort_values(["Model","Res"])
            st.dataframe(tbl.reset_index(drop=True), use_container_width=True)

            n_beats = sum(1 for r in rows_tbl if r["✓"] == "✅")
            n_total = sum(1 for r in rows_tbl if r["✓"] != "—")
            st.metric("Beats native", f"{n_beats}/{n_total}",
                      f"{n_beats/n_total*100:.0f}% of completed checkpoints" if n_total else "")

    st.subheader("P3 completion status")
    prog_df = pd.DataFrame({"Model": [MODEL_NAMES[k] for k in MODEL_KEYS],
                             "Done": [len(df_p3[df_p3.model==k]) for k in MODEL_KEYS],
                             "Total": [28]*8})
    prog_df["Progress"] = prog_df["Done"].astype(str) + "/" + prog_df["Total"].astype(str)
    prog_df["Pct"] = (prog_df["Done"]/prog_df["Total"]*100).round(0)
    st.dataframe(prog_df[["Model","Progress","Pct"]].rename(columns={"Pct":"% done"}),
                 use_container_width=True)
    total_done = len(df_p3.drop_duplicates(["model","dataset","res"]))
    st.progress(total_done/224, text=f"P3 overall: {total_done}/224 checkpoints ({total_done/224*100:.0f}%)")


# =============================================================================
# 🎯 ARCHITECTURE RECOMMENDER
# =============================================================================
elif page == "🎯 Architecture Recommender":
    st.title("Architecture & Sensor Recommender")
    st.caption("Get a principled recommendation for your specific deployment scenario based on measured aliasing data.")

    col_in, col_out = st.columns([1, 1.5])

    with col_in:
        st.subheader("Your Scenario")
        sel_domain = st.selectbox("Target domain", list(DS_LABELS.values()))
        sel_ds_k = [k for k,v in DS_LABELS.items() if v == sel_domain][0]

        src_fps  = st.number_input("Source video FPS", value=30, step=5, min_value=1)
        inf_fps  = st.slider("Inference FPS (frames sampled)", 1, 30, 8)
        spatial  = st.checkbox("Inference at non-native spatial resolution?", value=False)
        latency  = st.selectbox("Deployment tier", ["Edge (< 20ms/clip)", "Server (< 100ms)", "Offline"])

        effective_stride = max(1, round(src_fps / inf_fps))
        # Snap to grid
        for s in [1,2,4,8,16]:
            if effective_stride <= s:
                effective_stride = s
                break
        st.metric("Effective stride", effective_stride,
                  f"{src_fps}/{inf_fps} FPS → stride={effective_stride}")

    with col_out:
        st.subheader("Recommendation")
        tds_val = TDS.get(sel_ds_k, 0)

        # TDS tier
        if tds_val > 35:
            st.error(f"**TDS = {tds_val:.1f}pp — HIGH temporal demand**  \n"
                     "Dense sampling is essential. Use SSM or divided-attention Transformer.")
        elif tds_val > 18:
            st.warning(f"**TDS = {tds_val:.1f}pp — MODERATE temporal demand**  \n"
                       "Moderate aliasing at stride > 8. Prefer SSM/TimeSformer for safety margin.")
        else:
            st.success(f"**TDS = {tds_val:.1f}pp — LOW temporal demand**  \n"
                       "Appearance-dominated. Any architecture works; optimize for compute/latency.")

        # Expected accuracy at chosen stride from real data
        if not df_sw.empty:
            sub_rec = df_sw[(df_sw.dataset == sel_ds_k) &
                            (df_sw.coverage == 100) &
                            (df_sw.stride == effective_stride)].sort_values("acc", ascending=False)
            if not sub_rec.empty:
                st.subheader(f"Expected Top-1 accuracy at stride={effective_stride}")
                fig = px.bar(sub_rec, x="model_name", y="acc", color="family",
                             color_discrete_map=FAM_COLOR,
                             text="acc", labels={"model_name":"","acc":"Top-1 (%)"},
                             height=300, category_orders={"model_name": sub_rec["model_name"].tolist()})
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(margin=dict(t=20,b=40), showlegend=True,
                                  xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

        # Guidance cards
        st.subheader("Deployment guidance")
        if effective_stride >= 8 and tds_val > 18:
            st.error("⚠️ stride≥8 on moderate/high-TDS domain: expect significant aliasing. "
                     "Increase FPS or switch to VideoMamba/TimeSformer.")
        if spatial:
            st.warning("🖼️ Non-native resolution: Transformers (ViViT, VideoMAE, TimeSformer) and VideoMamba "
                       "are robust across 96–336px. CNNs require retraining at target resolution.")
        if latency.startswith("Edge"):
            st.info("⚡ Edge: R3D-18 or MC3-18 offer best latency/accuracy. "
                    "Use entropy routing to skip dense inference on ~84% of videos.")
        elif latency.startswith("Server"):
            st.info("🖥️ Server: TimeSformer or VideoMamba for temporal-critical datasets. "
                    "VideoMAE for highest accuracy on appearance datasets.")
        else:
            st.info("☁️ Offline: VideoMAE gives highest raw accuracy. Use full 16-frame dense sampling.")
