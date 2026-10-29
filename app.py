"""Streamlit demo for the disease-spread GNN project.

    streamlit run app.py

Reads data/processed/ (Person 1), outputs/demo/ (exported from Person 2's
trained models by src/demo/export_demo_data.py) and outputs/evaluation/
(Person 3's evaluation reports).
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
DEMO = ROOT / "outputs" / "demo"
EVAL = ROOT / "outputs" / "evaluation"

GRAPH_LABELS = {
    "geographic": "Geographic adjacency",
    "gravity": "Gravity model",
    "learned": "Learned graph",
    "hybrid": "Hybrid (geographic + learned)",
}
SERIES_COLORS = {"Actual": "#1f2937", "Model": "#2563eb", "Persistence": "#d97706"}
MAP_CENTER = {"lat": 22.5, "lon": 80.0}

st.set_page_config(page_title="Disease Spread GNN", page_icon="🦠", layout="wide")


# ---------------------------------------------------------------- data loading

@st.cache_data
def load_districts():
    gdf = gpd.read_file(DEMO / "districts.geojson")
    gdf["censuscode"] = gdf["censuscode"].astype(int)
    pts = gdf.geometry.representative_point()
    gdf["lat"], gdf["lon"] = pts.y, pts.x
    gdf["label"] = gdf["DISTRICT"] + ", " + gdf["ST_NM"]
    return json.loads(gdf.to_json()), gdf.drop(columns="geometry")


@st.cache_data
def load_nodes():
    return pd.read_csv(DATA / "nodes.csv")


@st.cache_data
def load_history():
    return pd.read_csv(DEMO / "history.csv", parse_dates=["week"])


@st.cache_data
def load_forecasts():
    return pd.read_csv(DEMO / "forecasts.csv", parse_dates=["week"])


@st.cache_data
def load_edges(graph):
    if graph == "geographic":
        edges = pd.read_csv(DATA / "edges_geographic_adjacency.csv")
    elif graph == "gravity":
        edges = pd.read_csv(DATA / "edges_gravity.csv")
    else:
        path = DEMO / "learned_edges.csv"
        if not path.exists():
            return None
        edges = pd.read_csv(path)
    return edges


@st.cache_data
def load_json(name):
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def demo_bundle_ready():
    return (DEMO / "forecasts.csv").exists() and (DEMO / "districts.geojson").exists()


def edge_trace(edges, lookup, color, width):
    """Edges as one line trace with None breaks, which plotly draws fast."""
    lat, lon = [], []
    for s, d in zip(edges["src"], edges["dst"]):
        if s in lookup.index and d in lookup.index:
            lat += [lookup.at[s, "lat"], lookup.at[d, "lat"], None]
            lon += [lookup.at[s, "lon"], lookup.at[d, "lon"], None]
    return go.Scattermap(lat=lat, lon=lon, mode="lines", line={"color": color, "width": width},
                         hoverinfo="skip", showlegend=False)


def district_picker(meta, key, default="Mumbai"):
    labels = meta.sort_values("label")["label"].tolist()
    default_label = next((l for l in labels if l.startswith(default + ",")), labels[0])
    label = st.selectbox("District", labels, index=labels.index(default_label), key=key)
    return int(meta.loc[meta["label"] == label, "censuscode"].iloc[0]), label


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------- pages

def page_overview():
    st.title("Predicting disease spread across India's districts with graph neural networks")
    st.markdown(
        "We forecast weekly COVID-19 cases for Indian districts. A **spatio-temporal GNN** "
        "learns how infection moves between districts and predicts each district's "
        "**transmission rate (β)**. A differentiable **SIR layer** then turns β into case "
        "counts and an interpretable **Rt**, which we use to rank high-risk districts."
    )

    nodes = load_nodes()
    history = load_history() if demo_bundle_ready() else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts (graph nodes)", f"{len(nodes)}")
    c2.metric("With usable case data", f"{history['censuscode'].nunique()}" if history is not None else "527")
    c3.metric("Weeks of data", "Apr 2020 – Oct 2022")
    c4.metric("Graph variants compared", "4")

    st.subheader("Pipeline")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
                fillcolor="#eef2ff", color="#6366f1"];
          edge [color="#6b7280"];
          data  [label="Case, population &\\nboundary data"];
          graphs [label="District graphs\\ngeographic · gravity\\nlearned · hybrid"];
          gnn   [label="GRU temporal encoder\\n+ graph convolution", fillcolor="#dbeafe", color="#2563eb"];
          beta  [label="β per district"];
          sir   [label="Differentiable SIR\\nnew = β·S·I / N", fillcolor="#dcfce7", color="#16a34a"];
          out   [label="Predicted cases · Rt\\nTop-k risk ranking", fillcolor="#fef3c7", color="#d97706"];
          data -> graphs -> gnn -> beta -> sir -> out;
        }
        """
    )

    st.subheader("Who built what")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Person 1: data and graphs**\n\nCleaned weekly district panel, census "
                "population, and the geographic and gravity graphs.")
    c2.markdown("**Person 2: spatio-temporal GNN**\n\nGRU + graph convolution, learned "
                "adjacency, and training for all four graph variants.")
    c3.markdown("**Person 3: SIR and evaluation**\n\nSIR output layer, wave-onset metrics, "
                "turning-point loss, rolling-origin evaluation, and risk ranking.")

    st.info("The test window is the **Omicron wave (27 Dec 2021 – 7 Feb 2022)**. The models "
            "never see these weeks during training.")


def page_data():
    st.title("Data and graphs")
    geojson, meta = load_districts()
    history = load_history()

    tab_map, tab_graph, tab_series = st.tabs(["Case map", "Graph variants", "District history"])

    with tab_map:
        weeks = sorted(history["week"].dt.date.unique())
        default = weeks.index(pd.Timestamp("2021-05-03").date()) if pd.Timestamp("2021-05-03").date() in weeks else 0
        week = st.select_slider("Week", options=weeks, value=weeks[default],
                                help="The default week is the Delta-wave peak.")
        wk = history[history["week"].dt.date == week].merge(meta, on="censuscode")
        wk["per_100k"] = wk["new_confirmed"] / wk["Population"] * 1e5
        fig = px.choropleth_map(
            wk, geojson=geojson, locations="censuscode", featureidkey="properties.censuscode",
            color="per_100k", color_continuous_scale="Reds", range_color=(0, wk["per_100k"].quantile(0.95) or 1),
            hover_name="label", hover_data={"new_confirmed": ":,.0f", "per_100k": ":.1f", "censuscode": False},
            labels={"per_100k": "New cases / 100k", "new_confirmed": "New cases"},
            map_style="carto-positron", center=MAP_CENTER, zoom=3.4, opacity=0.85, height=620,
        )
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(wk)} districts reported that week. Districts in grey have no "
                   "district-level data that week (e.g. their state stopped publishing).")

    with tab_graph:
        c1, c2 = st.columns([1, 3])
        with c1:
            graph = st.radio("Graph", ["geographic", "gravity", "learned"],
                             format_func=GRAPH_LABELS.get)
            focus_code, focus_label = district_picker(meta, "graph_focus", default="Pune")
            st.markdown({
                "geographic": "Edge = the two districts share a border.",
                "gravity": "Edge weight = pop_i · pop_j / distance². We keep each district's "
                           "top 6 links, so distant metros can connect.",
                "learned": "Edges the model learned from data. We show each district's top 3.",
            }[graph])
        edges = load_edges(graph)
        with c2:
            if edges is None:
                st.warning("Learned edges not exported yet. Run `src/demo/export_demo_data.py`.")
            else:
                lookup = meta.set_index("censuscode")
                focus = edges[(edges["src"] == focus_code) | (edges["dst"] == focus_code)]
                neighbours = set(focus["src"]) | set(focus["dst"])
                neighbours.discard(focus_code)
                fig = go.Figure()
                fig.add_trace(edge_trace(edges, lookup, "rgba(99,102,241,0.35)", 1))
                fig.add_trace(edge_trace(focus, lookup, "#dc2626", 3))
                fig.add_trace(go.Scattermap(
                    lat=meta["lat"], lon=meta["lon"], mode="markers", text=meta["label"],
                    marker={"size": 4, "color": "#4338ca"}, hoverinfo="text", showlegend=False))
                fig.update_layout(map={"style": "carto-positron", "center": MAP_CENTER, "zoom": 3.4},
                                  height=620, margin={"l": 0, "r": 0, "t": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"{len(edges):,} edges. **{focus_label}** is linked to: " +
                           ", ".join(sorted(lookup.loc[list(neighbours), "DISTRICT"])) if neighbours
                           else f"{len(edges):,} edges. {focus_label} has no edges in this graph.")

    with tab_series:
        code, label = district_picker(meta, "series_district")
        s = history[history["censuscode"] == code]
        fig = px.bar(s, x="week", y="new_confirmed", labels={"new_confirmed": "New cases", "week": ""},
                     color_discrete_sequence=["#2563eb"], height=420)
        fig.add_vrect(x0="2021-12-27", x1="2022-02-13", fillcolor="#fde68a", opacity=0.35,
                      line_width=0, annotation_text="test window", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Weekly new confirmed cases. Gaps are weeks with no district-level data.")


def page_forecast():
    st.title("Forecast explorer")
    st.markdown("Pick a district and a graph variant. For each test week, the model predicts β "
                "from the previous 8 weeks, and the SIR layer converts it to cases. "
                "**Persistence** is the baseline: next week equals this week.")
    _, meta = load_districts()
    fc = load_forecasts()
    history = load_history()

    c1, c2 = st.columns([1, 1])
    with c1:
        graph = st.selectbox("Graph variant", list(fc["graph"].unique()), format_func=GRAPH_LABELS.get,
                             index=list(fc["graph"].unique()).index("hybrid") if "hybrid" in fc["graph"].unique() else 0)
    with c2:
        in_test = meta[meta["censuscode"].isin(fc.loc[fc["valid"], "censuscode"].unique())]
        code, label = district_picker(in_test, "fc_district")

    d = fc[(fc["graph"] == graph) & (fc["censuscode"] == code)].sort_values("week")
    ctx = history[(history["censuscode"] == code) & (history["week"] >= "2021-10-01") &
                  (history["week"] <= "2022-02-13")]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctx["week"], y=ctx["new_confirmed"], name="Actual",
                             line={"color": SERIES_COLORS["Actual"], "width": 2}))
    fig.add_trace(go.Scatter(x=d["week"], y=d["predicted"], name=f"Model ({graph})",
                             line={"color": SERIES_COLORS["Model"], "width": 3}, mode="lines+markers"))
    fig.add_trace(go.Scatter(x=d["week"], y=d["persistence"], name="Persistence",
                             line={"color": SERIES_COLORS["Persistence"], "dash": "dash"}))
    fig.add_vrect(x0="2021-12-27", x1="2022-02-13", fillcolor="#fde68a", opacity=0.3, line_width=0)
    fig.update_layout(height=440, yaxis_title="Weekly new cases", hovermode="x unified",
                      legend={"orientation": "h", "y": 1.08}, margin={"t": 30})
    st.plotly_chart(fig, use_container_width=True)

    v = d[d["valid"]]
    if len(v):
        c1, c2, c3 = st.columns(3)
        c1.metric("Model MAE (this district)", f"{mae(v['predicted'], v['actual']):,.0f}")
        c2.metric("Persistence MAE", f"{mae(v['persistence'], v['actual']):,.0f}")
        c3.metric("Mean implied Rt", f"{v['rt'].mean():.2f}")

    st.subheader("All districts: test-window error by graph")
    rows = []
    for g, grp in fc[fc["valid"]].groupby("graph"):
        on, flat = grp[grp["onset"]], grp[~grp["onset"]]
        rows.append({"model": GRAPH_LABELS[g], "MAE all": mae(grp["predicted"], grp["actual"]),
                     "MAE onset weeks": mae(on["predicted"], on["actual"]),
                     "MAE flat weeks": mae(flat["predicted"], flat["actual"])})
    v = fc[fc["valid"] & (fc["graph"] == fc["graph"].iloc[0])]
    on, flat = v[v["onset"]], v[~v["onset"]]
    rows.append({"model": "Persistence baseline", "MAE all": mae(v["persistence"], v["actual"]),
                 "MAE onset weeks": mae(on["persistence"], on["actual"]),
                 "MAE flat weeks": mae(flat["persistence"], flat["actual"])})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:,.0f}" for c in ["MAE all", "MAE onset weeks", "MAE flat weeks"]})
                 .highlight_min(subset=["MAE all", "MAE onset weeks", "MAE flat weeks"], color="#dcfce7"),
                 hide_index=True, use_container_width=True)
    st.caption("From the models trained for this demo (one seed each), so the numbers differ "
               "slightly from the multi-seed results on the Results page. Onset weeks are weeks "
               "where cases grew at least 20% to 20 or more.")


def page_risk():
    st.title("Risk map: where should resources go this week?")
    geojson, meta = load_districts()
    fc = load_forecasts()

    c1, c2, c3, c4 = st.columns(4)
    graph = c1.selectbox("Graph variant", list(fc["graph"].unique()), format_func=GRAPH_LABELS.get, key="risk_graph")
    week = c2.selectbox("Week", sorted(fc["week"].dt.date.unique()))
    mode = c3.radio("Rank by", ["Predicted growth", "Predicted cases"], horizontal=False)
    k = c4.slider("Top k", 5, 50, 10, step=5)

    wk = fc[(fc["graph"] == graph) & (fc["week"].dt.date == week) & fc["valid"]].merge(meta, on="censuscode")
    wk["growth"] = wk["predicted"] / np.maximum(wk["persistence"], 1.0)
    score = "growth" if mode == "Predicted growth" else "predicted"
    wk = wk.sort_values(score, ascending=False)
    top = wk.head(k)
    hits = int(top["onset"].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Precision@{k}", f"{hits / k:.0%}", help="Share of flagged districts that actually hit a wave onset.")
    m2.metric("Base rate (random pick)", f"{wk['onset'].mean():.0%}")
    m3.metric("Districts with Rt > 1", f"{(wk['rt'] > 1).sum()} / {len(wk)}")

    left, right = st.columns([3, 2])
    with left:
        wk["Rt"] = wk["rt"].clip(0, 2.5)
        fig = px.choropleth_map(
            wk, geojson=geojson, locations="censuscode", featureidkey="properties.censuscode",
            color="Rt", color_continuous_scale="RdYlGn_r", range_color=(0.5, 1.5),
            hover_name="label", hover_data={"predicted": ":,.0f", "actual": ":,.0f", "rt": ":.2f",
                                            "Rt": False, "censuscode": False},
            map_style="carto-positron", center=MAP_CENTER, zoom=3.4, opacity=0.8, height=600,
        )
        fig.add_trace(go.Scattermap(lat=top["lat"], lon=top["lon"], mode="markers",
                                    marker={"size": 11, "color": "#111827"}, text=top["label"],
                                    hoverinfo="text", name=f"Top {k}"))
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"y": 0.02, "x": 0.01})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        table = top[["DISTRICT", "ST_NM", "predicted", "actual", "rt", "onset"]].rename(columns={
            "DISTRICT": "District", "ST_NM": "State", "predicted": "Predicted", "actual": "Actual",
            "rt": "Rt", "onset": "Surged?"})
        table["Surged?"] = table["Surged?"].map({True: "✅ yes", False: "—"})
        st.dataframe(table.style.format({"Predicted": "{:,.0f}", "Actual": "{:,.0f}", "Rt": "{:.2f}"}),
                     hide_index=True, use_container_width=True, height=560)
    st.caption("Rt = β·S/N, assuming a one-week infectious period. Black dots mark the top-k "
               "districts. *Surged?* is checked against what actually happened that week.")


def page_results():
    st.title("Evaluation results")
    st.caption("Person 3's evaluation. Multi-seed and rolling-origin numbers from outputs/evaluation/.")

    tab1, tab2, tab3, tab4 = st.tabs(["Headline", "Turning-point loss", "Rolling origin", "Risk ranking"])

    with tab1:
        w = load_json("wave_onset_comparison.json")["results"]
        base = w["persistence"]["wave_onset"]["MAE"]
        df = pd.DataFrame([{
            "model": GRAPH_LABELS.get(m, "Persistence baseline"),
            "MAE all": r["all"]["MAE"], "MAE onset": r["wave_onset"]["MAE"], "MAE flat": r["flat"]["MAE"],
            "onset vs baseline": None if m == "persistence" else (r["wave_onset"]["MAE"] - base) / base,
        } for m, r in w.items()])
        st.markdown("**Every graph variant beats persistence at wave onset**, the weeks that matter "
                    "for early warning, even though most lose on the overall average.")
        st.dataframe(df.style.format({"MAE all": "{:,.0f}", "MAE onset": "{:,.0f}", "MAE flat": "{:,.0f}",
                                      "onset vs baseline": lambda v: "" if pd.isna(v) else f"{v:+.1%}"}),
                     hide_index=True, use_container_width=True)
        st.image(str(EVAL / "fig1_onset_vs_flat.png"), use_container_width=True)
        st.image(str(EVAL / "fig4_forecast_examples.png"), caption="Example forecasts", use_container_width=True)

    with tab2:
        runs = load_json("weighted_ablation.json")["runs"]
        rows = []
        for g in GRAPH_LABELS:
            a, b = runs.get(f"{g}_w1"), runs.get(f"{g}_w5")
            if a and b:
                rows.append({"graph": GRAPH_LABELS[g],
                             "onset MAE (uniform)": f"{a['onset_MAE_mean']:,.0f} ± {a['onset_MAE_std']:.0f}",
                             "onset MAE (weighted ×5)": f"{b['onset_MAE_mean']:,.0f} ± {b['onset_MAE_std']:.0f}",
                             "change": f"{(b['onset_MAE_mean'] - a['onset_MAE_mean']) / a['onset_MAE_mean']:+.1%}"})
        st.markdown("Weighting wave-onset weeks 5× in the loss lowers onset error for **every** graph "
                    "(3 seeds each). The cost is higher error on flat weeks, which is the intended trade.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.image(str(EVAL / "fig2_weighting_ablation.png"), use_container_width=True)

    with tab3:
        rows = []
        for g in ["hybrid", "gravity"]:
            r = load_json(f"rolling_origin_{g}.json")["results"]
            if not rows:
                rows.append({"model": "Persistence baseline", **r["persistence"]})
            rows.append({"model": f"{GRAPH_LABELS[g]}, uniform", **r[f"{g}_w1"]})
            rows.append({"model": f"{GRAPH_LABELS[g]}, weighted", **r[f"{g}_w5"]})
        st.markdown("A real-time simulation: retrain from scratch at each of 12 origin weeks and "
                    "predict only the next week. The gravity graph does best at wave onset.")
        st.dataframe(pd.DataFrame(rows).rename(columns={"all": "MAE all", "onset": "MAE onset", "flat": "MAE flat"})
                     .style.format({"MAE all": "{:,.0f}", "MAE onset": "{:,.0f}", "MAE flat": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)

    with tab4:
        rr = load_json("risk_ranking.json")
        res = rr["results"]
        rows = [{"ranker": n, "NDCG@10 (growth)": r["surge"]["NDCG@10"], "P@10 (growth)": r["surge"]["P@10"],
                 "NDCG@10 (cases)": r["burden"]["NDCG@10"], "P@10 (cases)": r["burden"]["P@10"]}
                for n, r in res.items()]
        st.markdown("Ranked by predicted growth, the models order districts much better than the "
                    "baselines on **NDCG@10** (≈0.34 vs 0.20). On the yes/no call of which districts "
                    "will surge (**Precision@10**), persistence still wins.")
        st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True, use_container_width=True)
        rt = pd.DataFrame([{"model": n, "mean Rt": v["mean"], "share Rt > 1": v["share_above_1"]}
                           for n, v in rr["rt_summary"].items()])
        st.markdown("**Implied Rt** during Omicron: uniform-loss models give Rt < 1, which is "
                    "impossible during a growing wave. The turning-point weighted models (`_w5`) "
                    "give a plausible 1.1–1.2.")
        st.dataframe(rt.style.format({"mean Rt": "{:.2f}", "share Rt > 1": "{:.0%}"}), hide_index=True,
                     use_container_width=True)
        st.image(str(EVAL / "fig3_rt_distribution.png"), use_container_width=True)

    with st.expander("Limitations"):
        st.markdown(
            "- **Forecasts lag:** they peak about a week after the real wave, like persistence.\n"
            "- **β is a proxy:** it is computed from case counts with an SIR formula, not measured, and "
            "is capped at 2.0, so the fastest-growing districts are underestimated.\n"
            "- **Small sample:** 87 weekly samples. That's why results are averaged over seeds and "
            "rolling origins.\n"
            "- **Data ends early:** district-level reporting effectively stops after Oct 2022, and "
            "some states (e.g. UP) stop earlier.")


# ---------------------------------------------------------------- navigation

if not demo_bundle_ready():
    st.error("Demo data not found. Run:\n\n"
             "```\nPYTHONPATH=src python src/training/compare_graphs.py\n"
             "PYTHONPATH=src python src/demo/export_demo_data.py\n```")
    st.stop()

nav = st.navigation([
    st.Page(page_overview, title="Overview", icon="🏠", default=True),
    st.Page(page_data, title="Data and graphs", icon="🗺️", url_path="data"),
    st.Page(page_forecast, title="Forecast explorer", icon="📈", url_path="forecast"),
    st.Page(page_risk, title="Risk map", icon="🚨", url_path="risk"),
    st.Page(page_results, title="Results", icon="📊", url_path="results"),
])
nav.run()
