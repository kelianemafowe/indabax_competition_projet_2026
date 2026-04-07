import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib
import pickle
import json
from datetime import datetime, timedelta


# ============================
# CONFIG
# ============================
st.set_page_config(
    page_title="VRI Dashboard — Cameroun",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================
# CSS
# ============================
def load_css():
    with open("part_css.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# ============================
# DATA & MODEL
# ============================
@st.cache_data
def load_data():
    df = pd.read_excel("data/Dataset_complet_Meteo.xlsx")
    df["time"] = pd.to_datetime(df["time"])

    numeric_cols = [
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "precipitation_hours", "rain_sum",
        "wind_speed_10m_max", "et0_fao_evapotranspiration",
        "sunshine_duration", "latitude", "longitude"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["temperature_2m_mean", "precipitation_sum", "latitude", "longitude"], inplace=True)
    return df


@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/lgb_global_t7.pkl")
        features = joblib.load("models/feature_cols.pkl")
        return model, features
    except:
        return None, None


@st.cache_data
def load_df_feat():
    try:
        with open("outputs/df_feat.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None


df = load_data()
model, FEATURE_COLS = load_model()
df_feat = load_df_feat()


# ============================
# HELPERS VRI
# ============================
def compute_vri(T, precip, precip_hours, et0=None, is_dry=0):
    """Calcule le VRI avec la formule pondérée."""
    T_opt  = np.exp(-((T - 25) ** 2) / 50)
    # Proxy HR depuis température (Magnus)
    HR_est = min(100, max(0, 60 + (25 - T) * 2))
    HR_opt = np.exp(-((HR_est - 70) ** 2) / 450)
    P_opt  = 1.0 if (precip > 5 and precip_hours > 4) else 0.5
    vri = (0.4 * T_opt + 0.35 * HR_opt + 0.25 * P_opt) * (1 - is_dry)
    return float(np.clip(vri, 0, 1))


def vri_to_risk(vri):
    if vri < 0.33:
        return "FAIBLE", "vri-faible", "#00B050"
    elif vri < 0.66:
        return "MODÉRÉ", "vri-modere", "#FF9900"
    else:
        return "ÉLEVÉ", "vri-eleve", "#C00000"


def get_month_name(m):
    months = ["Jan","Fév","Mar","Avr","Mai","Juin",
              "Juil","Aoû","Sep","Oct","Nov","Déc"]
    return months[m - 1]


# ============================
# HEADER
# ============================
st.markdown("""
    <div class="header">
        <div class="header-icon"></div>
        <div class="header-text">
            <h1>Indice de Risque Vectoriel — Cameroun</h1>
            <p>Surveillance climatique & prédiction du risque paludisme · IndabaX 2026</p>
        </div>
        <div class="header-badge">t+7 jours</div>
    </div>
""", unsafe_allow_html=True)


# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Filtres</div>', unsafe_allow_html=True)

    ville_choisie = st.selectbox("Ville", sorted(df["city"].unique()))

    date_min = df["time"].min().date()
    date_max = df["time"].max().date()
    dates = st.date_input("Période", (date_min, date_max),
                          min_value=date_min, max_value=date_max)
    if len(dates) == 2:
        date_debut, date_fin = dates
    else:
        date_debut, date_fin = date_min, date_max

    st.markdown("---")
    st.markdown('<div class="sidebar-title">A propos du VRI</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info">
    <b>Formule :</b><br>
    VRI = (0.4·T<sub>opt</sub> + 0.35·HR<sub>opt</sub> + 0.25·P<sub>opt</sub>) × (1−S)
    <br><br>
    <b>Niveaux :</b><br>
    🟢 Faible : VRI &lt; 0.33<br>
    🟡 Modéré : 0.33 ≤ VRI &lt; 0.66<br>
    🔴 Élevé : VRI ≥ 0.66
    </div>
    """, unsafe_allow_html=True)


# ============================
# FILTRAGE
# ============================
df_ville = df[
    (df["city"] == ville_choisie) &
    (df["time"].dt.date >= date_debut) &
    (df["time"].dt.date <= date_fin)
].copy()

if df_ville.empty:
    st.warning("Aucune donnée disponible pour cette sélection.")
    st.stop()

# Calcul VRI pour la ville
df_ville["VRI"] = df_ville.apply(
    lambda r: compute_vri(
        r["temperature_2m_mean"],
        r["precipitation_sum"],
        r.get("precipitation_hours", 0),
    ), axis=1
)

vri_moy      = df_ville["VRI"].mean()
temp_moy     = df_ville["temperature_2m_mean"].mean()
pluie_total  = df_ville["precipitation_sum"].sum()
vent_max     = df_ville["wind_speed_10m_max"].max()
risk_label, risk_class, risk_color = vri_to_risk(vri_moy)


# ============================
# KPIs
# ============================
st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card kpi-vri {risk_class}">
        <div class="kpi-value">{vri_moy:.2f}</div>
        <div class="kpi-label">VRI Moyen</div>
        <div class="kpi-badge">{risk_label}</div>
    </div>""", unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{temp_moy:.1f}°C</div>
        <div class="kpi-label">Température moy.</div>
    </div>""", unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{pluie_total:.0f} mm</div>
        <div class="kpi-label">Précipitations</div>
    </div>""", unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{vent_max:.0f} km/h</div>
        <div class="kpi-label">Vent max</div>
    </div>""", unsafe_allow_html=True)

with kpi5:
    jours_eleve = (df_ville["VRI"] >= 0.66).sum()
    pct_eleve   = jours_eleve / len(df_ville) * 100 if len(df_ville) > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{jours_eleve}</div>
        <div class="kpi-label">Jours risque élevé</div>
        <div class="kpi-badge-gray">{pct_eleve:.0f}% de la période</div>
    </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ============================
# TABS
# ============================
tab1, tab2, tab3, tab4 = st.tabs([
    "Carte des risques",
    "Évolution temporelle",
    "Alertes par ville",
    "Prédiction VRI t+7"
])


# ============================
# TAB 1 — CARTE CAMEROUN UNIQUEMENT
# ============================
with tab1:
    st.markdown(f"### Carte du risque vectoriel — {ville_choisie} mise en évidence")

    # Coordonnées limites du Cameroun
    cameroon_bounds = [2.2, 8.8, 8.5, 16.2]  # [min_lat, max_lat, min_lon, max_lon]

    villes_agg = df.groupby("city").agg(
        lat=("latitude", "first"),
        lon=("longitude", "first"),
        temp=("temperature_2m_mean", "mean"),
        precip=("precipitation_sum", "mean"),
        precip_h=("precipitation_hours", "mean") if "precipitation_hours" in df.columns else ("precipitation_sum", "count")
    ).reset_index()

    # Filtrer seulement les villes au Cameroun
    villes_agg = villes_agg[
        (villes_agg["lat"].between(cameroon_bounds[0], cameroon_bounds[1])) &
        (villes_agg["lon"].between(cameroon_bounds[2], cameroon_bounds[3]))
    ]

    villes_agg["VRI"] = villes_agg.apply(
        lambda r: compute_vri(r["temp"], r["precip"], r.get("precip_h", 0)), axis=1
    )

    # Trouver les coordonnées de la ville sélectionnée
    ville_coords = villes_agg[villes_agg["city"] == ville_choisie]
    if not ville_coords.empty:
        ville_lat, ville_lon = ville_coords["lat"].iloc[0], ville_coords["lon"].iloc[0]
        zoom_level = 10  # Zoom sur la ville
        center = [ville_lat, ville_lon]
    else:
        # Centre du Cameroun par défaut
        center = [5.5, 12.5]
        zoom_level = 6

    m = folium.Map(
        location=center, 
        zoom_start=zoom_level,
        tiles="CartoDB dark_matter",
        min_lat=2.2, max_lat=8.8, 
        min_lon=8.5, max_lon=16.2
    )

    for _, row in villes_agg.iterrows():
        lbl, cls, color = vri_to_risk(row["VRI"])
        is_selected = row["city"] == ville_choisie
        radius = 14 if is_selected else 8
        weight = 3 if is_selected else 1

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="white" if is_selected else color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=weight,
            popup=folium.Popup(
                f"<b>{row['city']}</b><br>"
                f"VRI : {row['VRI']:.2f} ({lbl})<br>"
                f"Temp : {row['temp']:.1f}°C<br>"
                f"Précip : {row['precip']:.1f} mm/j",
                max_width=200
            ),
            tooltip=f"{row['city']} · VRI={row['VRI']:.2f}"
        ).add_to(m)

    # Légende
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(0,0,0,0.75);padding:12px 16px;border-radius:10px;
                font-family:sans-serif;font-size:13px;color:white">
        <b>Niveau de risque VRI</b><br>
        <span style="color:#00B050">●</span> Faible (&lt; 0.33)<br>
        <span style="color:#FF9900">●</span> Modéré (0.33–0.66)<br>
        <span style="color:#C00000">●</span> Élevé (≥ 0.66)<br>
        <span style="color:white">○</span> Ville sélectionnée
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=None, height=520, use_container_width=True)


# ============================
# TAB 2 — ÉVOLUTION
# ============================
with tab2:
    st.markdown(f"### Évolution du VRI et des variables météo — {ville_choisie}")

    col_g1, col_g2 = st.columns([2, 1])

    with col_g1:
        # VRI dans le temps
        fig_vri = go.Figure()
        fig_vri.add_trace(go.Scatter(
            x=df_ville["time"], y=df_ville["VRI"],
            mode="lines", name="VRI",
            line=dict(color="#2E75B6", width=1.5),
            fill="tozeroy", fillcolor="rgba(46,117,182,0.12)"
        ))
        fig_vri.add_hline(y=0.33, line_dash="dot", line_color="#FF9900",
                          annotation_text="Seuil modéré", annotation_position="right")
        fig_vri.add_hline(y=0.66, line_dash="dot", line_color="#C00000",
                          annotation_text="Seuil élevé", annotation_position="right")
        fig_vri.update_layout(
            title=f"VRI journalier — {ville_choisie}",
            xaxis_title="Date", yaxis_title="VRI",
            yaxis=dict(range=[0, 1]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_vri, use_container_width=True)

    with col_g2:
        # Distribution VRI par classe
        df_ville["Classe"] = df_ville["VRI"].apply(
            lambda v: "Faible" if v < 0.33 else "Modéré" if v < 0.66 else "Élevé"
        )
        dist = df_ville["Classe"].value_counts().reset_index()
        dist.columns = ["Classe", "Jours"]
        color_map = {"Faible": "#00B050", "Modéré": "#FF9900", "Élevé": "#C00000"}

        fig_pie = px.pie(dist, names="Classe", values="Jours",
                         color="Classe", color_discrete_map=color_map,
                         title="Répartition des niveaux de risque")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Graphiques météo mensuels
    st.markdown("#### Profil mensuel")
    df_ville["month"] = df_ville["time"].dt.month
    monthly = df_ville.groupby("month").agg(
        temp=("temperature_2m_mean", "mean"),
        precip=("precipitation_sum", "sum"),
        vri=("VRI", "mean")
    ).reset_index()
    monthly["month_name"] = monthly["month"].apply(get_month_name)

    fig_monthly = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Température moy. (°C)", "Précipitations (mm)", "VRI moyen")
    )
    fig_monthly.add_trace(go.Bar(
        x=monthly["month_name"], y=monthly["temp"],
        marker_color="#2E75B6", name="Temp"
    ), row=1, col=1)
    fig_monthly.add_trace(go.Bar(
        x=monthly["month_name"], y=monthly["precip"],
        marker_color="#70AD47", name="Précip"
    ), row=1, col=2)
    fig_monthly.add_trace(go.Bar(
        x=monthly["month_name"], y=monthly["vri"],
        marker_color=[color_map.get(
            "Faible" if v < 0.33 else "Modéré" if v < 0.66 else "Élevé", "#2E75B6"
        ) for v in monthly["vri"]],
        name="VRI"
    ), row=1, col=3)
    fig_monthly.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20),
        height=320
    )
    st.plotly_chart(fig_monthly, use_container_width=True)


# ============================
# TAB 3 — ALERTES
# ============================
with tab3:
    st.markdown("### État du risque vectoriel — toutes les villes")

    alertes = df.groupby(["city", "region"]).agg(
        temp=("temperature_2m_mean", "mean"),
        precip=("precipitation_sum", "mean"),
        precip_h=("precipitation_hours", "mean") if "precipitation_hours" in df.columns else ("precipitation_sum", "count")
    ).reset_index()
    alertes["VRI"] = alertes.apply(
        lambda r: compute_vri(r["temp"], r["precip"], r.get("precip_h", 0)), axis=1
    )
    alertes["Niveau"] = alertes["VRI"].apply(lambda v: vri_to_risk(v)[0])
    alertes = alertes.sort_values("VRI", ascending=False).reset_index(drop=True)

    # Filtres alertes
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filtre_niveau = st.selectbox("Filtrer par niveau",
                                     ["Tous", "ÉLEVÉ", "MODÉRÉ", "FAIBLE"])
    with col_f2:
        filtre_region = st.selectbox("Filtrer par région",
                                     ["Toutes"] + sorted(df["region"].unique()))

    alertes_filtre = alertes.copy()
    if filtre_niveau != "Tous":
        alertes_filtre = alertes_filtre[alertes_filtre["Niveau"] == filtre_niveau]
    if filtre_region != "Toutes":
        alertes_filtre = alertes_filtre[alertes_filtre["region"] == filtre_region]

    st.markdown(f"**{len(alertes_filtre)} villes affichées**")

    # Cartes alertes
    for _, row in alertes_filtre.iterrows():
        lbl, cls, color = vri_to_risk(row["VRI"])
        st.markdown(f"""
        <div class="alerte-card {cls}">
            <div class="alerte-left">
                <div>
                    <div class="alerte-ville">{row['city']}</div>
                    <div class="alerte-region">{row['region']}</div>
                </div>
            </div>
            <div class="alerte-right">
                <div class="alerte-vri">VRI : {row['VRI']:.2f}</div>
                <div class="alerte-niveau">{lbl}</div>
                <div class="alerte-details">Temp {row['temp']:.1f}°C · Précip {row['precip']:.1f}mm/j</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Heatmap région × mois
    st.markdown("#### Heatmap VRI moyen par région")
    df["month"] = df["time"].dt.month
    df["VRI_calc"] = df.apply(
        lambda r: compute_vri(r["temperature_2m_mean"], r["precipitation_sum"],
                              r.get("precipitation_hours", 0)), axis=1
    )
    heat_data = df.groupby(["region", "month"])["VRI_calc"].mean().reset_index()
    heat_pivot = heat_data.pivot(index="region", columns="month", values="VRI_calc")
    heat_pivot.columns = [get_month_name(m) for m in heat_pivot.columns]

    fig_heat = px.imshow(
        heat_pivot, color_continuous_scale="RdYlGn_r",
        title="VRI moyen mensuel par région",
        aspect="auto", zmin=0, zmax=1,
        labels=dict(color="VRI")
    )
    fig_heat.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20),
        height=380
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ============================
# TAB 4 — PRÉDICTION
# ============================
with tab4:
    st.markdown("### Prédiction du VRI dans 7 jours")

    if model is None:
        st.warning("Modèle non trouvé. Vérifiez que `models/lgb_global_t7.pkl` existe.")
    else:
        st.markdown("""
        <div class="pred-info">
        Ce module utilise le modèle <b>LightGBM global</b> entraîné sur les 42 villes du Cameroun
        pour prédire l'Indice de Risque Vectoriel à <b>t+7 jours</b>.
        </div>
        """, unsafe_allow_html=True)

        # ── Mode 1 : prédiction depuis df_feat (données réelles)
        if df_feat is not None and FEATURE_COLS is not None:
            st.markdown("#### Prédiction depuis les données historiques")

            villes_feat = sorted(df_feat["city"].unique())
            ville_pred = st.selectbox("Ville", villes_feat,
                                      index=villes_feat.index(ville_choisie)
                                      if ville_choisie in villes_feat else 0,
                                      key="ville_pred")

            df_ville_feat = df_feat[df_feat["city"] == ville_pred].sort_values("time")

            if not df_ville_feat.empty:
                last_row = df_ville_feat.iloc[-1]
                date_pred = last_row["time"]

                feat_available = [c for c in FEATURE_COLS if c in df_ville_feat.columns]
                X_last = df_ville_feat[feat_available].tail(1)

                # Compléter les features manquantes avec 0
                for col in FEATURE_COLS:
                    if col not in X_last.columns:
                        X_last[col] = 0.0
                X_last = X_last[FEATURE_COLS]

                pred_vri = float(np.clip(model.predict(X_last)[0], 0, 1))
                lbl_p, cls_p, color_p = vri_to_risk(pred_vri)

                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.markdown(f"""
                    <div class="pred-result {cls_p}">
                        <div class="pred-value">{pred_vri:.3f}</div>
                        <div class="pred-label">VRI prédit t+7</div>
                        <div class="pred-niveau">{lbl_p}</div>
                        <div class="pred-date">Pour le {(date_pred + timedelta(days=7)).strftime('%d/%m/%Y')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_pred2:
                    vri_actuel = float(last_row["VRI"]) if "VRI" in last_row else compute_vri(
                        last_row.get("temperature_2m_mean", 25),
                        last_row.get("precipitation_sum", 0),
                        last_row.get("precipitation_hours", 0)
                    )
                    delta = pred_vri - vri_actuel
                    tendance = "En hausse" if delta > 0.05 else "En baisse" if delta < -0.05 else "Stable"
                    st.markdown(f"""
                    <div class="pred-info-card">
                        <div class="pred-info-title">Tendance</div>
                        <div class="pred-info-value">{tendance}</div>
                        <div class="pred-info-sub">VRI actuel : {vri_actuel:.3f}<br>
                        Variation : {delta:+.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_pred3:
                    if lbl_p == "ÉLEVÉ":
                        recomm = "Déclencher protocole d'alerte. Campagne de démoustication immédiate."
                        bg = "#fff5f5"
                        border = "#C00000"
                    elif lbl_p == "MODÉRÉ":
                        recomm = "Surveillance renforcée. Préparer les équipes de terrain."
                        bg = "#fffbf0"
                        border = "#FF9900"
                    else:
                        recomm = "Situation normale. Maintenir la surveillance standard."
                        bg = "#f0fff4"
                        border = "#00B050"
                    st.markdown(f"""
                    <div class="pred-recomm" style="background:{bg};border-left:4px solid {border}">
                        <div class="pred-info-title">Recommandation</div>
                        <div class="pred-recomm-text">{recomm}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Courbe VRI des 30 derniers jours + prédiction
                st.markdown("#### Historique VRI + prédiction t+7")
                df_recent = df_ville_feat.tail(30).copy()
                df_recent["VRI_plot"] = df_recent.apply(
                    lambda r: compute_vri(
                        r.get("temperature_2m_mean", 25),
                        r.get("precipitation_sum", 0),
                        r.get("precipitation_hours", 0)
                    ), axis=1
                )

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=df_recent["time"], y=df_recent["VRI_plot"],
                    mode="lines+markers", name="VRI observé",
                    line=dict(color="#2E75B6", width=2),
                    marker=dict(size=4)
                ))
                # Point prédiction
                date_futur = date_pred + timedelta(days=7)
                fig_pred.add_trace(go.Scatter(
                    x=[date_futur], y=[pred_vri],
                    mode="markers", name="VRI prédit t+7",
                    marker=dict(color=color_p, size=14, symbol="star",
                                line=dict(color="white", width=2))
                ))
                # Ligne de connexion
                fig_pred.add_trace(go.Scatter(
                    x=[date_pred, date_futur],
                    y=[df_recent["VRI_plot"].iloc[-1], pred_vri],
                    mode="lines", name="",
                    line=dict(color=color_p, width=1.5, dash="dash"),
                    showlegend=False
                ))
                fig_pred.add_hline(y=0.33, line_dash="dot", line_color="#FF9900", opacity=0.5)
                fig_pred.add_hline(y=0.66, line_dash="dot", line_color="#C00000", opacity=0.5)
                fig_pred.add_vrect(
                    x0=date_pred, x1=date_futur,
                    fillcolor=color_p, opacity=0.07, line_width=0
                )
                fig_pred.update_layout(
                    title=f"VRI — {ville_pred} (30 derniers jours + prédiction t+7)",
                    xaxis_title="Date", yaxis_title="VRI",
                    yaxis=dict(range=[0, 1.05]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=40, b=20),
                    legend=dict(orientation="h", y=-0.15)
                )
                st.plotly_chart(fig_pred, use_container_width=True)

        # ── Mode 2 : saisie manuelle
        st.markdown("---")
        st.markdown("#### Prédiction manuelle (saisie des conditions météo)")

        with st.expander("Saisir les conditions météo manuellement", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                m_temp = st.number_input("Température moy. (°C)", value=27.0, step=0.5)
                m_precip = st.number_input("Précipitations (mm)", value=5.0, step=1.0)
            with c2:
                m_precip_h = st.number_input("Heures de pluie", value=3.0, step=0.5)
                m_et0 = st.number_input("ET0 FAO (mm)", value=4.0, step=0.5)
            with c3:
                m_vri_lag7 = st.number_input("VRI il y a 7 jours", value=0.45, min_value=0.0, max_value=1.0, step=0.01)
                m_dry = st.checkbox("Saison sèche ?", value=False)

            if st.button("Calculer la prédiction", type="primary"):
                # Construction d'un vecteur de features simplifié
                T_opt  = np.exp(-((m_temp - 25) ** 2) / 50)
                HR_est = min(100, max(0, 60 + (25 - m_temp) * 2))
                HR_opt = np.exp(-((HR_est - 70) ** 2) / 450)
                P_opt  = 1.0 if (m_precip > 5 and m_precip_h > 4) else 0.5
                is_dry = 1 if m_dry else 0

                vri_calc = compute_vri(m_temp, m_precip, m_precip_h, m_et0, is_dry)
                lbl_m, cls_m, color_m = vri_to_risk(vri_calc)

                st.markdown(f"""
                <div class="pred-result {cls_m}" style="max-width:300px;margin:auto">
                    <div class="pred-value">{vri_calc:.3f}</div>
                    <div class="pred-label">VRI calculé</div>
                    <div class="pred-niveau">{lbl_m}</div>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"T_opt = {T_opt:.3f} · HR_opt = {HR_opt:.3f} · P_opt = {P_opt:.2f} · (1−S) = {1-is_dry}")