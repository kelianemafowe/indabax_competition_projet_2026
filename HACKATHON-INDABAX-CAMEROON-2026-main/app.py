import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
import joblib

# ============================
# CONFIG
# ============================
st.set_page_config(
    page_title="Dashboard Qualité de l'Air",
    layout="wide"
)

# ============================
# CSS
# ============================
def load_css():
    with open("part_css.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ============================
# HEADER
# ============================
st.markdown("""
    <div class=header> 
    <h1>Dashboard Qualité de l'Air — Cameroun</h1>
    <p>Surveillance et analyse des conditions atmosphériques</p>
    </div>
""", unsafe_allow_html=True)

# ============================
# DATA
# ============================
@st.cache_data
def load_data():
    df = pd.read_excel("data/Dataset_complet_Meteo.xlsx")
    df["time"] = pd.to_datetime(df["time"])
    df["temperature_2m_mean"] = pd.to_numeric(df["temperature_2m_mean"], errors="coerce")
    df["precipitation_sum"] = pd.to_numeric(df["precipitation_sum"], errors="coerce")
    df["wind_speed_10m_max"] = pd.to_numeric(df["wind_speed_10m_max"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df

@st.cache_resource
def load_model():
    return joblib.load("data/model_air.pkl")

df = load_data()
model = load_model()

# ============================
# ALERTE
# ============================
def get_alerte(tmp):
    if tmp < 25:
        return "BON", "alerte-bon", "Air sain", "Conditions normales"
    elif tmp < 30:
        return "MODERE", "alerte-modere", "Air modéré", "Surveillance recommandée"
    else:
        return "DANGEREUX", "alerte-dangereux", "Air dégradé", "Limiter les activités extérieures"

# ============================
# SIDEBAR
# ============================
st.sidebar.header("Filtres")

date_min = df["time"].min().date()
date_max = df["time"].max().date()

dates = st.sidebar.date_input("Période", (date_min, date_max))

ville_choisie = st.sidebar.selectbox("Ville", sorted(df["city"].unique()))

date_debut, date_fin = dates

df_ville = df[
    (df["city"] == ville_choisie) &
    (df["time"].dt.date >= date_debut) &
    (df["time"].dt.date <= date_fin)
]

if df_ville.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

# ============================
# KPI
# ============================
col1, col2, col3, col4 = st.columns(4)

temp_moy = df_ville["temperature_2m_mean"].mean()
pluie = df_ville["precipitation_sum"].sum()
vent = df_ville["wind_speed_10m_max"].max()

label, _, titre, desc = get_alerte(temp_moy)

col1.metric("Température moyenne", f"{temp_moy:.1f} °C")
col2.metric("Précipitations", f"{pluie:.0f} mm")
col3.metric("Vent max", f"{vent:.1f} km/h")
col4.metric("Niveau", label)

# ============================
# ALERTES GLOBAL
# ============================
alertes = df.groupby("city")["temperature_2m_mean"].mean().reset_index()
alertes.columns = ["Ville", "Temp_moy"]

# ============================
# TABS
# ============================
tab1, tab2, tab3, tab4 = st.tabs(["Carte", "Graphiques", "Alertes", "Prédiction"])

# ============================
# CARTE
# ============================
with tab1:
    m = folium.Map(location=[3.8, 11.5], zoom_start=6)

    villes = df.groupby("city").agg(
        lat=("latitude", "first"),
        lon=("longitude", "first")
    ).reset_index()

    for _, row in villes.iterrows():
        temp = df[df["city"] == row["city"]]["temperature_2m_mean"].mean()

        color = "green" if temp < 25 else "orange" if temp < 30 else "red"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color=color,
            fill=True,
            popup=f"{row['city']} - {temp:.1f}°C"
        ).add_to(m)

    st_folium(m, width=700, height=500)

# ============================
# GRAPHIQUES
# ============================
with tab2:
    df_ville["month"] = df_ville["time"].dt.month

    temp_mois = df_ville.groupby("month")["temperature_2m_mean"].mean().reset_index()

    fig = px.bar(temp_mois, x="month", y="temperature_2m_mean",
                 title="Température moyenne mensuelle")

    st.plotly_chart(fig, use_container_width=True)

# ============================
# ALERTES
# ============================
with tab3:
    for _, row in alertes.iterrows():
        label, classe, titre, desc = get_alerte(row["Temp_moy"])

        st.markdown(f"""
            <div class="{classe}">
                <strong>{row['Ville']} — {titre}</strong><br>
                Température moyenne : {row['Temp_moy']:.1f} °C
            </div>
        """, unsafe_allow_html=True)

# ============================
# PREDICTION
# ============================
with tab4:
    st.subheader("Prédiction")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Température", value=28.0)
        precipitation = st.number_input("Précipitations", value=5.0)

    with col2:
        wind_speed = st.number_input("Vent", value=10.0)
        latitude = st.number_input("Latitude", value=3.8)
        longitude = st.number_input("Longitude", value=11.5)

    if st.button("Exécuter"):
        try:
            X = [[temperature, precipitation, wind_speed, latitude, longitude]]
            pred = model.predict(X)[0]

            label, _, titre, desc = get_alerte(pred)

            st.success(f"Résultat : {pred:.2f} °C")
            st.write(titre)
            st.write(desc)

        except Exception as e:
            st.error(e)