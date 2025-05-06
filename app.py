import os
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mistralai import Mistral

# ------------------------------------------------------------------ #
# Initialisation Mistral
# ------------------------------------------------------------------ #
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("❌ Variable d’environnement MISTRAL_API_KEY manquante.")
    st.stop()

client = Mistral(api_key=api_key)
MODEL_ID = "mistral-small-latest"

# ------------------------------------------------------------------ #
# Barème éco (1 = très écolo ; 3 = impact élevé)
# ------------------------------------------------------------------ #
SCORE_MAP = {
    "chauffage": {"≤ 19 °C": 1, "20-21 °C": 2, "≥ 22 °C": 3},
    "veille": {"Jamais": 1, "Parfois": 2, "Toujours": 3},
    "eclairage": {"LED": 1, "Basse consommation": 2, "Classique": 3},
    "transport": {"Vélo / marche": 1, "Transports en commun": 2, "Voiture": 3},
    "recyclage": {"Oui": 1, "Parfois": 2, "Non": 3},
}


def calc_scores(answers: dict) -> dict:
    return {k: SCORE_MAP[k][answers[k]] for k in answers}


# ------------------------------------------------------------------ #
# Appel LLM
# ------------------------------------------------------------------ #
def get_mistral_response(answers: dict) -> str:
    user_prompt = (
        "Tu es un expert en écologie. Voici les habitudes de consommation énergétique "
        "d’un·e utilisateur·ice :\n"
        f"- Chauffage : {answers['chauffage']}\n"
        f"- Veille : {answers['veille']}\n"
        f"- Éclairage : {answers['eclairage']}\n"
        f"- Transport : {answers['transport']}\n"
        f"- Recyclage : {answers['recyclage']}\n\n"
        "Fournis une courte analyse de son profil écologique et donne 3 conseils concrets "
        "pour améliorer son comportement."
    )
    try:
        resp = client.chat.complete(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "Tu es un assistant écologique bienveillant."},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        st.error(f"❌ Erreur API Mistral : {exc}")
        return ""


# ------------------------------------------------------------------ #
# Interface Streamlit
# ------------------------------------------------------------------ #
st.set_page_config(page_title="ÉcoConso Mistral", layout="centered")
st.title("🌱 ÉcoConso – Diagnostic écologique (avec Mistral)")
st.caption("Réponds aux questions pour analyser ton profil et recevoir des conseils personnalisés.")

PH = "Sélectionner une option…"

with st.form("eco_form"):
    chauffage = st.selectbox("1. Température du chauffage ?", [PH, "≤ 19 °C", "20-21 °C", "≥ 22 °C"])
    veille = st.selectbox("2. Appareils laissés en veille ?", [PH, "Jamais", "Parfois", "Toujours"])
    eclairage = st.selectbox("3. Type d’éclairage ?", [PH, "LED", "Basse consommation", "Classique"])
    transport = st.selectbox("4. Transport principal ?", [PH, "Vélo / marche", "Transports en commun", "Voiture"])
    recyclage = st.selectbox("5. Tu recycles ?", [PH, "Oui", "Parfois", "Non"])
    submitted = st.form_submit_button("Analyser")

if submitted:
    # --- Validation simple --------------------------------------------------
    selections = {
        "chauffage": chauffage,
        "veille": veille,
        "eclairage": eclairage,
        "transport": transport,
        "recyclage": recyclage,
    }
    if any(v == PH for v in selections.values()):
        st.warning("⚠️ Merci de sélectionner une option pour **toutes** les questions.")
        st.stop()

    # --- 1) Conseils IA ------------------------------------------------------
    st.subheader("🔍 Analyse Mistral en cours…")
    with st.spinner("Génération des conseils écologiques…"):
        result = get_mistral_response(selections)

    if result:
        st.success("✅ Résultat généré par Mistral :")
        st.markdown(result)

    # --- 2) Scoring + bar chart ---------------------------------------------
    scores = calc_scores(selections)
    score_df = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])
    st.subheader("📊 Tes scores par catégorie (1 = écolo, 3 = impact élevé)")
    st.bar_chart(score_df, use_container_width=True)

    # --- 3) Radar global -----------------------------------------------------
    with st.expander("📌 Voir un radar global"):
        categories = list(scores.keys())
        values = list(scores.values())
        values.append(values[0])
        angles = np.linspace(0, 2 * np.pi, len(categories) + 1)

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 3)
        ax.set_title("Radar - Profil écologique global", pad=20)

        st.pyplot(fig)
