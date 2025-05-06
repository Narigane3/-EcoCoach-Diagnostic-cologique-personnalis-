import os
from dotenv import load_dotenv

import streamlit as st
import pandas as pd              # ← NEW
import numpy as np               # ← NEW
import matplotlib.pyplot as plt  # ← NEW

from mistralai import Mistral     # SDK v2+

# ------------------------------------------------------------------ #
# Initialisation Mistral
# ------------------------------------------------------------------ #
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("❌ Variable d’environnement MISTRAL_API_KEY manquante.")
    st.stop()

client = Mistral(api_key=api_key)
MODEL_ID = "mistral-small-latest"   # alias stable recommandé   [oai_citation:0‡Mistral AI Documentation](https://docs.mistral.ai/getting-started/clients/)

# ------------------------------------------------------------------ #
# Barème éco (1 = très écolo ; 3 = impact élevé)
# ------------------------------------------------------------------ #
SCORE_MAP = {
    "chauffage": {"≤ 19 °C": 1, "20-21 °C": 2, "≥ 22 °C": 3},
    "veille":     {"Jamais": 1, "Parfois": 2, "Toujours": 3},
    "eclairage":  {"LED": 1, "Basse consommation": 2, "Classique": 3},
    "transport":  {"Vélo / marche": 1, "Transports en commun": 2, "Voiture": 3},
    "recyclage":  {"Oui": 1, "Parfois": 2, "Non": 3},
}

def calc_scores(answers: dict) -> dict:
    """Transforme les réponses en scores numériques dict(cat -> int)."""
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
st.set_page_config(page_title="EcoCoach Mistral", layout="centered")
st.title("🌱 EcoCoach – Diagnostic écologique (avec Mistral)")
st.caption("Réponds aux questions pour analyser ton profil et recevoir des conseils personnalisés.")

with st.form("eco_form"):
    chauffage = st.selectbox("1. Température du chauffage ?", ["≤ 19 °C", "20-21 °C", "≥ 22 °C"])
    veille     = st.selectbox("2. Appareils laissés en veille ?", ["Jamais", "Parfois", "Toujours"])
    eclairage  = st.selectbox("3. Type d’éclairage ?", ["LED", "Basse consommation", "Classique"])
    transport  = st.selectbox("4. Transport principal ?", ["Vélo / marche", "Transports en commun", "Voiture"])
    recyclage  = st.selectbox("5. Tu recycles ?", ["Oui", "Parfois", "Non"])
    submitted  = st.form_submit_button("Analyser")

# ------------------------------------------------------------------ #
# Résultats + graphiques
# ------------------------------------------------------------------ #
if submitted:
    user_data = {
        "chauffage": chauffage,
        "veille": veille,
        "eclairage": eclairage,
        "transport": transport,
        "recyclage": recyclage,
    }

    # --- 1) Conseils IA
    st.subheader("🔍 Analyse Mistral en cours…")
    with st.spinner("Génération des conseils écologiques…"):
        result = get_mistral_response(user_data)

    if result:
        st.success("✅ Résultat généré par Mistral :")
        st.markdown(result)

    # --- 2) Scoring + bar chart
    scores = calc_scores(user_data)
    score_df = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])
    st.subheader("📊 Tes scores par catégorie (1 = écolo, 3 = impact élevé)")
    st.bar_chart(score_df, use_container_width=True)   # streamlit sugar API   [oai_citation:1‡Streamlit Docs](https://docs.streamlit.io/develop/api-reference/charts/st.bar_chart?utm_source=chatgpt.com)

    # --- 3) Radar global (optionnel)
    with st.expander("📌 Voir un radar global"):
        categories = list(scores.keys())
        values = list(scores.values())
        values.append(values[0])            # boucle pour fermer le polygone
        angles = np.linspace(0, 2*np.pi, len(categories) + 1)

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 3)
        ax.set_title("Radar - Profil écologique global", pad=20)

        st.pyplot(fig)