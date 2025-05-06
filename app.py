# eco_conso_app.py
import os
from dotenv import load_dotenv

import streamlit as st
from mistralai import Mistral          # ✅ nouvel import conseillé par le quick-start

# ------------------------------------------------------------------ #
# Initialisation
# ------------------------------------------------------------------ #
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")     # ← doit exister dans l'env

if not api_key:
    st.error("❌ Variable d’environnement MISTRAL_API_KEY manquante.")
    st.stop()

client = Mistral(api_key=api_key)          # client réutilisable
MODEL_ID = "mistral-small-latest"          # alias stable recommandé

# ------------------------------------------------------------------ #
# Appel API
# ------------------------------------------------------------------ #
def get_mistral_response(answers: dict) -> str:
    """Retourne l’analyse + 3 conseils concrets ou une chaîne vide en cas d’erreur."""
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
                {"role": "user",   "content": user_prompt},
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

with st.form("eco_form"):
    chauffage = st.selectbox("1. Température du chauffage ?", ["≤ 19 °C", "20-21 °C", "≥ 22 °C"])
    veille     = st.selectbox("2. Appareils laissés en veille ?", ["Jamais", "Parfois", "Toujours"])
    eclairage  = st.selectbox("3. Type d’éclairage ?", ["LED", "Basse consommation", "Classique"])
    transport  = st.selectbox("4. Transport principal ?", ["Vélo / marche", "Transports en commun", "Voiture"])
    recyclage  = st.selectbox("5. Tu recycles ?", ["Oui", "Parfois", "Non"])
    submitted  = st.form_submit_button("Analyser")

if submitted:
    user_data = {
        "chauffage": chauffage,
        "veille": veille,
        "eclairage": eclairage,
        "transport": transport,
        "recyclage": recyclage,
    }

    st.subheader("🔍 Analyse Mistral en cours…")
    with st.spinner("Génération des conseils écologiques…"):
        result = get_mistral_response(user_data)

    if result:
        st.success("✅ Résultat généré par Mistral :")
        st.markdown(result)