# 🌱 EcoConso – Application Streamlit avec Mistral AI

Cette application permet aux utilisateurs de répondre à un questionnaire écologique et de recevoir des conseils
personnalisés générés par l'IA de Mistral.

## 📦 Prérequis

- Python 3.9 ou supérieur

### 📦 Dépendances

- Streamlit
- Mistral AI
- Python-dotenv
- Pandas
- NumPy
- Matplotlib

## 🚀 Installation

1. Clone le dépôt :
   ```bash
   git clone https://github.com/ton-utilisateur/eco-coach.git
   cd eco-coach
    ```
2. Installe les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Crée un fichier `.env` à la racine du projet et ajoute ta clé API Mistral :
   ```bash
    MISTRAL_API_KEY=ta_cle_api
    ```
4. Lance l'application :
   ```bash
   streamlit run app.py
   ```