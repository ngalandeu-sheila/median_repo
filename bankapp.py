import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os

# -----------------------------------------------------------------------------
# Configuration de la page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction Souscription Dépôt",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Prédiction de souscription à un dépôt à terme")
st.markdown(
    """
    Cette application utilise un modèle **Random Forest** entraîné sur le fichier `bank.csv`.
    Renseignez les caractéristiques du client dans la barre latérale pour obtenir une prédiction.
    """
)

# -----------------------------------------------------------------------------
# Chargement et préparation des données (mise en cache)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # Chargement
    df = pd.read_csv('bank.csv', sep=';')

    st.subheader("Aperçu du jeu de données")
    st.dataframe(df.head())

    # Suppression de la colonne 'duration' (non réaliste pour la prédiction)
    df = df.drop(columns=['duration'])

    # Séparation features / target
    X = df.drop('y', axis=1)
    y = df['y']

    # Encodage de la cible
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Identification des colonnes numériques et catégorielles
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Création du préprocesseur
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    # Pipeline final avec Random Forest
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])

    # Entraînement sur toutes les données (pas de split, on veut un modèle prêt à l'emploi)
    model.fit(X, y_encoded)

    return model, le, numerical_cols, categorical_cols

model, target_encoder, numerical_cols, categorical_cols = load_and_train_model()

# -----------------------------------------------------------------------------
# Définition des options pour les listes déroulantes
# -----------------------------------------------------------------------------
# On récupère les modalités directement depuis les données originales (via le cache)
@st.cache_data
def get_categories():
    df = pd.read_csv('bank.csv', sep=';')
    df = df.drop(columns=['duration'])
    cat_options = {}
    for col in categorical_cols:
        cat_options[col] = sorted(df[col].unique())
    return cat_options

cat_options = get_categories()

# -----------------------------------------------------------------------------
# Barre latérale - Saisie des caractéristiques
# -----------------------------------------------------------------------------
st.sidebar.header("Caractéristiques du client")

with st.sidebar.expander("Données personnelles", expanded=True):
    age = st.number_input("Âge", min_value=18, max_value=100, value=40, step=1)
    job = st.selectbox("Emploi", cat_options['job'])
    marital = st.selectbox("Situation familiale", cat_options['marital'])
    education = st.selectbox("Niveau d'éducation", cat_options['education'])

with st.sidebar.expander("Situation financière", expanded=True):
    default = st.selectbox("Crédit en défaut ?", cat_options['default'])
    balance = st.number_input("Solde annuel moyen (€)", value=1000, step=100)
    housing = st.selectbox("Prêt immobilier ?", cat_options['housing'])
    loan = st.selectbox("Prêt personnel ?", cat_options['loan'])

with st.sidebar.expander("Dernier contact", expanded=True):
    contact = st.selectbox("Type de contact", cat_options['contact'])
    day = st.number_input("Jour du mois", min_value=1, max_value=31, value=15, step=1)
    month = st.selectbox("Mois", cat_options['month'])

with st.sidebar.expander("Campagne précédente", expanded=True):
    campaign = st.number_input("Nombre de contacts durant cette campagne", min_value=1, value=1, step=1)
    pdays = st.number_input("Jours depuis le dernier contact (-1 si jamais contacté)", value=-1, step=1)
    previous = st.number_input("Nombre de contacts avant cette campagne", min_value=0, value=0, step=1)
    poutcome = st.selectbox("Résultat de la campagne précédente", cat_options['poutcome'])

predict_btn = st.sidebar.button("Prédire", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Page principale - Résultats
# -----------------------------------------------------------------------------
if predict_btn:
    # Construction du DataFrame d'entrée
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    input_df = pd.DataFrame([input_dict])

    with st.spinner("Calcul en cours..."):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

    classe = target_encoder.inverse_transform([prediction])[0]

    st.subheader("Résultat de la prédiction")
    col1, col2, col3 = st.columns(3)

    with col1:
        if classe == 'yes':
            st.success("### ✅ OUI")
            st.markdown("Le client est susceptible de souscrire.")
        else:
            st.error("### ❌ NON")
            st.markdown("Le client ne souscrira probablement pas.")

    with col2:
        st.metric("Probabilité de souscription", f"{proba[1]:.2%}")

    with col3:
        st.metric("Classe prédite", classe.upper())

    st.subheader("Probabilités par classe")
    prob_df = pd.DataFrame({
        'Classe': ['Non', 'Oui'],
        'Probabilité': proba
    })
    st.bar_chart(prob_df.set_index('Classe'))

    with st.expander("Voir les caractéristiques saisies"):
        st.dataframe(input_df, use_container_width=True)

else:
    st.info("👈 Remplissez les informations du client dans la barre latérale puis cliquez sur **Prédire**.")
    # Optionnel : afficher un aperçu des données
    if st.checkbox("Afficher un aperçu des données d'entraînement"):
        df_preview = pd.read_csv('bank.csv', sep=';').drop(columns=['duration'])
        st.dataframe(df_preview.head())