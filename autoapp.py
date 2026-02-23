#Voici le code final de l'application Streamlit, sans dépendance aux fichiers .pkl. L'entraînement du modèle est intégré et utilise un petit jeu de données synthétique pour fonctionner immédiatement.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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
    Cette application utilise un modèle **Random Forest** entraîné sur un échantillon de données.
    Aucun fichier externe n'est nécessaire.
    """
)

# -----------------------------------------------------------------------------
# Entraînement du modèle sur données synthétiques (remplaçable par un vrai fichier)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    np.random.seed(42)
    n_samples = 1000

    # Simulation de variables numériques
    age = np.random.randint(18, 95, n_samples)
    balance = np.random.normal(1000, 500, n_samples).clip(0)
    day = np.random.randint(1, 32, n_samples)
    campaign = np.random.randint(1, 10, n_samples)
    pdays = np.random.choice([-1, 1, 2, 3, 4, 5], n_samples, p=[0.8, 0.05, 0.05, 0.03, 0.04, 0.03])
    previous = np.random.randint(0, 5, n_samples)

    # Simulation de variables catégorielles
    job = np.random.choice(['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                            'retired', 'self-employed', 'services', 'student', 'technician',
                            'unemployed', 'unknown'], n_samples)
    marital = np.random.choice(['divorced', 'married', 'single', 'unknown'], n_samples,
                               p=[0.1, 0.6, 0.25, 0.05])
    education = np.random.choice(['primary', 'secondary', 'tertiary', 'unknown'], n_samples,
                                 p=[0.2, 0.5, 0.2, 0.1])
    default = np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.95, 0.02, 0.03])
    housing = np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.5, 0.45, 0.05])
    loan = np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.8, 0.15, 0.05])
    contact = np.random.choice(['cellular', 'telephone', 'unknown'], n_samples, p=[0.7, 0.2, 0.1])
    month = np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples)
    poutcome = np.random.choice(['failure', 'other', 'success', 'unknown'], n_samples, p=[0.1, 0.2, 0.1, 0.6])
    y = np.random.choice(['yes', 'no'], n_samples, p=[0.12, 0.88])

    df = pd.DataFrame({
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome, 'y': y
    })

    # Préparation des données
    X = df.drop('y', axis=1)
    y = df['y']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    model.fit(X, y_encoded)

    return model, le

model, target_encoder = load_and_train_model()

# -----------------------------------------------------------------------------
# Options pour les listes déroulantes
# -----------------------------------------------------------------------------
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
               'retired', 'self-employed', 'services', 'student', 'technician',
               'unemployed', 'unknown']
marital_options = ['divorced', 'married', 'single', 'unknown']
education_options = ['primary', 'secondary', 'tertiary', 'unknown']
default_options = ['no', 'yes', 'unknown']
housing_options = ['no', 'yes', 'unknown']
loan_options = ['no', 'yes', 'unknown']
contact_options = ['cellular', 'telephone', 'unknown']
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcome_options = ['failure', 'other', 'success', 'unknown']

# -----------------------------------------------------------------------------
# Barre latérale - saisie utilisateur
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Caractéristiques du client")

with st.sidebar.expander("👤 Données personnelles", expanded=True):
    age = st.number_input("Âge", min_value=18, max_value=100, value=40, step=1)
    job = st.selectbox("Emploi", job_options)
    marital = st.selectbox("Situation familiale", marital_options)
    education = st.selectbox("Niveau d'éducation", education_options)

with st.sidebar.expander("Situation financière", expanded=True):
    default = st.selectbox("Crédit en défaut ?", default_options)
    balance = st.number_input("Solde annuel moyen (€)", value=1000, step=100)
    housing = st.selectbox("Prêt immobilier ?", housing_options)
    loan = st.selectbox("Prêt personnel ?", loan_options)

with st.sidebar.expander("Dernier contact", expanded=True):
    contact = st.selectbox("Type de contact", contact_options)
    day = st.number_input("Jour du mois", min_value=1, max_value=31, value=15, step=1)
    month = st.selectbox("Mois", month_options)

with st.sidebar.expander("Campagne précédente", expanded=True):
    campaign = st.number_input("Nombre de contacts durant cette campagne", min_value=1, value=1, step=1)
    pdays = st.number_input("Jours depuis le dernier contact (-1 si jamais contacté)", value=-1, step=1)
    previous = st.number_input("Nombre de contacts avant cette campagne", min_value=0, value=0, step=1)
    poutcome = st.selectbox("Résultat de la campagne précédente", poutcome_options)

predict_btn = st.sidebar.button("Prédire", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Page principale - affichage des résultats
# -----------------------------------------------------------------------------
if predict_btn:
    input_dict = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }
    input_df = pd.DataFrame([input_dict])

    with st.spinner("Calcul en cours..."):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

    classe = target_encoder.inverse_transform([pred])[0]

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

