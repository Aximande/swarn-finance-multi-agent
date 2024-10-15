import streamlit as st
import openai
import logging
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import plotly.express as px
from collections import defaultdict
import pickle
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()  # Charge les variables d'environnement depuis .env si présent

# Fonction pour masquer la clé API
def mask_api_key(api_key):
    if api_key:
        return api_key[:6] + "*" * (len(api_key) - 6)
    return ""

# Récupération de la clé API
api_key = os.getenv("OPENAI_API_KEY", "")

# Interface utilisateur pour la clé API
st.sidebar.title("Configuration OpenAI")
user_api_key = st.sidebar.text_input("Entrez votre clé API OpenAI", value=mask_api_key(api_key), type="password")

if user_api_key:
    os.environ["OPENAI_API_KEY"] = user_api_key
    st.sidebar.success("Clé API configurée avec succès!")
else:
    st.sidebar.warning("Veuillez entrer une clé API OpenAI valide.")

# Vérifiez si la clé API est définie avant de continuer
if not os.getenv("OPENAI_API_KEY"):
    st.error("La clé API OpenAI n'est pas définie. Veuillez la configurer dans la barre latérale.")
    st.stop()

logging.basicConfig(level=logging.DEBUG)

# Cette ligne doit être la première commande Streamlit
st.set_page_config(page_title="Assistant Financier IA Qonto", layout="wide", initial_sidebar_state="collapsed")

# Appliquer un style personnalisé
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #6200ee; color: white;}
    .stHeader {background-color: white; padding: 20px;}
    .results-section {background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

from agents import financial_planning_orchestrator_agent, document_analysis_agent, swarm_client, analyze_financial_image, analyze_spreadsheet, visualize_and_optimize_data
from utils import pretty_print_messages

# En-tête avec logos
st.image("swarm-logo.png", width=600)


# Titre et sous-titre centrés
st.markdown("<h1 style='text-align: center;'>Swarm Finance Multi-Agent</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Analysez et optimisez vos finances d'entreprise, comme Qonto</h3>", unsafe_allow_html=True)

st.markdown("---")

# Barre latérale
with st.sidebar:
    st.header("Options")
    business_type = st.selectbox(
        "Type d'entreprise",
        ["Auto-entrepreneur", "Entreprise avec équipes"]
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de téléchargement de fichier
uploaded_file = st.file_uploader("Glissez-déposez votre document financier (.csv, .xls, .xlsx) ou capture d'écran (.png, .jpg, .jpeg)",
                                 type=["csv", "xls", "xlsx", "png", "jpg", "jpeg"])

# Initialiser analysis
analysis = ""

# Initialisation des variables
analysis_result = None

# Ajoutez ces nouvelles fonctions pour l'analyse Qonto
def preprocess_qonto_data(df):
    possible_mappings = [
        {
            'Operation date (UTC)': 'Date',
            'Total amount (incl. VAT)': 'Montant',
            'Currency': 'Devise',
            'Counterparty name': 'Bénéficiaire',
            'Payment method': 'Méthode_Paiement',
            'Initiator': 'Initiateur',
            'Team': 'Équipe',
            'Category': 'Catégorie'
        },
        {
            'Settlement date (UTC)': 'Date',
            'Total amount (incl. VAT)': 'Montant',
            'Counterparty name': 'Bénéficiaire'
        }
    ]

    best_mapping = max(possible_mappings, key=lambda m: len(set(m.keys()) & set(df.columns)))
    df = df.rename(columns={k: v for k, v in best_mapping.items() if k in df.columns})

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

    if 'Montant' in df.columns:
        df['Montant'] = pd.to_numeric(df['Montant'], errors='coerce')

    return df

def analyze_qonto_data(df, business_type):
    analysis = {
        "total_transactions": len(df),
        "total_debit": df['Montant'][df['Montant'] < 0].sum() if 'Montant' in df.columns else 0,
        "total_credit": df['Montant'][df['Montant'] > 0].sum() if 'Montant' in df.columns else 0,
        "balance": df['Balance'].iloc[-1] if 'Balance' in df.columns else None,
        "currency": df['Devise'].iloc[0] if 'Devise' in df.columns else "EUR",
        "date_range": f"Du {df['Date'].min()} au {df['Date'].max()}" if 'Date' in df.columns else "Non spécifié",
    }

    if 'Bénéficiaire' in df.columns and 'Montant' in df.columns:
        analysis["top_expenses"] = df[df['Montant'] < 0].groupby('Bénéficiaire')['Montant'].sum().abs().nlargest(5).to_dict()
        analysis["top_income"] = df[df['Montant'] > 0].groupby('Bénéficiaire')['Montant'].sum().nlargest(5).to_dict()

    if business_type == "Auto-entrepreneur":
        if 'Catégorie' in df.columns and 'Montant' in df.columns:
            analysis["expenses_by_category"] = df[df['Montant'] < 0].groupby('Catégorie')['Montant'].sum().to_dict()
    else:
        if 'Équipe' in df.columns and 'Montant' in df.columns:
            analysis["expenses_by_team"] = df[df['Montant'] < 0].groupby('Équipe')['Montant'].sum().to_dict()

    if 'Méthode_Paiement' in df.columns and 'Montant' in df.columns:
        analysis["expenses_by_payment_method"] = df[df['Montant'] < 0].groupby('Méthode_Paiement')['Montant'].sum().to_dict()

    if 'Initiateur' in df.columns and 'Montant' in df.columns:
        analysis["top_initiators"] = df[df['Montant'] < 0].groupby('Initiateur')['Montant'].sum().nlargest(5).to_dict()

    recurring_df = identify_recurring_expenses(df)
    analysis['recurring_expenses'] = recurring_df.to_dict('records')

    return analysis

def generate_insights(analysis):
    insights = [
        f"Total des transactions : {analysis['total_transactions']}",
        f"Total des dépenses : {analysis['total_debit']:.2f} {analysis['currency']}",
        f"Total des revenus : {analysis['total_credit']:.2f} {analysis['currency']}",
        f"Solde actuel : {analysis['balance']:.2f} {analysis['currency']}" if analysis['balance'] else "Solde non disponible",
        f"Période analysée : {analysis['date_range']}",
        "\nTop 5 des dépenses par bénéficiaire :",
        *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['top_expenses'].items()],
        "\nTop 5 des revenus par source :",
        *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['top_income'].items()],
    ]

    if 'expenses_by_category' in analysis:
        insights.extend([
            "\nDépenses par catégorie :",
            *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['expenses_by_category'].items()],
        ])

    if 'expenses_by_team' in analysis:
        insights.extend([
            "\nDépenses par équipe :",
            *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['expenses_by_team'].items()],
        ])

    if 'expenses_by_payment_method' in analysis:
        insights.extend([
            "\nDépenses par méthode de paiement :",
            *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['expenses_by_payment_method'].items()],
        ])

    if 'top_initiators' in analysis:
        insights.extend([
            "\nTop 5 des initiateurs de dépenses :",
            *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['top_initiators'].items()],
        ])

    if 'recurring_expenses' in analysis:
        insights.extend([
            "\nDépenses récurrentes potentielles (abonnements) :",
            *[f"- {expense['Bénéficiaire']}: {expense['Montant']:.2f} {analysis['currency']} (Frquence: {expense['Fréquence']})"
              for expense in analysis['recurring_expenses'][:5]]  # Limitons à 5 pour la lisibilité
        ])

    return "\n".join(insights)

def identify_recurring_expenses(df, threshold=2, min_amount=1):
    """
    Identifie les dépenses récurrentes potentielles.

    :param df: DataFrame contenant les transactions
    :param threshold: Nombre minimum d'occurrences pour considérer une dépense comme récurrente
    :param min_amount: Montant minimum pour considérer une dépense
    :return: DataFrame des dépenses récurrentes
    """
    recurring = defaultdict(list)

    for _, row in df[df['Montant'] < -min_amount].iterrows():
        key = (row['Bénéficiaire'], abs(round(row['Montant'], 2)))
        recurring[key].append(row['Date'])

    recurring_expenses = []
    for (beneficiary, amount), dates in recurring.items():
        if len(dates) >= threshold:
            recurring_expenses.append({
                'Bénéficiaire': beneficiary,
                'Montant': amount,
                'Fréquence': len(dates),
                'Première occurrence': min(dates),
                'Dernière occurrence': max(dates)
            })

    return pd.DataFrame(recurring_expenses).sort_values('Montant', ascending=False)

def analyze_recurring_expenses(df):
    recurring_df = identify_recurring_expenses(df)
    total_recurring = recurring_df['Montant'].sum() * recurring_df['Fréquence'].mean()

    fig = px.bar(recurring_df, x='Bénéficiaire', y='Montant',
                 title='Dépenses récurrentes potentielles',
                 labels={'Bénéficiaire': 'Bénéficiaire', 'Montant': 'Montant (€)'},
                 hover_data=['Fréquence', 'Première occurrence', 'Dernière occurrence'])

    return recurring_df, total_recurring, fig

def save_analysis_history(analysis_result):
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    st.session_state.analysis_history.append(analysis_result)
    with open('analysis_history.pkl', 'wb') as f:
        pickle.dump(st.session_state.analysis_history, f)

def load_analysis_history():
    try:
        with open('analysis_history.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def generate_recommendations(current_analysis, history):
    recommendations = []
    if history:
        avg_expenses = sum(h['total_debit'] for h in history) / len(history)
        if current_analysis['total_debit'] > avg_expenses:
            recommendations.append("Vos dpenses sont supérieures à la moyenne historique. Envisagez de revoir votre budget.")
        # ... autres recommandations basées sur l'historique ...
    return recommendations

def create_expense_trend_chart(df, key, aggregation='D'):
    if df is None or df.empty or 'Montant' not in df.columns or 'Date' not in df.columns:
        return None

    df_expenses = df[df['Montant'] < 0].copy()
    if df_expenses.empty:
        return None

    df_expenses['Montant'] = df_expenses['Montant'].abs()
    df_expenses = df_expenses.groupby(df_expenses['Date'].dt.to_period(aggregation))['Montant'].sum().reset_index()
    df_expenses['Date'] = df_expenses['Date'].dt.to_timestamp()

    fig = px.line(df_expenses, x='Date', y='Montant', title=f'Tendance des dépenses ({aggregation})')
    return fig

@st.cache_data
def load_and_preprocess_data(file):
    try:
        if file.type.startswith('image'):
            return None  # Les images ne sont pas traitées comme des DataFrames
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding_errors='replace')
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            raise ValueError("Format de fichier non supporté")

        if df.empty:
            raise ValueError("Le fichier est vide")

        return preprocess_qonto_data(df)
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement du fichier : {str(e)}")
        return None

def process_qonto_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        raise ValueError("Format de fichier non supporté")

    df = preprocess_qonto_data(df)
    analysis_result = analyze_qonto_data(df, business_type)
    insights = generate_insights(analysis_result)

    return df, analysis_result, insights

# Initialisation des variables de session
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None

if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'file_analysis' not in st.session_state:
    st.session_state['file_analysis'] = None

if uploaded_file is not None:
    result = load_and_preprocess_data(uploaded_file)

    if isinstance(result, pd.DataFrame):
        st.session_state['df'] = result
        st.session_state['analysis_result'] = analyze_qonto_data(result, business_type)
        insights = generate_insights(st.session_state['analysis_result'])
        st.session_state['file_analysis'] = insights
        st.write("Analyse détaillée des données Qonto :")
        st.write(insights)

        # Ajoutez ici le code pour envoyer les insights à l'API et afficher la réponse

    else:
        st.session_state['df'] = None
        st.warning("Le fichier uploadé n'a pas pu être converti en DataFrame.")

    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)
        analysis = analyze_financial_image(image)
        st.write(analysis)
    elif uploaded_file.name.endswith(('.csv', '.xls', '.xlsx')):
        # Traitement des fichiers CSV et Excel
        pass  # Déjà traité ci-dessus
    else:
        st.warning("Type de fichier non pris en charge")
        analysis = "Aucune analyse disponible pour ce type de fichier."

    # Assurez-vous que analysis est définie avant de l'utiliser
    if analysis:
        response = swarm_client.run(
            agent=financial_planning_orchestrator_agent,
            messages=[{"role": "user", "content": f"Analyser ce document : {analysis}"}],
            context_variables={"document_analysis": analysis}
        )

        message = response.messages
        to_display = pretty_print_messages(messages=message)
        with st.chat_message("assistant"):
            st.markdown(to_display)
        st.session_state.messages.append({"role": "assistant", "content": to_display})
    else:
        st.write("Aucune analyse n'a été effectuée.")

    if 'analysis_result' in st.session_state and st.session_state['analysis_result']:
        save_analysis_history(st.session_state['analysis_result'])
        history = load_analysis_history()
        recommendations = generate_recommendations(st.session_state['analysis_result'], history)
        st.subheader("Recommandations basées sur l'historique")
        for rec in recommendations:
            st.write(f"• {rec}")

    # Dans la section d'analyse
    if 'df' in st.session_state and st.session_state['df'] is not None and not st.session_state['df'].empty:
        aggregation = st.selectbox(
            "Agrégation des données",
            ['D', 'W', 'M'],
            format_func=lambda x: {'D': 'Jour', 'W': 'Semaine', 'M': 'Mois'}[x],
            key='expense_trend_aggregation_main'
        )
        chart = create_expense_trend_chart(st.session_state['df'], key='expense_trend_main', aggregation=aggregation)
        if chart is not None:
            st.plotly_chart(chart, key='expense_trend_chart_main')
        else:
            st.warning("Impossible de créer le graphique de tendance des dépenses.")
    else:
        st.info("Aucune donnée disponible pour créer le graphique de tendance des dépenses.")

    if st.session_state['df'] is not None:
        date_range = st.date_input("Sélectionnez la période", [st.session_state['df']['Date'].min().date(), st.session_state['df']['Date'].max().date()])
        df_filtered = st.session_state['df'][(st.session_state['df']['Date'].dt.date >= date_range[0]) & (st.session_state['df']['Date'].dt.date <= date_range[1])]

        # Utilisez df_filtered pour d'autres visualisations ou analyses basées sur la période sélectionnée
    else:
        st.info("Téléchargez un fichier pour pouvoir sélectionner une période d'analyse.")

# React to user input
if prompt := st.chat_input("Posez votre question financière ici"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Incluez l'analyse du fichier dans le contexte
    context_variables = {}
    if st.session_state['file_analysis']:
        context_variables["file_analysis"] = st.session_state['file_analysis']

    response = swarm_client.run(
        agent=financial_planning_orchestrator_agent,
        messages=st.session_state.messages,
        context_variables=context_variables
    )
    message = response.messages
    agent = response.agent
    to_display = pretty_print_messages(messages=message)
    with st.chat_message("assistant"):
        st.markdown(to_display)
    st.session_state.messages.append({"role": "assistant", "content": to_display})

# Bouton pour effacer l'historique
if st.button("Effacer l'historique de conversation", key="clear_history"):
    st.session_state.messages = []
    st.rerun()

# Fonctions d'analyse et de visualisation
@st.cache_data
def generate_overview_chart(df):
    # Créer et retourner un graphique Plotly basé sur df
    pass  # À implémenter selon vos besoins

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="analyse_financiere.csv">Télécharger les résultats (CSV)</a>'
    return href

def visualize_recurring_expenses(df, key):
    fig = px.bar(df, x='Bénéficiaire', y='Montant',
                 title='Dépenses récurrentes potentielles',
                 labels={'Bénéficiaire': 'Bénéficiaire', 'Montant': 'Montant (€)'},
                 hover_data=['Fréquence', 'Première occurrence', 'Dernière occurrence'])
    return fig

# Section des résultats
st.header("Résultats de l'analyse")

if "file_analysis" in st.session_state and st.session_state["file_analysis"]:
    st.write(st.session_state["file_analysis"])
else:
    st.info("Veuillez télécharger un fichier pour voir l'analyse et les visualisations.")

# Dans votre code principal
if 'analysis_result' in st.session_state and st.session_state['analysis_result']:
    if 'recurring_expenses' in st.session_state['analysis_result']:
        recurring_df = pd.DataFrame(st.session_state['analysis_result']['recurring_expenses'])
        st.plotly_chart(visualize_recurring_expenses(recurring_df, key='recurring_expenses'), key='recurring_expenses_chart')

    # Ajoutez ici d'autres visualisations basées sur analysis_result

    # Par exemple, pour le graphique de tendance des dépenses
    if 'df' in st.session_state and st.session_state['df'] is not None and not st.session_state['df'].empty:
        aggregation = st.selectbox(
            "Agrégation des données",
            ['D', 'W', 'M'],
            format_func=lambda x: {'D': 'Jour', 'W': 'Semaine', 'M': 'Mois'}[x],
            key='expense_trend_aggregation_results'
        )
        chart = create_expense_trend_chart(st.session_state['df'], key='expense_trend_results', aggregation=aggregation)
        if chart is not None:
            st.plotly_chart(chart, key='expense_trend_chart_results')
        else:
            st.warning("Impossible de créer le graphique de tendance des dépenses.")
    else:
        st.info("Aucune donnée disponible pour créer le graphique de tendance des dépenses.")

# Pied de page
st.markdown("---")
st.caption("Assistant Financier IA spécialisé Qonto by Alexandre Lavallée")
