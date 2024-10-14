# Import required libraries
from swarm import Swarm, Agent
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import pandas as pd
import base64
import io
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

# Initialize Swarm client
client = OpenAI()
swarm_client = Swarm()

# Define specialized financial agents

# 1. Risk Management Agent
risk_management_agent = Agent(
    name="Risk Management Agent",
    instructions="""
        This agent focuses on protecting the user's financial well-being.
        Using the insurance and asset information provided by the Orchestrator,
        it evaluates overall financial risks, analyzes existing coverage,
        and identifies any protection gaps.

        The Risk Management Agent recommends appropriate insurance products,
        suggests asset protection strategies, and advises on contingency planning.
        It helps ensure financial stability in the face of unexpected events,
        regularly updating the Orchestrator with risk assessments and mitigation strategies.
    """,
)

# 2. Investment Strategy Agent
investment_strategy_agent = Agent(
    name="Investment Strategy Agent",
    instructions="""
        This agent handles all aspects of the user's investment portfolio.
        Based on the current investment information received from the Orchestrator,
        it assesses risk tolerance, sets investment goals, and recommends asset allocation strategies.

        The Investment Strategy Agent monitors market trends, suggests portfolio adjustments,
        and analyzes various investment opportunities across different asset classes.
        It aims to optimize returns while managing risk,
        consistently updating the Orchestrator with its findings and recommendations.
    """,
)

# 3. Budget Analysis Agent
budget_analysis_agent = Agent(
    name="Budget Analysis Agent",
    instructions="""
        This agent focuses on day-to-day financial management.
        Using the income and expense data provided by the Orchestrator,
        it analyzes spending patterns and creates detailed budgets.

        The Budget Analysis Agent identifies potential savings opportunities,
        suggests ways to optimize cash flow, and provides accurate financial projections.
        It continually monitors transactions to ensure alignment with the user's budget and financial goals,
        feeding updates back to the Orchestrator for comprehensive planning.
    """,
)

# Define transfer functions for agent-specific tasks

def transfer_to_budget_agent():
    """
    Transfer to questions/information related to Budget/Expense management.
    """
    return budget_analysis_agent

def transfer_to_investment_agent():
    """
    Transfer to questions/information related to Investment strategy management.
    """
    return investment_strategy_agent

def transfer_to_risk_agent():
    """
    Transfer to questions/information related to Insurance/Risk management.
    """
    return risk_management_agent

# Orchestrator Agent - central hub for managing all other agents
financial_planning_orchestrator_agent = Agent(
    name="Financial Planning Orchestrator Agent",
    instructions="""
        This agent serves as the system's central hub and initial point of contact.
        It begins by thoroughly gathering information about the user's current financial situation,
        including investments, income sources, expenses, and existing insurance policies.

        Once a comprehensive financial profile is established, the agent distributes relevant data
        to the specialized agents for detailed analysis.

        After receiving insights from other agents, it synthesizes the information to provide cohesive,
        personalized financial advice. The Orchestrator prioritizes tasks, manages inter-agent communication,
        and translates complex financial data into actionable, easy-to-understand guidance for the user.
    """,
    functions=[transfer_to_budget_agent, transfer_to_investment_agent, transfer_to_risk_agent]
)

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def analyze_financial_image(image):
    base64_image = encode_image(image)
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            with OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0) as client:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Vous êtes un expert en analyse financière. Analysez l'image fournie et fournissez un résumé des informations financières pertinentes."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analysez cette image financière et résumez les informations clés."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Tentative {attempt + 1} échouée. Nouvelle tentative dans {retry_delay} secondes...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return f"Une erreur s'est produite lors de l'analyse de l'image après {max_retries} tentatives : {str(e)}"

def analyze_spreadsheet(file):
    try:
        # Lire le fichier
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xls'):
            df = pd.read_excel(file, engine='xlrd')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            return "Format de fichier non pris en charge. Veuillez utiliser .csv, .xls ou .xlsx."

        # Analyse simplifiée
        total_expenses = df['Debit'].sum()
        top_expenses = df.groupby('Counterparty name')['Debit'].sum().nlargest(5)

        analysis = f"""
        Analyse simplifiée des dépenses :

        1. Dépenses totales : {total_expenses:.2f}
        2. Top 5 des catégories de dépenses :
        {top_expenses.to_string()}

        Recommandations :
        1. Concentrez-vous sur la réduction des dépenses dans la catégorie principale.
        2. Examinez les opportunités d'économies dans les 5 principales catégories de dépenses.
        """

        return analysis
    except Exception as e:
        return f"Une erreur s'est produite lors de l'analyse du fichier : {str(e)}"

# Modifier la fonction existante analyze_csv pour utiliser analyze_spreadsheet
def analyze_csv(csv_file):
    return analyze_spreadsheet(csv_file)

def transfer_to_document_analysis_agent():
    return document_analysis_agent

main_agent = Agent(
    name="Agent Principal",
    instructions="Vous êtes un assistant financier capable de discuter et de diriger l'analyse de documents.",
    functions=[transfer_to_document_analysis_agent]
)

document_analysis_agent = Agent(
    name="Agent d'Analyse de Document",
    instructions="Vous êtes spécialisé dans l'analyse de documents financiers, y compris les images et les fichiers CSV.",
    functions=[analyze_financial_image, analyze_csv]
)

def visualize_and_optimize_data(file):
    # Lire le fichier
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        return "Format de fichier non pris en charge. Veuillez utiliser .csv, .xls ou .xlsx."

    # Afficher un aperçu des données
    st.write("Aperçu des données :")
    st.dataframe(df.head())

    # Afficher les informations sur les colonnes
    st.write("Informations sur les colonnes :")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Visualiser la distribution des valeurs numériques
    st.write("Distribution des valeurs numériques :")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[numeric_columns], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Permettre à l'utilisateur de sélectionner les colonnes à conserver
    st.write("Sélectionnez les colonnes à conserver pour l'analyse :")
    selected_columns = st.multiselect("Colonnes", df.columns.tolist(), default=df.columns.tolist())

    # Créer un nouveau DataFrame avec les colonnes sélectionnées
    optimized_df = df[selected_columns]

    # Afficher les statistiques de base du DataFrame optimisé
    st.write("Statistiques de base du DataFrame optimisé :")
    st.write(optimized_df.describe())

    return optimized_df
