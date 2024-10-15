# Import required libraries
from swarm import Swarm, Agent
import os
from dotenv import load_dotenv
import openai
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import streamlit as st
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize Swarm client
swarm_client = Swarm()
openai_client = openai.Client()

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

def encode_image(image):
    if isinstance(image, Image.Image):
        # Si c'est déjà un objet Image, on le convertit en bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    elif hasattr(image, 'read'):
        # Si c'est un objet file-like (comme un BytesIO ou un fichier ouvert)
        return base64.b64encode(image.read()).decode('utf-8')
    elif isinstance(image, bytes):
        # Si c'est déjà un objet bytes
        return base64.b64encode(image).decode('utf-8')
    else:
        raise ValueError("Format d'image non pris en charge")

def analyze_financial_image(image):
    base64_image = encode_image(image)
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Vous êtes un expert en analyse financière. Analysez l'image fournie et fournissez un résumé des informations financières pertinentes."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analysez cette image financière et résumez les informations clés."
                            },
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
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return "Format de fichier non pris en charge. Veuillez utiliser .csv, .xls ou .xlsx."

        # Analyse plus détaillée
        total_transactions = len(df)
        total_income = df['Credit'].sum()
        total_expenses = df['Debit'].sum()
        net_cash_flow = total_income - total_expenses
        top_expenses = df.groupby('Counterparty name')['Debit'].sum().nlargest(10)
        top_income_sources = df.groupby('Counterparty name')['Credit'].sum().nlargest(5)
        monthly_expenses = df.groupby(pd.to_datetime(df['Date']).dt.to_period('M'))['Debit'].sum()
        monthly_income = df.groupby(pd.to_datetime(df['Date']).dt.to_period('M'))['Credit'].sum()

        analysis = f"""
        Analyse détaillée des transactions :

        1. Nombre total de transactions : {total_transactions}
        2. Total des revenus : {total_income:.2f}
        3. Total des dépenses : {total_expenses:.2f}
        4. Flux de trésorerie net : {net_cash_flow:.2f}
        5. Top 10 des catégories de dépenses :
        {top_expenses.to_string()}
        6. Top 5 des sources de revenus :
        {top_income_sources.to_string()}
        7. Évolution mensuelle des dépenses :
        {monthly_expenses.to_string()}
        8. Évolution mensuelle des revenus :
        {monthly_income.to_string()}

        Recommandations :
        1. Concentrez-vous sur la réduction des dépenses dans les principales catégories.
        2. Analysez les tendances mensuelles pour identifier les opportunités d'optimisation.
        3. Évaluez la stabilité des sources de revenus.
        """

        return analysis
    except Exception as e:
        return f"Une erreur s'est produite lors de l'analyse du fichier : {str(e)}\n" \
               f"Détails de l'erreur : {e.__class__.__name__}\n" \
               f"Ligne : {e.__traceback__.tb_lineno}"

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

    # Générer une analyse textuelle
    analysis = f"""
    Analyse des données financières :

    1. Le fichier contient {len(df)} entrées et {len(df.columns)} colonnes.
    2. Les colonnes principales sont : {', '.join(selected_columns)}.
    3. La moyenne des dépenses est de {optimized_df['Debit'].mean():.2f}.
    4. La transaction la plus élevée est de {optimized_df['Debit'].max():.2f}.
    5. Il y a {optimized_df['Counterparty name'].nunique()} contreparties uniques.
    """

    return optimized_df, analysis

def get_financial_advice(analysis, data_json):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Vous êtes un conseiller financier expert. Analysez les données fournies et donnez des conseils financiers détaillés et personnalisés."
                },
                {
                    "role": "user",
                    "content": f"Voici l'analyse d'une situation financière : {analysis}\n\nEt voici les données brutes : {data_json}\n\nPouvez-vous fournir une analyse approfondie et des conseils financiers personnalisés basés sur ces informations ?"
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de la génération des conseils : {str(e)}"

def process_financial_data(file):
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    analysis = analyze_spreadsheet(file)
    data_json = df.to_json(orient='records')
    advice = get_financial_advice(analysis, data_json)

    response = swarm_client.run(
        agent=financial_planning_orchestrator_agent,
        messages=[
            {"role": "user", "content": f"Voici l'analyse des données financières : {analysis}\n\nEt voici les conseils générés : {advice}\n\nPouvez-vous synthétiser ces informations et fournir un plan d'action global détaillé ?"}
        ]
    )

    orchestrator_response = response.messages[-1]["content"]

    return df, analysis, advice, orchestrator_response

def preprocess_date(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    return df.dropna(subset=['Date'])

def preprocess_qonto_data(df):
    # Renommez les colonnes si nécessaire
    column_mapping = {
        'Operation date (UTC)': 'Date',
        'Total amount (incl. VAT)': 'Montant',
        'Currency': 'Devise',
        'Counterparty name': 'Bénéficiaire',
        'Payment method': 'Méthode_Paiement',
        'Initiator': 'Initiateur',
        'Team': 'Équipe',
        'Category': 'Catégorie'
    }
    df = df.rename(columns=column_mapping)

    # Convertissez la colonne de date
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')

    # Assurez-vous que le montant est numérique
    df['Montant'] = pd.to_numeric(df['Montant'], errors='coerce')

    return df

def generate_insights(analysis):
    insights = [
        f"Total des transactions : {analysis['total_transactions']}",
        f"Total des dépenses : {analysis['total_debit']:.2f} {analysis['currency']}",
        f"Total des revenus : {analysis['total_credit']:.2f} {analysis['currency']}",
        f"Solde actuel : {analysis['balance']:.2f} {analysis['currency']}",
        f"Période analysée : {analysis['date_range']}",
        "\nTop 5 des dépenses par bénéficiaire :",
        *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['top_expenses'].items()],
        "\nTop 5 des revenus par source :",
        *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['top_income'].items()],
        "\nDépenses par catégorie :",
        *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['expenses_by_category'].items()],
        "\nDépenses par méthode de paiement :",
        *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['expenses_by_payment_method'].items()],
        "\nDépenses par équipe :",
        *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['expenses_by_team'].items()],
        "\nTop 5 des initiateurs de dépenses :",
        *[f"- {k}: {v:.2f} {analysis['currency']}" for k, v in analysis['top_initiators'].items()]
    ]
    return "\n".join(insights)

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
        analysis["top_expenses"] = df[df['Montant'] < 0].groupby('Bénéficiaire')['Montant'].sum().nlargest(5).to_dict()
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

    return analysis

def process_qonto_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        raise ValueError("Format de fichier non supporté")

    df = preprocess_qonto_data(df)
    analysis_result = analyze_qonto_data(df)
    insights = generate_insights(analysis_result)

    return df, analysis_result, insights
