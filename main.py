import streamlit as st
import openai
import logging
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)

# Cette ligne doit être la première commande Streamlit
st.set_page_config(page_title="Assistant Financier IA", layout="wide")

import os
from agents import financial_planning_orchestrator_agent, document_analysis_agent, swarm_client, analyze_financial_image, analyze_spreadsheet, visualize_and_optimize_data
from utils import pretty_print_messages

st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Entrez votre clé API OpenAI", type="password")
if not api_key:
    st.sidebar.warning("Veuillez entrer votre clé API OpenAI pour continuer.")
    st.stop()

st.title("Assistant Financier IA")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File uploader for documents
uploaded_file = st.file_uploader(
    "Glissez-déposez votre document financier (.csv, .xls, .xlsx) ou capture d'écran (.png, .jpg, .jpeg)",
    type=["csv", "xls", "xlsx", "png", "jpg", "jpeg"],
    key="document_uploader"
)

# Initialiser analysis
analysis = ""

if uploaded_file is not None:
    if uploaded_file.name.endswith(('.csv', '.xls', '.xlsx')):
        st.write(f"Fichier {uploaded_file.type} détecté")
        analysis = analyze_spreadsheet(uploaded_file)
        st.write(analysis)

        # Envoi d'une version tronquée à l'API
        truncated_analysis = analysis[:3000]  # Limiter à 3000 caractères
        try:
            response = swarm_client.run(
                agent=financial_planning_orchestrator_agent,
                messages=[{"role": "user", "content": f"Analyser ce résumé financier et donner des conseils : {truncated_analysis}"}],
                context_variables={"document_analysis": truncated_analysis}
            )
            message = response.messages
            to_display = pretty_print_messages(messages=message)
            with st.chat_message("assistant"):
                st.markdown(to_display)
            st.session_state.messages.append({"role": "assistant", "content": to_display})
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'analyse : {str(e)}")
    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
        analysis = analyze_financial_image(uploaded_file)
        st.write(analysis)
    else:
        st.write("Type de fichier non pris en charge")
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

# React to user input
if prompt := st.chat_input("Posez votre question financière ici"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = swarm_client.run(agent=financial_planning_orchestrator_agent, messages=st.session_state.messages)
    message = response.messages
    agent = response.agent
    to_display = pretty_print_messages(messages=message)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(to_display)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": to_display})

# Add a button to clear the chat history
if st.button("Effacer l'historique de conversation"):
    st.session_state.messages = []
    st.rerun()
