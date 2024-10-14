# Swarm Finance Multi-Agent

Ce projet utilise le système Swarm d'OpenAI pour créer un assistant financier personnel multi-agent basé sur GPT-4.

## Prérequis

- Python 11
- Git
- une clef API OpenAI pour utiliser GPT-4o ;) - : [tuto pour trouver sa clef](https://docs.google.com/document/d/1rXTdYsDDOY7ukc_95Ee69YKFVbzslrxYIF2Px0LqPLo/edit?usp=sharing) 

## Installation

1. Créez un environnement virtuel Python 11 :
   ```
   python3.11 -m venv venv
   ```

2. Activez l'environnement virtuel :
   ```
   source venv/bin/activate  # Sur Unix ou MacOS
   venv\Scripts\activate  # Sur Windows
   ```

3. Clonez le dépôt :
   ```
   git clone https://github.com/Aximande/swarn-finance-multi-agent.git
   ```

4. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

5. Installez Swarm depuis GitHub :
   ```
   pip install git+https://github.com/openai/swarm.git
   ```

## Structure du projet

swarn-finance-multi-agent/
├── main.py
├── agents.py
├── utils.py
└── requirements.txt
