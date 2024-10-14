# Swarm Finance Multi-Agent

Ce projet utilise le système Swarm d'OpenAI pour créer un assistant financier personnel multi-agent basé sur GPT-4o, avec une interface utilisateur Streamlit.

## Qu'est-ce que Streamlit ?

Streamlit est un framework open-source qui permet de créer facilement des applications web interactives pour la data science et le machine learning. Il transforme vos scripts Python en applications web sans nécessiter de connaissances en développement web. C'est l'outil que nous utilisons pour rendre notre assistant financier accessible et facile à utiliser.

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
main.py
agents.py
utils.py
requirements.txt



## Lancement de l'application Streamlit

Une fois que vous avez terminé toutes les étapes d'installation, suivez ces instructions pour lancer l'application Streamlit :

1. Ouvrez votre terminal (Invite de commandes sur Windows, Terminal sur Mac ou Linux).

2. Naviguez vers le dossier de votre projet. Si vous n'êtes pas sûr de comment faire, voici les étapes :
   - Sur Windows, tapez `cd chemin\vers\votre\projet`
   - Sur Mac ou Linux, tapez `cd chemin/vers/votre/projet`
   Remplacez "chemin/vers/votre/projet" par le chemin réel vers le dossier de votre projet.

3. Assurez-vous que votre environnement virtuel est activé. Vous devriez voir `(venv)` au début de votre ligne de commande. Si ce n'est pas le cas, activez-le :
   - Sur Windows : `venv\Scripts\activate`
   - Sur Mac ou Linux : `source venv/bin/activate`

4. Une fois dans le bon dossier et avec l'environnement virtuel activé, lancez l'application en tapant :
   ```
   streamlit run main.py
   ```

5. Après quelques secondes, vous devriez voir un message dans le terminal indiquant que l'application est en cours d'exécution. Il affichera également une URL locale, généralement `http://localhost:8501`.

6. Votre navigateur web devrait s'ouvrir automatiquement avec l'application. Si ce n'est pas le cas, copiez l'URL affichée dans le terminal (généralement `http://localhost:8501`) et collez-la dans la barre d'adresse de votre navigateur web.

7. Vous devriez maintenant voir l'interface de votre application Streamlit !

Si vous rencontrez des problèmes ou si l'application ne se lance pas, assurez-vous que toutes les étapes d'installation ont été correctement suivies et que toutes les dépendances sont installées.

Pour arrêter l'application, retournez dans le terminal et appuyez sur Ctrl+C.

## Workflow d'utilisation

Voici un exemple de workflow pour utiliser notre application avec vos données bancaires de Qonto :

1. Exportez vos transactions depuis Qonto :
   - Connectez-vous à votre compte Qonto.
   - Allez dans la section "Transactions".
   - Utilisez les filtres pour sélectionner les transactions de 2024 (ou la période souhaitée).
   - Cliquez sur "Exporter" et choisissez le format CSV (données complètes ou simplifiées selon vos besoins).

2. Préparez votre fichier CSV :
   - Une fois le fichier téléchargé, ne le modifiez pas. L'application est conçue pour traiter le format standard de Qonto.

3. Utilisez l'application Streamlit :
   - Lancez l'application comme expliqué dans la section précédente.
   - Dans l'interface de l'application, vous verrez une zone de "drag and drop" ou un bouton pour télécharger votre fichier.
   - Faites glisser votre fichier CSV Qonto dans cette zone ou utilisez le bouton pour le sélectionner.

4. Analyse des données :
   - Une fois le fichier chargé, l'application traitera automatiquement vos transactions.
   - Vous verrez apparaître des analyses, des graphiques et des recommandations basées sur vos données financières.

5. Interaction avec l'assistant :
   - Utilisez la zone de chat pour poser des questions spécifiques à l'assistant sur vos finances.
   - L'assistant multi-agent utilisera les données de vos transactions pour fournir des réponses personnalisées et des conseils financiers.

🚧 N'oubliez pas : vos données financières sont sensibles. Cette application les traite **localement** sur votre machine pour plus de sécurité, mais veillez à ne pas partager votre écran ou vos fichiers CSV avec des personnes non autorisées.

