# Swarm Finance Multi-Agent

Ce projet utilise le système Swarm d'OpenAI pour créer un assistant financier personnel multi-agent basé sur GPT-4o.

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
