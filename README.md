# ♟ OpenZero Chess

**OpenZero** est un moteur de jeu d’échecs basé sur un **réseau de neurones convolutionnel** inspiré d’**AlphaZero**, couplé à un algorithme de recherche **Monte Carlo Tree Search (MCTS)**.

L’application JavaFX permet de :
- Jouer contre OpenZero (avec ou sans MCTS)
- Observer des parties entre deux IA
- Visualiser en direct les évaluations positionnelles

---
## 🚀 Lancer l'application

### 🧰 Prérequis

- [Java 17+](https://adoptium.net/)
- [Maven 3.6+](https://maven.apache.org/)
- Un environnement JavaFX fonctionnel (JavaFX est géré automatiquement via Maven)

### 🏁 Étapes de démarrage

```bash
# 1. Clonez le dépôt
git clone https://github.com/Max89100/OpenZero.git
cd OpenZero

# 2. Récupérez le modèle OpenZero5.zip
# Téléchargez le depuis la branche model-files
# Et placez le dans le dossier /resources comme indiqué dans l'arborescence ci-dessous

# 3. Compilez le projet
mvn clean install

# 4. Lancez l'application
mvn javafx:run

# Arborescence du projet
OpenZero/
├── .idea/                        # Répertoire IntelliJ (à ignorer dans Git)
├── src/
│   └── main/
│       ├── java/
│       │   └── openzero/
│       │       ├── gui/         # Interface JavaFX
│       │       ├── MCTS/        # Arbre de recherche Monte Carlo
│       │       ├── nn/          # Réseau de neurones
│       │       ├── pipeline/    # Nécessaire pour l'entraînement
│       │       └── utils/       # Fonctions utilitaires
│       └── resources/
│           ├── pieces/          # Pièces d’échecs (.png)
│           ├── sounds/          # Sons du jeu (pas implémenté)
│           ├── style/           # CSS (dark-theme.css)
│           ├── OpenZero.png     # Logo de l'application
│           └── OpenZero5.zip    # Modèle CNN 
├── DatasetMaker.py              # Script Python d'entraînement
├── target/                      # Répertoire Maven (à ignorer)
├── bot_battle_results.txt       # Logs de parties entre bots
├── pom.xml                      # Configuration Maven
├── README.md                    # Ce fichier
└── .gitignore                   # Fichiers à exclure du suivi Git
