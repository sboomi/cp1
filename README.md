Cas pratique 1
==============================

**IMPORTANT NOTE:** for the sake of evaluation, the contents of the `README` are written exclusively in French. The comments, commits and technical wording are written in English for the sake of reusability.

## Détails du projet

En tant que nouvelle employée dans une société, je dois reprendre le code d'un employé licencié pour incompétence. Son code vise à analyser les commentaires LaFourchette en utilisant un `joblib` et une API Flask.

Le projet se décompose **initialement** comme ceci:

```
├── comments_train.csv
├── run.py
└── sentiment_pipe.joblib
```

Le code se lance comme une application Flask classique. Soit on lance le script `run.py` comme un entrypoint, soit on définit une variable d'environnement `FLASK_APP` dont la valeur est le chemin du script d'entrée, avant de lancer `flask run`.

### Les routes

L'application s'ouvre sur `http://localhost:8080/` et a deux routes :

* `/welcome`: une route de bienvenue (méthode GET)
* `/sentiment`: une route d'analyse de sentiment (méthode POST) : elle accepte un objet JSON qui prend deux clés, `token` et `text`. Si l'identifiant `token` est correct, la réponse renvoie une analyse `Positif` ou `Négatif`.

```json
// Requête
{"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
"text": "cette vidéo est géniale!"}

// Réponse
{
  "Status Code": 200,
  "prediction": "Positif",
  "text": "cette vidéo est géniale!"
}
```

### Données et modèle

Les données sont un CSV de 1617 commentaires qui sont tous en Français. Ils sont divisés en deux types d'avis: positifs et négatifs. La répartition des classes n'es pas égale (63% de commentaires positifs et 37% de commentaires négatifs).

Le modèle vise à faire de l'analyse de sentiments. La pipeline `sentiment_pipe` est constitué d'un TF-IDF et d'un SVM pour la classification.

Organisation finale
------------

    ├── LICENSE
    ├── Makefile           <- Makefile avec commandes `make data` ou `make train`
    ├── README.md          <- README pour les développeurs lisant le projet
    ├── data
    │   ├── external       <- Ressources externes au projet.
    │   ├── interim        <- Données issues d'un résultat intermédiaire.
    │   ├── processed      <- Données finales prêtes à être exploitées par le modèle 
    │   └── raw            <- Données originales.
    │
    ├── docs               <- Projet Sphinx de base pour la doc (sphinx-doc.org)
    │
    ├── models             <- Modèles entraînés et sérialisés, prédictions de modèles ou résumés de modèles.
    │
    ├── notebooks          <- Jupyter notebooks. La convention est un nombre n° de version, les initiales de l'auteur et un délimiteur `-` (ex. `1.0-sb-initial-exploration`) 
    │
    ├── references         <- Dictionnaires de données, manuels et autres références
    │
    ├── reports            <- Rapport d'analyse (HTML, PDF, LaTeX, etc)
    │   └── figures        <- Images générées pour le rapport
    │
    ├── requirements.txt   <- Le fichier pour reproduire l'environnement Python, générées avec `pip freeze > requirements.txt`
    │
    ├── setup.py           <- rend le projet pip installable (pip install -e .) de telle sorte à importer `src`
    ├── src                <- Code source du projet
    │   ├── __init__.py    <- Transforme src en module Python
    │   ├── api            <- Crée une API pour tester les résultats
    │   │   └── run.py     <- Point d'entrée de l'API
    │   ├── data           <- Scripts pour télécharger ou générer des données
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts pour transformer des données brutes en données de modélisation
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts pour entraîner les modèles et créer des prédictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts pour des figures exploratoires et des données visuelles
    │       └── visualize.py
    │
    └── tox.ini            <- fichier tox avec paramètres pour lancer tox (tox.readthedocs.io)


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

