# Cas pratique 1

**IMPORTANT NOTE:** for the sake of evaluation, the contents of the `README` are written exclusively in French. The comments, commits and technical wording are written in English for the sake of reusability.

## Détails du projet

En tant que nouvelle employée dans une société, je dois reprendre le code d'un employé licencié pour incompétence. Son code vise à analyser les commentaires YouTube en utilisant un `joblib` et une API Flask.

Le projet se décompose initialement comme ceci:

```
.
├── README.md
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

