# FastAPI vs Flask

Une API (Application Programming Interface) est un protocole de communication entre différentes applications et programmes. Il permet d’exploiter le mécanisme sous-jacent en l'interprétant comme commandes. Elles sont très utilisées pour les applications web, les bases de données et la traduction entre langages informatiques (ex. de C++ à Python via des modules comme Numpy).

En fonction du protocole, les API suivent des règles différentes. C’est le cas notamment de la majorité des API web actuelles, qui suivent le protocole REST (REpresentational State Transfer). Ce protocole suit la convention HTTP qui est définie par les règles suivantes :

1. Besoin d’un **sujet** : on identifie une URI sur laquelle se connecter
2. Besoin d’un **verbe** : on identifie le type d’opération à utiliser sur cette URI
3. Chaque requête dont le sujet et verbe sont définis doivent retourner une **réponse** qui représente les données recherchées
4. Facultatif: certaines réponses peuvent avoir un complément à faire passer dans le corps de la requête

Les API web communiquent en faisant passer les données nécessaires sous un certain format comme le format JSON ou XML. On dit que ces API produisent les données pour que les applications reliées puissent les consommer. 

En Python il existe plusieurs moyens de créer ou d’utiliser les APIs, comme la bibliothèque ``requests``. Dans notre cas d’étude, on dispose de deux moyens : l’API Flask via l’extension Flask RESTful et FastAPI.

## Flask + Flask RESTful

Flask par défaut est un framework web de Python qui est minimaliste au possible. A l’instar de Django, qui fournit tous les modules nécessaires pour créer une application web de grande envergure, Flask ne dispose que du routeur et des templates. On doit donc l’enrichir d’extensions comme Flask-SQLAlchemy ou encore Flask RESTful. 

Flask RESTful est l’extension qui permet d’utiliser le serveur Flask en API. Elle s’organise en objets ``Resource``, qui peuvent hériter des méthodes correspondant aux verbes HTTP. La ressource doit ensuite être déclarée auprès de l’API avec ``add_ressource``.

```python
from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)
```

Chaque ressource peut hériter de plusieurs méthodes à la fois et chacune des ces méthodes peut détecter le corps de la requête, l’analyser et y traiter les informations.

Le point faible de cette API est le manque de contrôle sur les informations qui y sont passées en corps de message. Il est possible de typer les objets en entrée en utilisant la fonction ``marshal_with()``, mais tout ceci requiert davantage de code. 

```python
from flask_restful import fields, marshal_with

resource_fields = {
    'task':   fields.String,
    'uri':    fields.Url('todo_ep')
}

class TodoDao(object):
    def __init__(self, todo_id, task):
        self.todo_id = todo_id
        self.task = task

        # This field will not be sent in the response
        self.status = 'active'

class Todo(Resource):
    @marshal_with(resource_fields)
    def get(self, **kwargs):
        return TodoDao(todo_id='my_todo', task='Remember the milk')
```


De plus, comme Flask fait uniquement du templating, tous les échanges client<->serveur sont synchrones. Lorsqu’on a besoin de requêtes AJAX, il faut implémenter des fonctionnalités JavaScript ce qui prend beaucoup plus de temps.

## FastAPI

FastAPI est une bibliothèque de fonctions qui a été créée courant 2019 et qui est entrain de monter rapidement en puissance au cours de l’année 2020, et qui pourrait bien dépasser le framework REST de Django.

![fastapi_star_history](figures\fastapi_star_history.jpg)

La bibliothèque a l’avantage de se limiter uniquement aux fonctions API nécessaires et s’avère être plus rapide que la plupart des frameworks web/API Python existants.

```python
from typing import Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
```

FastAPI, tout comme Flask, instancie des routes dans son application. A l’instar de Flask, on note deux différences majeures dans le code :

Les routes autorisent le typage Python pour renforcer les arguments passés dans la fonction. 
Les dictionnaires renvoyés en résultat sont directement interprétés comme du JSON. Inutile de passer par la fonction jsonify
En plus d’être rapide à utiliser, la bibliothèque propose une documentation sur l’API interactive (`/docs`) ou alternative (``/redoc``), qui est directement compilée à partir du nom de la route, du verbe de la route, des arguments de fonctions ainsi que de leur type.

La documentation interactive s’actualise au moindre changement, si on fait passer --reload dans les paramètres, et permet de générer un lien de requête ou un lien curl à l’aide de son propre service REST (ce qui évite à utiliser Postman ou Insomnia).

FastAPI privilégie surtout le typage pour renforcer la sécurité des arguments passés en objets. Il est donc possible de customiser chacun des types en utilisant ``BaseModel`` de ``pydantic``, par exemple.
