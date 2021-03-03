# Les 5 points d'analyse

* Interpréter les indicateurs de performance de l’intelligence artificielle disponibles,
* A partir des éléments d’interprétation, définir les caractéristiques des améliorations à apporter,
* Intégrer les améliorations à l'algorithme d’intelligence artificielle
* Communiquer une estimation de charge au regard du besoin d’évolution de l’application,
* Intégrer l’évolution fonctionnelle,
* Tester la non régression de l’application suite à l’intégration de l’évolution

## Indicateurs de performance de l'IA

Comme la problématique est centrée autour de l'analyse de sentiments, il s'agit donc d'un problème de **classification** où les métriques de performance sont les suivants :

* Matrice de confusion
* Précision et recall
* Aire sous la courbe ROC
* Score F1 (dans le cas où les données sont très déséquilibrées)

Si on veut rajouter des réseaux de neurones du deep learning, on peut rajouter **l'évolution de la fonction de perte** d'après un critère appelé perte d'entropie croisée (*cross-entropy loss*).

## Amélioration d'algorithme

En premier lieu il faudrait diviser le CSV en une partie d'entraînement et d'une partie test/validation, de préférence 80/20 % avec une répartition des classes appropriées.

Chaque commentaire devra être tokenisé et lemmatisé avant d'être converti en matrice bag-of-words. Enfin, comme il s'agit d'une classification binaire, chaque classe sera associée à 1 et 0.

Une régression logistique serait peut-être appropriée pour ce genre de problématique dans un premier temps.

En cas d'échec, on pourrait essayer les algorithmes suivants :

* Forêts aléatoires
* Réseau de neurones à plusieurs couches
* Plus de préprocessing en faisant appel à un algorithme non-supervisé

## Estimation de charge

|            | Mercredi 3/3                                                 | Jeudi 4/3                                                    | Vendredi 5/3                                      | Lundi 29/3   | Mardi 30/3   |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- | ------------ | ------------ |
| Matin      | Estimation du cahier des charges et de la problématique      | Implémentation et mise en place de l'algorithme via scripts en mise en place d'un 2e cas | Refonte de l'API Flask avec ajout de l'API FastAI | A déterminer | A déterminer |
| Après-midi | Etude sur l'analyse de sentiments + aperçu des performances du modèle actuel | Analyse des performances des modèles avec métriques et estimation du meilleur | A déterminer                                      | A déterminer | A déterminer |

## Evolution fonctionnelle

On aimerait pouvoir utiliser ce modèle dans une API où l'utilisateur, si il est en possession d'un token, puisse envoyer un message sur cette API. L'application lui renvoie son appréciation, positive ou négative, et en complément le degré de prédiction de l'algorithme.