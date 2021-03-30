import click
import logging
import numpy as np
import pandas as pd
import json
import time
import datetime as dt
import coloredlogs
from joblib import dump, load
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.ml_model import generate_best_model
from visualization.visualize import plot_confusion_matrix


@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.option("--compare-models", "-cm", "comp_model",
              type=click.Path(exists=True))
@click.option("--random-seed", "-rs", "seed", type=int, default=32451365)
def main(csv_path: str,
         model_path: str,
         comp_model: str,
         seed: int = 32451365
         ) -> None:
    rng = np.random.RandomState(seed)
    start_time = time.time()
    logger = logging.getLogger('ml-model')
    logger.info("Begin training on traditional ML")

    logger.info(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"CSV dimensions: {df.shape}")
    logger.info(f"Number of classes: {df.y.unique().size}")
    dist_class = df.y.value_counts(normalize=True)
    dist_fmt = ' / '.join([str(round(v*100, 2)) for v in dist_class.values])
    logger.info(f"Class proportion: {dist_fmt}")

    labels = sorted(df.y.unique().tolist())
    logger.info(f"Labels: {labels}")
    X = df.x.values
    y = df.y.apply(lambda x: labels.index(x)).values

    logger.info("Beginning train and test split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.8,
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state=rng)

    list_models = {
        "svm": "SVM",
        "naive_bayes": "Naive Bayes",
        "lr": "Logistic regression"
    }
    model_path = Path(model_path)

    if comp_model:
        list_models["old_model"] = "Old model"

    for model_abbr, model_name in list_models.items():
        logger.info(f"Beginning estimation of {model_name}")
        if model_abbr == "old_model":
            model = load(comp_model)
            logger.info(f"Retrieved {model_name} at {comp_model}")
        else:
            model, model_score = generate_best_model(model_abbr,
                                                     X_train, y_train)
            logger.info(f"Generated {model_name} with score of {model_score}")
            model_joblib = model_path / f'{model_abbr}_pipe.joblib'
            dump(model, model_joblib)
            logger.info(f"{model_name} saved at {model_joblib}")

        model_cr = classification_report(y_test,
                                         model.predict(X_test),
                                         target_names=labels,
                                         output_dict=True)

        with open(model_path / f'{model_abbr}_results.json', 'w') as f:
            json.dump(model_cr, f, indent=4)

        logger.info(f"{model_name} accuracy: {model_cr['accuracy']}")
        for label in labels:
            lab_metrics = model_cr[label]
            logger.info(f"CLASS {label.upper()}")
            for k, v in lab_metrics.items():
                logger.info(f"{model_name} {k}: {v}")

        plot_confusion_matrix(y_test, model.predict(X_test),
                              labels=labels,
                              fn=model_path / f'{model_abbr}_cm.png')
        logger.info(f"Confusion matrix for {model_name} plotted")

    end_time = time.time()
    tot_time = str(dt.timedelta(seconds=end_time-start_time))
    h, mn, s = tot_time.split(":")
    logger.info(f"Script ended in {h}h, {mn}min and {s}s")


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    coloredlogs.install()

    main()
