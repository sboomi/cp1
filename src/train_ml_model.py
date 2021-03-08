import click
import logging
import pandas as pd
import json
from joblib import dump
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.ml_model import generate_best_model
from visualization.visualize import plot_confusion_matrix


@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
def main(csv_path, model_path):
    logger = logging.getLogger('ml-model')
    logger.info("Begin training on traditional ML")
    
    logger.info(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"CSV dimensions: {df.shape}")
    logger.info(f"Number of classes: {df.y.unique().size}")
    dist_class = df.y.value_counts(normalize=True)
    dist_fmt = ' / '.join([str(round(v*100, 2)) for v in dist_class.values])
    logger.info(f"Class proportion: {dist_fmt}")
    
    labels = sorted(df.y.unique().tolist(), reverse=True)
    X = df.x.values
    y = df.y.apply(lambda x: labels.index(x)).values
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        shuffle=True,
                                                        stratify=y)
    
    svm, svm_score = generate_best_model("svm", X_train, y_train)
    nb, nb_score = generate_best_model("naive_bayes", X_train, y_train)
    logger.info(f"Generated SVM model with score of {svm_score}")
    logger.info(f"Generated Naive Bayes model with score of {nb_score}")
    
    model_path = Path(model_path)
    svm_path = model_path / 'svm_pipe.joblib'
    nb_path = model_path / 'nb_pipe.joblib'
    
    dump(svm, svm_path)
    logger.info(f"SVM model saved at {svm_path}")
    
    dump(nb, nb_path)
    logger.info(f"Naive Bayes model saved at {nb_path}")
    
    # With SVM
    svm_cr = classification_report(y_test, 
                                   svm.predict(y_test),
                                   target_names=labels,
                                   output_dict=True)
    
    with open(model_path / 'svm_results.json', 'wb') as f:
        json.dump(svm_cr, f, indent=4)
        
    logger.info(f"SVM accuracy: {svm_cr['accuracy']}")
    for label in labels:
        lab_metrics = svm_cr[label]
        logger.info(f"CLASS {label.upper()}")
        for k, v in lab_metrics:
            logger.info(f"SVM {k}: {v}")
          
    # With Naive Bayes  
    nb_cr = classification_report(y_test,
                                  nb.predict(y_test),
                                  target_names=labels,
                                  output_dict=True)
    
    with open(model_path / 'nb_results.json', 'wb') as f:
        json.dump(nb_cr, f, indent=4)
        
    logger.info(f"Naive Bayes accuracy: {nb_cr['accuracy']}")
    for label in labels:
        lab_metrics = nb_cr[label]
        logger.info(f"CLASS {label.upper()}")
        for k, v in lab_metrics:
            logger.info(f"Naive Bayes {k}: {v}")
            
    plot_confusion_matrix(y_test, svm.predict(y_test), 
                          fn=model_path / 'svm_cm.png')
    logger.info("Confusion matrix for SVM plotted")
    
    plot_confusion_matrix(y_test, nb.predict(y_test), 
                          fn=model_path / 'naive_bayes_cm.png')
    logger.info("Confusion matrix for Naive Bayes plotted")
    
    
if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()