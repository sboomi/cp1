# -*- coding: utf-8 -*-
import click
import logging
import coloredlogs
import pandas as pd
import unicodedata
import string
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from nltk import stem


def clean_text(txt: str,
               stemmer: stem.api.StemmerI
               ) -> str:
    new_txt = ''.join([c.lower() if c not in string.punctuation else ' ' for c
                       in txt])
    new_txt = (unicodedata.normalize('NFKD', new_txt)
               .encode('ascii', 'ignore')
               .decode('utf-8', 'ignore'))
    new_txt = " ".join([word for word in new_txt.split() if word])
    new_txt = ' '.join([stemmer.stem(word) for word in new_txt.split()])
    return new_txt


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('csv_file', type=click.Path(exists=True))
def main(input_filepath: str,
         output_filepath: str,
         csv_file: str
         ) -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)
    output_filepath.mkdir(exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info(f"N° of files: {len(list(input_filepath.iterdir()))}")

    stemmer = stem.regexp.RegexpStemmer('s$|es$|era$|erez$|ions$| <etc> ')
    logger.info(f"Using {stemmer.__class__.__name__} as stemmer.")

    logger.info(f"Loading CSV from {csv_file}")
    df = pd.read_csv(csv_file)

    logger.info(f"{df.shape[0]} lines / {df.shape[1]} columns")

    df.comment = df.comment.apply(lambda x: clean_text(x, stemmer=stemmer))

    df.columns = ['x', 'y']

    logger.info(f"Saving at {output_filepath / 'comments_clean.csv'}")
    df.to_csv(output_filepath / "comments_clean.csv", index=None)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    coloredlogs.install()

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
