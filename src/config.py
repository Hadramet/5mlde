import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_FOLDER = os.path.join(BASE_PATH, 'data')
DATA_PATH = os.path.join(DATA_FOLDER, 'spam_emails_1.csv')

OUTPUT_FOLDER = os.path.join(BASE_PATH, 'results')
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, 'models')
MODEL_PATH = os.path.join(MODEL_FOLDER, 'model.pkl')

TV_FOLDER = os.path.join(OUTPUT_FOLDER, 'tv')
TV_PATH = os.path.join(TV_FOLDER, 'tv.pkl')

COLUMN_TO_DROP = ['Unnamed: 0', "label", "label_num"]
LABEL = {'ham': 0, 'spam': 1}
