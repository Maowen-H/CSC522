import logging
import datetime
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import seaborn as sns

data = pd.read_csv("./datasets/train.csv")
data = data[['selected_text', 'sentiment']]
data = data.dropna()

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['label'] = data['sentiment'].map(label_mapping)


def clean_text(text):
    text = str(text).lower()  # 
    text = re.sub(r'\d+', '', text)  #
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  #
    text = re.sub(r'[^a-z\s]', '', text)  #
    return text


stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return words


data['selected_text'] = data['selected_text'].apply(clean_text).apply(tokenize_and_remove_stopwords)


word2vec_model = Word2Vec(sentences=data['selected_text'], vector_size=100, window=5, min_count=1, workers=4)

def get_word2vec_features(text):
    vector = np.zeros(100)
    count = 0
    for word in text:
        if word in word2vec_model.wv:
            vector += word2vec_model.wv[word]
            count += 1
    if count > 0:
        vector /= count
    return vector


X = np.array([get_word2vec_features(text) for text in data['selected_text']])
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# -----------------
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))  #
])


param_grid = {
    'rf__n_estimators': [50, 100, 200],  #
    'rf__max_depth': [10, 20,30,40, None],  # 
    'rf__min_samples_split': [2, 5, 10]  # 
}


class TQDMGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, **fit_params):
        n_candidates = len(self.param_grid['rf__n_estimators']) * \
                       len(self.param_grid['rf__max_depth']) * \
                       len(self.param_grid['rf__min_samples_split'])
        n_folds = self.cv
        total = n_candidates * n_folds  # 

        with tqdm(total=total, desc="Grid Search Progress") as pbar:
            def callback(*args, **kwargs):
                pbar.update(1)

            
            super().fit(X, y, **fit_params)

            pbar.close()

        return self

grid_search = TQDMGridSearchCV(
    pipeline, param_grid, cv=5, 
    scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'},
    refit='f1_macro', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"log_Rf{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")

logging.info("Best Parameters: %s", best_params)
logging.info("Classification Report:\n%s", report)


print("Best Parameters:", best_params)
print("Classification Report:\n", report)


