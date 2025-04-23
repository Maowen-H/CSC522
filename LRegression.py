import os
import datetime
import logging
import pandas as pd
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report



from sklearn.linear_model import LogisticRegression

# -----------------

data = pd.read_csv("./datasets/train.csv")
data = data[['selected_text', 'sentiment']]
data = data.dropna()


label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['label'] = data['sentiment'].map(label_mapping)



X_train, X_test, y_train, y_test = train_test_split(data['selected_text'], data['label'], test_size=0.3, random_state=1234)

pipeline = Pipeline([
   ('tfidf', TfidfVectorizer(stop_words='english')),  
    ('lr', LogisticRegression(max_iter=1000, random_state=42))  
])


param_grid = {
    'lr__C': [0.1, 1, 10,100],  #
    'tfidf__ngram_range': [(1, 1), (1, 2)],  
    'tfidf__max_features': [5000, 10000]  
}

class TQDMGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, **fit_params):

        n_candidates = len(self.param_grid['lr__C']) * \
                       len(self.param_grid['tfidf__ngram_range']) * \
                       len(self.param_grid['tfidf__max_features'])
        n_folds = self.cv
        total = n_candidates * n_folds  

   
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
log_filename = os.path.join(log_dir, f"log_LR{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")

logging.info("Best Parameters: %s", best_params)
logging.info("Classification Report:\n%s", report)


print("Best Parameters:", best_params)
print("Classification Report:\n", report)
