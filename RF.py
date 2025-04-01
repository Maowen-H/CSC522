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

# -----------------
# 1. 数据加载和预处理
# -----------------
data = pd.read_csv("./datasets/train.csv")
data = data[['selected_text', 'sentiment']]
data = data.dropna()

# **三分类**: 将情感标签转换为 0, 1, 2
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['label'] = data['sentiment'].map(label_mapping)

# -----------------
# 2. 文本预处理
# -----------------
def clean_text(text):
    text = str(text).lower()  # 转换为小写
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # 去除URL
    text = re.sub(r'[^a-z\s]', '', text)  # 只保留字母和空格
    return text

# 停用词去除
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return words

# 预处理数据
data['selected_text'] = data['selected_text'].apply(clean_text).apply(tokenize_and_remove_stopwords)

# -----------------
# 3. Word2Vec 向量化
# -----------------
# 训练一个 Word2Vec 模型
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

# 提取文本的 Word2Vec 特征
X = np.array([get_word2vec_features(text) for text in data['selected_text']])
y = data['label']

# -----------------
# 4. 数据划分
# -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# -----------------
# 5. Pipeline（Word2Vec + RandomForest）
# -----------------
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))  # 使用随机森林模型
])

# -----------------
# 6. 超参数优化（Grid Search）
# -----------------
param_grid = {
    'rf__n_estimators': [50, 100, 200],  # 设置不同的树数量
    'rf__max_depth': [10, 20,30,40, None],  # 设置树的最大深度
    'rf__min_samples_split': [2, 5, 10]  # 设置分裂一个节点所需的最小样本数
}

# tqdm 进度条包装 GridSearchCV
class TQDMGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, **fit_params):
        n_candidates = len(self.param_grid['rf__n_estimators']) * \
                       len(self.param_grid['rf__max_depth']) * \
                       len(self.param_grid['rf__min_samples_split'])
        n_folds = self.cv
        total = n_candidates * n_folds  # 计算总训练次数

        with tqdm(total=total, desc="Grid Search Progress") as pbar:
            def callback(*args, **kwargs):
                pbar.update(1)

            # 直接调用 GridSearchCV 的 fit 方法，进度条通过 callback 更新
            super().fit(X, y, **fit_params)

            pbar.close()

        return self

grid_search = TQDMGridSearchCV(
    pipeline, param_grid, cv=5, 
    scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'},
    refit='f1_macro', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# 获取最优参数
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# -----------------
# 7. 模型评估
# -----------------
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)

# -----------------
# 8. 记录日志
# -----------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"log_Rf{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")

logging.info("Best Parameters: %s", best_params)
logging.info("Classification Report:\n%s", report)

# 打印输出
print("Best Parameters:", best_params)
print("Classification Report:\n", report)
