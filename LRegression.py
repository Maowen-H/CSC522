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


# -----------------
# 导入逻辑回归模型
from sklearn.linear_model import LogisticRegression

# -----------------
# 1. 读取数据
data = pd.read_csv("./datasets/train.csv")
data = data[['selected_text', 'sentiment']]
data = data.dropna()

# **三分类**: 将情感标签转换为 0, 1, 2
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['label'] = data['sentiment'].map(label_mapping)




# -----------------
# 3. 数据划分
X_train, X_test, y_train, y_test = train_test_split(data['selected_text'], data['label'], test_size=0.3, random_state=1234)
# 确保数据正常
print("训练集样本数:", len(X_train))
print("测试集样本数:", len(X_test))
print("类别分布:\n", data['label'].value_counts())

# -----------------
# 4. Pipeline（TF-IDF + 逻辑回归）
pipeline = Pipeline([
   ('tfidf', TfidfVectorizer(stop_words='english')),  # 使用TF-IDF进行特征提取
    ('lr', LogisticRegression(max_iter=1000, random_state=42))  # 逻辑回归模型
])

# -----------------
# 5. 超参数优化（Grid Search）
param_grid = {
    'lr__C': [0.1, 1, 10,100],  # 逻辑回归的正则化参数C
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # 使用1-gram和2-gram
    'tfidf__max_features': [5000, 10000]  # TF-IDF的最大特征数
}

# tqdm 进度条包装 GridSearchCV
class TQDMGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, **fit_params):
        # 计算网格搜索的候选数量
        n_candidates = len(self.param_grid['lr__C']) * \
                       len(self.param_grid['tfidf__ngram_range']) * \
                       len(self.param_grid['tfidf__max_features'])
        n_folds = self.cv
        total = n_candidates * n_folds  # 计算总训练次数

        # 设置进度条
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
# 6. 模型评估
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)

# -----------------
# 7. 记录日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"log_LR{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")

logging.info("Best Parameters: %s", best_params)
logging.info("Classification Report:\n%s", report)

# 打印输出
print("Best Parameters:", best_params)
print("Classification Report:\n", report)
