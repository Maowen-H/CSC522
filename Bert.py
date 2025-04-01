import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, Trainer, TrainingArguments, BertModel
from datasets import Dataset
import pandas as pd
import datetime
import os
import logging
import numpy as np
import torch.nn.functional as F

# -----------------
# 自定义BERT模型
# -----------------
class MyBert(nn.Module):
    def __init__(self, num_labels=3):
        super(MyBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = outputs[0]  
        pooled_output = hidden_state[:, 0]  

        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits  # 返回损失和logits
        return logits  # 只返回logits

# -----------------
# 日志设置
# -----------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"log_Bert{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")

# -----------------
# 加载数据并预处理
# -----------------
data = pd.read_csv("./datasets/train.csv")
data = data[['selected_text', 'sentiment']]

# **三分类**: 将情感标签转换为 0, 1, 2
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['label'] = data['sentiment'].map(label_mapping)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['selected_text'], data['label'], test_size=0.3, random_state=1234)
X_train = X_train.astype(str)
print(X_train.dtype)
print(X_train.head())

# -----------------
# BERT Tokenizer
# -----------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['selected_text'], padding="max_length", truncation=True, max_length=512)

# 将数据转换为Dataset格式
train_dataset = Dataset.from_pandas(pd.DataFrame({'selected_text': X_train, 'label': y_train}))
test_dataset = Dataset.from_pandas(pd.DataFrame({'selected_text': X_test, 'label': y_test}))

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# -----------------
# 加载BERT模型
# -----------------
model = MyBert(num_labels=3)
print(model.forward.__code__.co_varnames)

# -----------------
# 定义训练参数
# -----------------
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=50,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=16,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
    evaluation_strategy="epoch",     
    save_strategy="epoch",           
)

# -----------------
# 使用Trainer进行训练和评估
# -----------------
# def compute_metrics(p):
#     preds = np.argmax(p.predictions, axis=-1)  # 获取最大概率的标签
#     labels = p.label_ids
#     accuracy = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1
#     }
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# -----------------
# 开始训练
# -----------------
logging.info("Training started.")
trainer.train()

# -----------------
# 评估模型
# -----------------
logging.info("Evaluating the model.")
results = trainer.evaluate()

logging.info(f"Evaluation Results: {results}")

# -----------------
# 输出评估结果
# -----------------
print("Evaluation Results:")
print(f"Accuracy: {results['eval_accuracy']}")
print(f"Precision (Macro): {results['eval_precision']}")
print(f"Recall (Macro): {results['eval_recall']}")
print(f"F1 Score (Macro): {results['eval_f1']}")
