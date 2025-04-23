import torch
from transformers import BertTokenizer
import pandas as pd
from torch.nn import functional as F
from Bert import MyBert
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_path)


model = MyBert(num_labels=3)
model.load_state_dict(torch.load(os.path.join(model_path, "training_args.bin"), map_location=device))
model.to(device)
model.eval()


label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def classify_sentiment(text):
    try:
       
        encoding = tokenizer(str(text), truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

   
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            if isinstance(logits, tuple):  
                logits = logits[1]
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            score = torch.max(probs).item()

        return pd.Series([label_map[pred], score])
    except Exception as e:
        return pd.Series(["ERROR", 0.0])



input_file = "youtube_comments_with_category.csv"
output_file = "youtube_comments_with_3sentiment.csv"

df = pd.read_csv(input_file, encoding='utf-8-sig', quotechar='"', escapechar='\\', on_bad_lines='skip')


df[['sentiment', 'score']] = df['comment_text'].apply(classify_sentiment)


df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Save to  {output_file}")
