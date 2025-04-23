import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


input_file = "youtube_comments_with_category.csv"
output_file = "youtube_comments_with_3sentiment.csv"


df = pd.read_csv(
    input_file,
    encoding='utf-8-sig',
    quotechar='"',
    escapechar='\\',
    on_bad_lines='skip'  
)


model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)


label_map = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}


def classify_sentiment(text):
    try:
        result = classifier(str(text)[:512])[0]  
        label = label_map.get(result['label'], result['label'])
        return pd.Series([label, result['score']])
    except Exception as e:
        return pd.Series(["ERROR", 0.0])


df[['sentiment', 'score']] = df['comment_text'].apply(classify_sentiment)


df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Save as: {output_file}")
