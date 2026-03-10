from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class SentimentClassifier:
    @staticmethod
    def classify(df, column_text="cleaned_text"):
        # =========================
        #  PARTE 3: SENTIMENT MODEL
        # =========================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)


        def predict_sentiment_triple(text: str):
            inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = sentiment_model(**inputs)
                logits = output.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            p_negative = probs[0]
            p_neutral = probs[1]
            p_positive = probs[2]
            return p_negative, p_neutral, p_positive

        # Creiamo tre colonne nel DataFrame

        df["negative"] = 0.0
        df["neutral"] = 0.0
        df["positive"] = 0.0

        for i, row in df.iterrows():
            text = row[column_text]
            p_neg, p_neu, p_pos = predict_sentiment_triple(text)
            df.at[i, "negative"] = float(p_neg)
            df.at[i, "neutral"] = float(p_neu)
            df.at[i, "positive"] = float(p_pos)


        return df, ["negative","neutral","positive"]