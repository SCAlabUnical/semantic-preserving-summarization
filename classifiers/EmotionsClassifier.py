import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionsClassifier:
    @staticmethod
    def classify(df, column_text="cleaned_text"):
        # =========================
        #  PARTE 2: EMOTION MODEL
        # =========================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
        emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(device)

        emotion_labels = emotion_model.config.id2label

        def predict_emotions(text: str):
            inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = emotion_model(**inputs)
                logits = output.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            return probs

        emotions_name=[]
        # Creiamo le colonne delle emozioni
        for label_id, label_name in emotion_labels.items():
            df[f"em_{label_name}"] = 0.0
            emotions_name.append(f"em_{label_name}")

        # Calcoliamo le probabilità di emozione per ogni riga
        for i, row in df.iterrows():
            text = row[column_text]
            emotion_probs = predict_emotions(text)
            for label_id, label_name in emotion_labels.items():
                df.at[i, f"em_{label_name}"] = float(emotion_probs[label_id])


        return df,emotions_name
