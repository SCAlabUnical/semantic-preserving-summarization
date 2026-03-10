import torch
from sentence_transformers import SentenceTransformer
from torch import nn


# Modello di sentence embedding
class TextScoringModel(nn.Module):
    def __init__(self, embedding_dim):
        super(TextScoringModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

    def forward(self, x):
        return self.model(x)



class RepresentativeClassifier:
    @staticmethod
    def classify(df):
        transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        embedding_dim = transformer_model.get_sentence_embedding_dimension()
        model = TextScoringModel(embedding_dim)
        model.load_state_dict(torch.load("classifiers/model/representative_model.pt", weights_only=True))
        model.eval()
        sentences = df["cleaned_text"].to_list()
        embeddings = transformer_model.encode(sentences, convert_to_tensor=True)
        scores = model(embeddings)

        df["representative_score"] = scores.detach().cpu().numpy()

        return df