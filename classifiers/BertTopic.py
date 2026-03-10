import numpy as np
import openai
import pandas as pd
import torch
from bertopic.backend import OpenAIBackend
from bertopic.representation import MaximalMarginalRelevance
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

from constants import API_KEY


class BertTopic:


    @staticmethod
    def classify(df, column_text="cleaned_text", use_gpt=False):
        # ---------------------
        # PARAMETRI E MODELLI
        # ---------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Devices used:", device)

        # Quanti vicini considerare in UMAP?
        num_neighbors = 20
        # Dimensione minima del cluster HDBSCAN
        min_clusters = 22
        # Semina di random per UMAP (per replicabilità)
        seed_val = 42

        top_n_words=10


        umap_model = UMAP(
            n_neighbors=num_neighbors,
            n_components=5,
            metric='cosine',
            min_dist=0.05,
            random_state=seed_val
        )


        # 3) HDBSCAN per clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_clusters,
            min_samples=5,  # Conservative clustering
            cluster_selection_epsilon=0.05,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
        )

        # 4) Vectorizer (CountVectorizer)
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1,2))

        # 5) ClassTfidfTransformer
        ctfidf_model = ClassTfidfTransformer(bm25_weighting= True, reduce_frequent_words=True)

        sentence_model = SentenceTransformer("joe32140/ModernBERT-base-msmarco").to(device) #  joe32140/ModernBERT-base-msmarco 20 0.02 0.05   // sentence-transformers/all-MiniLM-L6-v2  25 0.03 0.1  //

        client = openai.OpenAI(api_key=API_KEY)
        sentence_model = OpenAIBackend(client, "text-embedding-ada-002")

        representation_model = MaximalMarginalRelevance(diversity=0.2)

        # Istanziamo il modello di topic
        topic_model = BERTopic(
            umap_model=umap_model,  # Step 2 - Riduzione dimensionale
            hdbscan_model=hdbscan_model,  # Step 3 - Clustering
            vectorizer_model=vectorizer_model,  # Step 4 - Tokenizzazione
            ctfidf_model=ctfidf_model,  # Step 5 - c-TFIDF
            calculate_probabilities=True,
            top_n_words=top_n_words,
            representation_model=representation_model,
            #nr_topics=num_topic,
            verbose=True,
        )


        reviews = df[column_text].tolist()

        # Fit_transform del topic model
        embeddings = sentence_model.embed(reviews, verbose=True)   # embed if openAI embedding / encode if Bert embedding
        topics, probs = topic_model.fit_transform(reviews, embeddings)


        # -------------------------------------------
        # Ottenere le Top Words per ciascun topic
        # -------------------------------------------
        # Ricaviamo tutti i topic unici (potrebbe esserci anche -1 per gli outlier)
        unique_topics = set(topics)

        # print frequenza dei topic
        topics_freq = topic_model.get_topic_freq()
        print(topics_freq)

        df['topicID']=topics

        def create_titles_with_gpt(topics_list):
            client = OpenAI(api_key=API_KEY)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": """Your task is to generate a concise title for each list based on the tokens it contains.
                                        The title should: be a maximum of 2-4 words, accurately represent the overall theme or idea of the tokens in the list, the output must be only the titles in a python list.
                                        Output example: ["Title1","Title2","Title 3"]
                                        
                                        Topic list:""" + str(topics_list),
                    }
                ],
                model="gpt-4o",
                temperature=0
            )
            response = chat_completion.choices[0].message.content
            start=response.find("[")
            end=response.rfind("]")

            title_list = eval(response[start:end+1])

            return title_list

        if use_gpt:
            topic_info = topic_model.get_topic_info()
            topic_keywords = {row["Topic"]: row["Representation"] for _, row in topic_info.iterrows()}

            if -1 in topic_keywords: topic_keywords.pop(-1)
            topics_name = create_titles_with_gpt(list(topic_keywords.values()))

        else:
            if -1 in unique_topics: unique_topics.remove(-1)
            topics_name = [f"topic_{t_id}" for t_id in unique_topics]


        print(topics_name)


        ## fill df with probs

        df_topics = []
        for i, row_probs in enumerate(probs):
            doc_info = {}
            # Per ogni topic t (0, 1, 2, ..., -1 se esiste)
            # Nota: row_probs è un array che di solito va da 0 a (n_topic - 1).
            # Se c'è -1, BERTopic potrebbe inserirlo come ultima posizione.
            # Quindi iteriamo su range(len(row_probs)).

            for t_id in range(len(row_probs)):
                # La probabilità del topic t_id per questo documento
                p = float(row_probs[t_id])
                doc_info[topics_name[t_id]] = p

            df_topics.append(doc_info)

        df_topics = pd.DataFrame(df_topics)

        # Concateniamo le nuove colonne (tutti i topic) al DataFrame originale
        df = pd.concat([df.reset_index(drop=True), df_topics.reset_index(drop=True)], axis=1)



        ## generate x,y
        # 2. Use UMAP to reduce dimensionality to 2D for visualization
        umap_model = UMAP(n_neighbors=num_neighbors, n_components=2, min_dist=0.01, metric='cosine',
                          random_state=seed_val)
        umap_embeddings = umap_model.fit_transform(embeddings)
        # Create a DataFrame for visualization
        plot_df = pd.DataFrame(umap_embeddings, columns=["UMAP_1", "UMAP_2"])
        df = pd.concat([df.reset_index(drop=True), plot_df.reset_index(drop=True)], axis=1)


        #################################################

        # Get the top words for each topic
        topic_words = topic_model.get_topics()

        # Define k (number of top words)
        k = 5

        # Extract top-k words for each topic ordered by score
        top_k_words = {
            topic: sorted(words_scores, key=lambda x: x[1], reverse=True)[:k]
            for topic, words_scores in topic_words.items()
        }

        if -1 in top_k_words: top_k_words.pop(-1)

        # Print results
        for topic, words in top_k_words.items():
            print(f"Topic {topic}: {[word for word, score in words]}")


        words = set([word for _, words in top_k_words.items() for word in words])
        vote = {
            1: "terrible",
            2: "poor",
            3: "average",
            4: "good",
            5: "excellent",
        }

        for v in vote.values():
            if v in words:
                words.remove(v)

        df["relevance_score"] = df[column_text].apply(lambda x: get_score(x,words))
        df["relevance_score"] /= df["relevance_score"].max()


        #print(df["representative_score"])

        return df, topics_name

def get_score(x, words):
    score = 0
    unique, counts = np.unique(x.split(), return_counts=True)
    counter=dict(zip(unique, counts))

    for word,_ in words:
        if word in counter:
            score+=counter[word]

    return score