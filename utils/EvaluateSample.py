import json
import os
import random

import nltk
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from constants import API_KEY
from stratifiers.StratifierRandom import StratifierRandom
from utils.LLM import ask_gpt
from utils.PreProcesser import  generate_summary

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")





def evaluate_sample(df, sample_df, stratifier, data_type, file_name, M, relevance=False, k=10):
    print("Evaluate sample...  ",end="")

    os.makedirs(f"data/{data_type}/result/sample", exist_ok=True)
    result_file = f"data/{data_type}/result/sample/{file_name}_result.csv"

    if not os.path.exists(result_file):
        df_result=pd.DataFrame( columns=['stratifier', 'sample', 'relevance','f1', "precision","recall", "jaccard"])

    else:
        df_result=pd.read_csv(result_file)



    reviews = []
    for _, row in sample_df.iterrows():
        reviews.append({'review_text':row["review_text"]})



    k = 10

    result={"f1":[],"precision":[],"recall":[],"jaccard":[] }


    for _ in range(k):
        random.shuffle(reviews)

        if stratifier == "Random":
            sample = StratifierRandom().stratify(df, None, M)
            reviews = []
            for _, row in sample.iterrows():
                reviews.append({'review_text':row["review_text"]})
        #print(reviews)

        summary = generate_summary(df,data_type, file_name)
        summary = summary["summary"].tolist()

        f1,precision,recall,jaccard = compute_gpt_scores(file_name, data_type, reviews, summary)

        result["f1"].append(f1)
        result["precision"].append(precision)
        result["recall"].append(recall)
        result["jaccard"].append(jaccard)


    ###add row
    update = False
    for idx, row in df_result.iterrows():
        if row[['stratifier','sample','relevance']].eq([stratifier, M, relevance]).all():
            for metric in result:
                df_result[metric].iat[idx] = result[metric]
            update = True


    if not update:
        df_result.loc[len(df_result)] = [stratifier, M, relevance, result['f1'], result['precision'], result['recall'], result['jaccard']]

    df_result.to_csv(result_file, index=False)

    global topics_embeddings
    topics_embeddings = None
    print()

    #print("\r", end="")


topics_embeddings=None
def compute_gpt_scores(file_name,data_type, sample_reviews, summary):
    global topics_embeddings

    file_path = f"data/{data_type}/ground_truth/topic/{file_name}.json"
    os.makedirs(f"data/{data_type}/ground_truth/topic/", exist_ok=True)

    if os.path.exists(file_path):
        topics = json.load(open(file_path))
    else:
        topics = extract_topics(summary)
        with open(file_path, "w") as outfile:
            json.dump(topics, outfile, indent=4)


    topics_sample = extract_topics(sample_reviews)
    #print(sample_reviews)

    #print()
    #print("Real topics:",topics)
    print("Sample Topics:",topics_sample)
    #print()


    if topics_embeddings is None: topics_embeddings=get_topics_embeddings(topics)
    topics_sample_embeddings=get_topics_embeddings(topics_sample)

    f1, precision, recall, jaccard = compute_metrics(topics, topics_sample, topics_embeddings,topics_sample_embeddings)

    print(f"f1:{f1:.2f} - jaccard:{jaccard:.2f} /", end=" ")

    return f1,precision,recall,jaccard


def compute_metrics(topics, topics_sample, topics_embeddings, topics_sample_embeddings):
    jaccard = 0
    f1 = 0
    precision_t = 0
    recall_t = 0
    matching = {}
    # print("\n\nTOPIC MATCHING")
    for topic_type in topics_embeddings:
        topics_sample_size = len(topics_sample[topic_type])
        topics_size = len(topics[topic_type])
        matching[topic_type] = 0
        try:
            cos_sim_matrix = util.cos_sim(topics_embeddings[topic_type], topics_sample_embeddings[topic_type])
        except:
            continue

        while torch.max(cos_sim_matrix) > 0.50:
            arg_max = torch.argmax(cos_sim_matrix).item()
            x, y = arg_max // cos_sim_matrix.size(1), arg_max % cos_sim_matrix.size(1)
            #print(topics[topic_type][x], " <-> ", topics_sample[topic_type][y], "   ", torch.max(cos_sim_matrix).item())
            cos_sim_matrix[x, :] = -1
            cos_sim_matrix[:, y] = -1
            matching[topic_type] += 1

        recall = matching[topic_type] / topics_size
        try:
            precision = matching[topic_type] / topics_sample_size
            f1 += 2 * (recall * precision) / (recall + precision)
        except:
            f1 += recall
            precision = 0

        jaccard += matching[topic_type] / (topics_size + topics_sample_size - matching[topic_type])
        precision_t += precision
        recall_t += recall

    #print("##################################################################\n\n")

    f1 = f1 / len(topics)
    jaccard = jaccard / len(topics)
    precision_t = precision_t / len(topics)
    recall_t = recall_t / len(topics)

    return f1, precision_t, recall_t, jaccard

model = None
def get_topics_embeddings(topics):
    global model
    topics_embeddings={}

    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for topic_type in topics:
        topics_embeddings[topic_type]=model.encode(topics[topic_type],convert_to_tensor=True)

    return topics_embeddings



def extract_topics(reviews):
    #print("\nstart 4o-mini for topic extraction")
    # prompt_for_products = """I have a collection of customer reviews about a product. Your task is to analyze these reviews and extract the main and most general topics that characterize the
    #             product. Identify the most general recurring themes in the reviews (e.g., overall
    #             product quality, long-term durability, user-friendly design, affordability and pricing,
    #             products problems etc.). Each topic should be a short but clear phrase that is self explanatory rather than a single word. Group the topics into positive aspects (those
    #             that receive mostly favorable reviews), neutral aspect (those that received contradictory reviews), negative aspects (those that are predominantly criticized in the reviews). Provide the output as a Python dict containing
    #             3 lists the extracted topic names as strings. A topic that is inside a list can’t be also in the other.
    #             Format the output as follows:
    #             {
    #             ”positive_topics”: [”Product build quality”, ”Ease of assembly and setup”, ...]
    #             "neutral_topics" : [”Battery life and charging speed”, ...]
    #             ”negative_topics”: [”short duration”, ”difficult to use”, ”slow and unresponsive”, ...]
    #             }
    #             Only return the dictionary, with no extra text.
    #
    #             Reviews: """

   #  prompt = """
   #  I have a collection of tweets about a 2024 US election. Your task is to analyze these tweets and extract the main and most general topics that characterize the
   #  election. Identify the most general recurring themes in the tweet. Each topic should be a short but clear phrase that is self explanatory rather than a single word. Group the topics into topics related to Donald Trump and topics related to Khamala Harris.
   #  Provide the output as a Python dict containing 2 lists the extracted topic names as strings. A topic that is inside a list can’t be also in the other.
   #  Format the output as follows:
   #  {
   #      ”khamala_topics”: [...]
   #      ”trump_topics”: [...]
   #  }
   #  Only return the dictionary, with no extra text.
   #
   # Tweets: """

    prompt = """
     I have a collection of reddit post about depressed people. Your task is to analyze these post and extract the main and most general topics that characterize the
     depression. Identify the most general recurring themes in the posts. Each topic should be a short but clear phrase that is self explanatory rather than a single word. Group the topics into topics that show the depression and topics of no-depression.
     Provide the output as a Python dict containing 2 lists the extracted topic names as strings. A topic that is inside a list can’t be also in the other.
     Format the output as follows:
     {
         ”depression_topics”: [...]
         ”no_depression_topics”: [...]
     }
     Only return the dictionary, with no extra text.

    Posts: """

    topics = ask_gpt(prompt=prompt+str(reviews))
    try:
        topics = eval(topics)
    except:
        topics = eval(topics.replace("```python", "").replace("```", ""))

    return topics
