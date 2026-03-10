import os
import random

import pandas as pd
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from stratifiers.StratifierRandom import StratifierRandom
from utils.LLM import ask_gpt
from utils.PreProcesser import clean_text, generate_summary, join_summary


def get_ground_report(df,file_name, prompt, data_type):

    os.makedirs(f"data/{data_type}/ground_truth/report/", exist_ok=True)
    path=f"data/{data_type}/ground_truth/report/{file_name}_report.txt"
    if not os.path.exists(path):
        summary=generate_summary(df, data_type, file_name)
        summary=summary["summary"].tolist()
        report=ask_gpt(prompt+str(summary))

        with open(path, "w") as f:
            f.write(report)

        return report

    with open(path, "r") as f:
        report = f.read()

    return report


def generate_and_evaluate_report(df, sample_df, DIM, stratifier, data_type, file_name, M, words, relevance=False, k=5):
    print("Generate and evaluate report...  ", end="")

    os.makedirs(f"data/{data_type}/result/report/", exist_ok=True)
    report_file = f"data/{data_type}/result/report/{file_name}_report.csv"

    if not os.path.exists(report_file):
        df_report = pd.DataFrame(columns=['stratifier', 'sample', 'words', 'relevance', "rouge1", "rouge2", "rougeL", "embedding_sim",'report'])

    else:
        df_report = pd.read_csv(report_file)




    reviews = []
    for _, row in sample_df.iterrows():
        reviews.append({'review_text': row['summary']})

    # prompt_for_review = f"""You are provided with a list of reviews for a product.
    #            Your task is to write a brief and balanced report about the product that summarizes and captures the meaning of the reviews and covers all the main topics.
    #            The review should be a concise, single paragraph without line breaks or colons.
    #
    #            Reviews: """

    prompt = f"""
You are provided with a list of tweets scraped from the 2024 US election.
Your task is to write a brief and balanced report that summarizes and captures the meaning of the tweets, identifies whether the overall sentiment leans more toward Donald Trump or Kamala Harris, and explains the main topics and reasons behind the support for that candidate.
The report should be a concise, single paragraph without line breaks or colons.

Tweet:"""

    res =  {"embedding_sim":0, "rouge1":0, "rouge2":0, "rougeL":0}
    for i in range(k):


        random.shuffle(reviews)
        if stratifier == "Random":
            sample = StratifierRandom().stratify(df, None, M)
            sample = join_summary(sample,data_type, file_name)
            reviews = []
            for _, row in sample.iterrows():
                reviews.append({'review_text': row['summary']})


        sample_report = ask_gpt(prompt + str(reviews))

        #print("\n\n\n")
        #print(sample_report)
        #print("\n\n\n")

        ground_report = get_ground_report(df, file_name, prompt, data_type)

        cleaned_ground_report = clean_text(ground_report)
        cleaned_sample_report = clean_text(sample_report)

        embedding_sim=compute_embedding_similarity(cleaned_ground_report,cleaned_sample_report)

        rouge1,rouge2,rougeL = compute_rouge_score(cleaned_ground_report,cleaned_sample_report)

        res["embedding_sim"]+=embedding_sim
        res["rouge1"]+=rouge1
        res["rouge2"]+=rouge2
        res["rougeL"]+=rougeL


    ###add row
    update = False
    for idx, row in df_report.iterrows():
        if row[['stratifier', 'sample', 'words', 'relevance']].eq([stratifier, M, words, relevance]).all():
            df_report.loc[idx, "rouge1"] = res["rouge1"]/k
            df_report.loc[idx, "rouge2"] = res["rouge2"]/k
            df_report.loc[idx, "rougeL"] = res["rougeL"]/k
            df_report.loc[idx, "embedding_sim"] = res["embedding_sim"]/k

            df_report.loc[idx, "report"] = sample_report
            update = True

    if not update:
        df_report.loc[len(df_report)] = [stratifier, M, words, relevance, rouge1, rouge2, rougeL, embedding_sim, sample_report]

    df_report.to_csv(report_file, index=False)

    print("\r", end="")


model1=None
def compute_embedding_similarity(ground_report, sample_report):
    global model1

    if model1 is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1 = SentenceTransformer('joe32140/ModernBERT-base-msmarco').to(device)

    embeddings = model1.encode(ground_report, convert_to_tensor=True)
    embedding_report = model1.encode(sample_report, convert_to_tensor=True)
    score=model1.similarity(embeddings, embedding_report)
    return score.item()

def compute_rouge_score(ground_report, sample_report):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_report, sample_report)
    return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure