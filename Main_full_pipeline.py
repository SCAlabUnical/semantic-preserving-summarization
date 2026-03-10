import pandas as pd
import os
from enum import Enum

from classifiers.BertTopic import BertTopic
from classifiers.EmotionsClassifier import EmotionsClassifier
from classifiers.SentimentClassifier import SentimentClassifier
from stratifiers.StratifierKDE import StratifierKDE
from stratifiers.StratifierKnapsack import StratifierKnapsack
from stratifiers.StratifierKullbackLeibler import StratifierKullbackLeibler
from stratifiers.StratifierKullbackLeiblerOpt import StratifierKullbackLeiblerOpt
from stratifiers.StratifierRandom import StratifierRandom
from stratifiers.StratifierRelevance import StratifierRelevance
from utils.DensityGraphCombined import generateDensityPlots
from utils.EvaluateReport import generate_and_evaluate_report
from utils.EvaluateSample import evaluate_sample

from utils.HistogramCombined import generateHistogramPlots
import warnings

from utils.PreProcesser import preprocess, generate_summary, join_summary
from utils.ResultGraph import generateResultGraph
from utils.SIGIR_clusters import generateClusterPlot
from utils.StratifierCommonFunctions import remove_outliers
from utils.TokenCounter import count_tokens

#warnings.filterwarnings('ignore')

relevance=True

class StratifierType(Enum):
    KDE = StratifierKDE(use_relevance_score=relevance, alpha=0.60)
    KullLei = StratifierKullbackLeibler(use_relevance_score=relevance, k=20, alpha=0.60)
    Knapsack = StratifierKnapsack(use_relevance_score=relevance, )
    Random = StratifierRandom()
    Relevance = StratifierRelevance()

    def __str__(self):
        return self.name  # Return the name of the enum member (e.g., 'KDE')




#file_name = "Toothbrush"  ##filename without extension

data_type = "amazon"

field_name={"tripadvisor": ["title","text","rating"], "amazon": ["reviewTitle","reviewDescription","ratingScore"], "elezioni_politiche": ["text"], "reddit": ["Title", "Text"]}


file_names = ["Toothbrush","Filament","Bike","EspressoMachine","ApplePen","Thermostat","Monitor","ToothbrushHeads","Helmet","Gloves"]
#file_names = ["014","036","019","061","082","104","133","135","144","147"]
#file_names = ["user1","user2","user3"]


M_values = [1, 2, 5, 10, 15, 20]
stratifiers = [StratifierType.Knapsack, StratifierType.KullLei,StratifierType.Relevance,StratifierType.KDE,StratifierType.Random]
metrics = ["f1", "precision", "recall", "jaccard"]




for file_name in file_names:
    print("PROCESSING " + file_name)
    df = pd.read_json(f'data/{data_type}/raw/{file_name}.json')

    df = preprocess(df, field_name[data_type])


    generate_summary(df, data_type, file_name)


    print("STARTING CLASSIFIERS...")

    df, topics = BertTopic.classify(df, use_gpt=False)   ## with use_gpt=True make the topic title with gpt
    df, emotions = EmotionsClassifier.classify(df)
    df, sentiments = SentimentClassifier.classify(df)


    print("DONE")

    df = df.drop('cleaned_text', axis=1)
    os.makedirs(f'data/{data_type}/ready/',exist_ok=True)
    df.to_csv(f'data/{data_type}/ready/{file_name}.csv',index=False)

    df = remove_outliers(df)

    # Define DIM to describe dimensions and classes
    DIM = [sentiments, emotions, topics]
    print("STARTING STRATIFIERS...")

    generateHistogramPlots(file_name,data_type, df, DIM)
    generateDensityPlots(file_name, data_type, df, DIM)

    for M in M_values:
        print(f"Sampling with M={M}...")
        for stratifier in stratifiers:
            print(f"    Executing:{stratifier.name}")
            df_sample = stratifier.value.stratify(df, DIM, M)

            df_sample = join_summary(df_sample,data_type, file_name)
            result_file_name = f"{file_name}_{str(stratifier)}_sample_{M}"

            # Ensure the directory exists before saving
            output_dir = f"data/{data_type}/samples/{file_name}"  # Extract the directory from the path
            os.makedirs(output_dir, exist_ok=True)  # Create the directory

            output_path = f"{output_dir}/{result_file_name}.csv"
            df_sample.to_csv(output_path, index=False)

            generateHistogramPlots(result_file_name, data_type, df_sample, DIM)
            generateDensityPlots(result_file_name, data_type, df_sample,DIM)
            generateClusterPlot(result_file_name, data_type, relevance, df, df_sample, topics)

            evaluate_sample(df, df_sample, stratifier.name, data_type, file_name, M, relevance=relevance, k=10)

            count_tokens(result_file_name, data_type, relevance, df_sample, df)

            generate_and_evaluate_report(df, df_sample, DIM, stratifier.name, data_type, file_name, M, words=150, relevance=relevance, k=10)

    print("DONE")