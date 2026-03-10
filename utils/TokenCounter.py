import os

import pandas as pd
import tiktoken

from stratifiers.StratifierRandom import StratifierRandom
from utils.PreProcesser import join_summary


def count_tokens(file_name, data_type, relevance, df_sample,df, model="gpt-4o-mini", summary=False):
    name_dataset, method, _, sample_size = file_name.split("_")
    sample_size = int(sample_size)
    encoding = tiktoken.encoding_for_model(model)
    tokens_size = 0
    if method == "Random":
        avg_size=100
        for k in range(avg_size):
            sample = StratifierRandom().stratify(df, None, sample_size)
            sample = join_summary(sample, data_type, name_dataset)
            for _, row in sample.iterrows():
                tokens_size += len(encoding.encode(row["summary"]))

        tokens_size = tokens_size / avg_size

    else:
        for _, row in df_sample.iterrows():
            tokens_size+=len(encoding.encode(row["summary"]))

    save_token_size(name_dataset, data_type, method, sample_size, relevance, tokens_size)



def save_token_size(name_dataset, data_type, stratifier, M, relevance, tokens_size):

    os.makedirs(f"data/{data_type}/result/tokens_usage", exist_ok=True)
    tokens_usage_file=f"data/{data_type}/result/tokens_usage/{name_dataset}_result.csv"

    if not os.path.exists(tokens_usage_file):
        df_cluster = pd.DataFrame(columns=['stratifier', 'sample', 'relevance', "tokens"])

    else:
        df_cluster = pd.read_csv(tokens_usage_file)



    ###add row
    update = False
    for idx, row in df_cluster.iterrows():
        if row[['stratifier', 'sample', 'relevance']].eq([stratifier, M, relevance]).all():
            df_cluster.loc[idx, "tokens"] = tokens_size
            update = True

    if not update:
        df_cluster.loc[len(df_cluster)] = [stratifier, M, relevance, tokens_size]

    df_cluster.to_csv(tokens_usage_file, index=False)