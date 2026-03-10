import math
import sys

import numpy as np
import pandas as pd
import scipy.special
from networkx.algorithms.hybrid import kl_connected_subgraph
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
from utils.StratifierCommonFunctions import log_progress, kl_divergence
import time


class StratifierKDE:

    def __init__(self, use_relevance_score=False, alpha=0.6, debug=False):
        self.debug = debug
        self.scoring_function = lambda x, y, i: min((alpha + 0.025 * i), 0.7) * x + (1 - min((alpha + 0.025 * i), 0.7)) * (1 - y) if use_relevance_score else x
    def stratify(self, df, DIM, M=10,):

        idx_selected = set()

        #DIM=[DIM[2]]
        kde_dataset = []
        w = []
        bins = np.linspace(0, 1, 20)

        relevance_scores=np.array(df["relevance_score"])

        for x in DIM:
            for y in x:
                #w.append(1 / len(x))
                frequencies, _ = np.histogram(df[y].to_numpy(), bins=bins, density=True)
                kde_dataset.append(frequencies)


        selected_sample_pandas = []
        for i in range(M):
            kl_scores = np.full(len(df), sys.float_info.max)
            max_div = 1e-10
            for idx, row in df.iterrows():
                if idx not in idx_selected:

                    selected_sample_copy = selected_sample_pandas.copy()
                    selected_sample_copy.append(row)
                    sample_df = pd.DataFrame(selected_sample_copy)


                    kde_sample = []
                    for el in DIM:
                        for y in el:
                            v = sample_df[y].to_numpy(),
                            frequencies, _ = np.histogram(v, bins=bins, density=True)
                            kde_sample.append(frequencies)


                    kl_div = 0
                    for j in range(len(kde_dataset)):
                        p = kde_dataset[j]
                        q = kde_sample[j]

                        # Ensure no division by zero (add a small constant to q if needed)
                        q += 1e-10

                        # Compute the KL divergence
                        kl_div +=  kl_divergence(p, q)


                    kl_scores[idx] = kl_div
                    if kl_div > max_div:
                        max_div = kl_div


            scores = self.scoring_function(kl_scores/max_div, relevance_scores, i)
            idx_best = np.argmin(scores)
            selected_sample_pandas.append(df.iloc[idx_best])
            idx_selected.add(idx_best)
            relevance_scores[idx_best] = 0
            log_progress(i, M)

        return pd.DataFrame(selected_sample_pandas)

