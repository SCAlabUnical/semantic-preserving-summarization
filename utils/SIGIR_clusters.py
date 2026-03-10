import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import KDTree

from stratifiers.StratifierRandom import StratifierRandom
from stratifiers.StratifierRandomGpt import StratifierRandomGpt


def generateClusterPlot(file_name, data_type, relevance, df, sample_df, topics):
    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)
    name_dataset, method, _, sample_size = file_name.split("_")
    sample_size = int(sample_size)

    plot_dir = f"plots/{data_type}/{name_dataset}/sample_{sample_size}/{method}"


    # Creazione della directory di output
    os.makedirs(plot_dir, exist_ok=True)



    # Set global font size
    plt.rcParams.update({'font.size': 16})  # Adjust this value to increase/decrease font size globally

    # Define marker types and colors for each label
    #marker_styles = ['o', 's', 'D', '^', 'v', '*', '>']  # Different markers for labels
    marker_styles = ['o','s','D']
    #unique_labels = merged_data_df['topicID'].unique()
    marker_mapping = {label: marker_styles[i % len(marker_styles)] for i, label in enumerate(topics)}

    # Define colors
    #colors = list(plt.cm.tab10(range(len(topics))))  # Convert to a list
    colors = ['blue', 'orange', 'green', 'red', 'purple',
              'brown', 'pink', 'gray', 'olive', 'cyan',
              'yellow', 'teal', 'gold', 'magenta', 'indigo']
    colors=colors[:len(topics)]


    # Plotting the data
    plt.figure(figsize=(17, 10))

    for i,(label, color) in enumerate(zip(topics, colors)):
        data_subset = df[df['topicID'] == i]
        marker = marker_mapping[label]
        if label!=-1:
                plt.scatter(data_subset['UMAP_1'], data_subset['UMAP_2'], label=label, c=[color], marker=marker, edgecolor='white', s=100)



    plt.scatter(sample_df['UMAP_1'], sample_df['UMAP_2'], c="black", marker="x", label="Selected reviews",  edgecolor='white', s=400)



    # Compute Coverage Score (Average Nearest Neighbor Distance)
    def coverage_score(set_A, set_B):
        tree_B = KDTree(set_B)
        distances, _ = tree_B.query(set_A)

        mean_distance = np.mean(distances)
        max_distance = np.max(distances)

        return 1 - (mean_distance / max_distance) if max_distance > 0 else 1

    set_A = df[["UMAP_1", "UMAP_2"]]
    set_B = sample_df[["UMAP_1", "UMAP_2"]]

    if method =="Random":
        #only for random
        score=0
        t=100
        for i in range(t):
            sample_df = StratifierRandom().stratify(df,None,sample_size)
            set_B = sample_df[["UMAP_1", "UMAP_2"]]
            score += coverage_score(set_A, set_B)
        score/=t
    else:

        score = coverage_score(set_A, set_B)



    # Adding axis labels and legend
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.title(f"{file_name} - Coverage Score: {score:.3f}", fontsize=16)
    plt.legend(title="Legend", fontsize=12, title_fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1),
               handleheight=1.5,  # Adjust vertical space between marker symbols
               labelspacing=1.5)   # Adjust vertical space between labels)
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig( f"{plot_dir}/{file_name}_clusters.svg", format="svg")

    plt.show()
    plt.close()

    save_cluster_coverage_score(name_dataset, data_type, method, sample_size, relevance, score)

def save_cluster_coverage_score(name_dataset, data_type,stratifier,M,relevance,score):

    os.makedirs(f"data/{data_type}/result/cluster", exist_ok=True)
    cluster_file=f"data/{data_type}/result/cluster/{name_dataset}_result.csv"

    if not os.path.exists(cluster_file):
        df_cluster = pd.DataFrame(columns=['stratifier', 'sample', 'relevance', "coverage_score"])

    else:
        df_cluster = pd.read_csv(cluster_file)



    ###add row
    update = False
    for idx, row in df_cluster.iterrows():
        if row[['stratifier', 'sample', 'relevance']].eq([stratifier, M, relevance]).all():
            df_cluster.loc[idx, "coverage_score"] = score
            update = True

    if not update:
        df_cluster.loc[len(df_cluster)] = [stratifier, M, relevance, score]

    df_cluster.to_csv(cluster_file, index=False)
