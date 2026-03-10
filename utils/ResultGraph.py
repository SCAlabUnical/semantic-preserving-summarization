import os
from ast import literal_eval

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


plt.rcParams.update({
    "font.size": 12,           # Default font size
    "axes.titlesize": 12,      # Title font size
    "axes.labelsize": 10,      # Axis label font size
    "xtick.labelsize": 10,     # X-tick label size
    "ytick.labelsize": 10,     # Y-tick label size
    "legend.fontsize": 12      # Legend font size
})


def generateResultGraph(file_name,stratifiers, metrics, words=100, relevance=False):
    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)


    plot_dir = f"plots/{file_name}"

    # Creazione della directory di output
    os.makedirs(plot_dir, exist_ok=True)

    df=pd.read_csv(f"data/reports/{file_name}_report.csv",converters={m : literal_eval for m in metrics})
    num_cols = len(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(num_cols * 5, 5))
    fig.suptitle(f"{file_name} - relevance:{str(relevance)}")

    for i,stratifier in enumerate(stratifiers):
        for j,metric in enumerate(metrics):

            ax=axes[j]

            x_data,y_data = get_data(df, stratifier, metric, words, relevance)

            x_data = x_data[:6]
            y_data = y_data[:6]

            # Plot the data on the current subplot
            ax.plot(x_data, y_data,linestyle='--', marker='o', label=stratifier)


            #ax.set_ylim([0, 1])
            # Customize the subplot (optional)
            ax.set_title(f"Metric:{metric}")
            ax.set_xlabel("sample_size")
            ax.set_ylabel("score")

    ax.legend()
    # Adjust the layout so subplots don't overlap
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"plots/{file_name}/{file_name}_result_5_rep_{str(relevance)}.png")
    plt.show()
    plt.close()



def get_data(df, stratifier, metric, words, relevance):

    grouped=df.groupby(["stratifier","words","relevance"])

    data=grouped.get_group((stratifier,words,relevance))
    data_sorted =data.sort_values(by='sample')
    x_data = data_sorted['sample'].values
    y_data = data_sorted[metric].values
    #y_data = [np.mean(list(y)) for y in y_data]

    return x_data, y_data


def get_report_data(df, stratifier, words, metric, relevance):

    grouped=df.groupby(["stratifier","words","relevance"])
    #print(stratifier, words, relevance)
    data=grouped.get_group((stratifier,words,relevance))
    data_sorted =data.sort_values(by='sample')
    x_data = data_sorted['sample'].values
    y_data = data_sorted[metric].values
    #y_data = [np.mean(list(y)) for y in y_data]

    return x_data, y_data

def get_sample_data(df, stratifier, metric, relevance):
        grouped = df.groupby(["stratifier", "relevance"])

        data = grouped.get_group((stratifier, relevance))
        data_sorted = data.sort_values(by='sample')
        x_data = data_sorted['sample'].values
        y_data = data_sorted[metric].values
        #print(y_data)

        y_data = [np.mean(list(y)) for y in y_data]

        return x_data, y_data





def graph_topics_coverage(file_names, data_type, stratifiers, metric, relevance=False):
    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)


    #plot_dir = f"plots/{file_name}"

    # Creazione della directory di output
    #os.makedirs(plot_dir, exist_ok=True)


    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Average Topics Coverage across {len(file_names)} {data_type.capitalize()} items")
    for i,stratifier in enumerate(stratifiers):
            y_data = None
            for file_name in file_names:
                df = pd.read_csv(f"data/{data_type}/result/sample/{file_name}_result.csv",converters={m : literal_eval for m in metrics})
                x_data, y = get_sample_data(df, stratifier, metric, relevance)
                y = np.array(y)
                if y_data is None:
                    y_data = y
                else:
                    y_data += np.array(y)

            y_data = y_data / len(file_names)
            y_data=y_data[:6]
            x_data=x_data[:6]


            print(stratifier, [float(x) for x in y_data])

            plt.ylabel(f"{metric.capitalize()} score")
            plt.xlabel("sample size")
            plt.xticks(x_data)
            # Plot the data on the current subplot
            plt.plot(x_data, y_data, label=stratifier, linestyle='--', marker='o')

            #ax.set_ylim([0, 1])
            # Customize the subplot (optional)

    plt.legend()
    # Adjust the layout so subplots don't overlap
    plt.tight_layout()
    plt.savefig(f"plots/{data_type}/{data_type}_topic_coverage_{metric}.png")
    # Display the plot
    plt.show()
    plt.close()

def graph_report(file_names, data_type, stratifiers, metric, words=150, relevance=False):
    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)


    #plot_dir = f"plots/{file_name}"

    # Creazione della directory di output
    #os.makedirs(plot_dir, exist_ok=True)


    fig = plt.figure(figsize=(7, 6))
    fig.suptitle(f"Average Report Score across {len(file_names)} {data_type.capitalize()} items")
    for i,stratifier in enumerate(stratifiers):
            y_data = None
            for file_name in file_names:
                df = pd.read_csv(f"data/{data_type}/result/report/{file_name}_report.csv", converters={metric: literal_eval})
                x_data, y = get_report_data(df, stratifier, words, metric, relevance)
                y= np.array(y)
                if y_data is None:
                    y_data = y
                else:
                    y_data += np.array(y)

            y_data=y_data/len(file_names)

            x_data = x_data[:6]
            y_data = y_data[:6]


            print(stratifier, [float(x) for x in y_data])
            plt.xticks(x_data)

            # Plot the data on the current subplot
            plt.plot(x_data, y_data, label=stratifier, linestyle='--', marker='o')

            #ax.set_ylim([0, 1])
            plt.ylabel("score")
            plt.xlabel("sample size")

    plt.legend()
    # Adjust the layout so subplots don't overlap
    plt.tight_layout()

    plt.savefig(f"plots/{data_type}/{data_type}_report_similarity_{metric}.png")
    # Display the plot
    plt.show()
    plt.close()


def get_cluster_data(df, stratifier, relevance):

    grouped = df.groupby(["stratifier", "relevance"])
    data = grouped.get_group((stratifier, relevance))
    data_sorted = data.sort_values(by='sample')
    x_data = data_sorted['sample'].values
    y_data = data_sorted['coverage_score'].values

    return x_data, y_data


def graph_cluster(file_names, data_type, stratifiers, relevance=True):
    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)


    #plot_dir = f"plots/{file_name}"

    # Creazione della directory di output
    #os.makedirs(plot_dir, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    fig.suptitle(f"Average Coverage Score across {len(file_names)} {data_type.capitalize()} items")

    for i,stratifier in enumerate(stratifiers):
            y_data = None
            for file_name in file_names:
                df = pd.read_csv(f"data/{data_type}/result/cluster/{file_name}_result.csv")
                x_data, y = get_cluster_data(df, stratifier, relevance)

                y = np.array(y)
                if y_data is None:
                    y_data = y
                else:
                    y_data += np.array(y)


            y_data=y_data/len(file_names)



            # Plot the data on the current subplot
            plt.plot(x_data, y_data, label=stratifier, linestyle='--', marker='o')

            plt.xticks(x_data)

            # Adjust the layout so subplots don't overlap
            plt.tight_layout()

            plt.ylabel("coverage score")
            plt.xlabel("sample size")

            #ax.set_ylim([0.2, 3.4])
            #Customize the subplot (optional)

    plt.legend()
    plt.savefig(f"plots/{data_type}/{data_type}_cluster_coverage_score.png")
    # Display the plot
    plt.show()
    plt.close()

def get_tokens_usage_data(df, stratifier, relevance):

    grouped = df.groupby(["stratifier", "relevance"])
    data = grouped.get_group((stratifier, relevance))
    data_sorted = data.sort_values(by='sample')
    x_data = data_sorted['sample'].values
    y_data = data_sorted['tokens'].values

    return x_data, y_data


def graph_tokens_usage(file_names, data_type, stratifiers,relevance=True):
    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)

    #plot_dir = f"plots/{file_name}"

    # Creazione della directory di output
    #os.makedirs(plot_dir, exist_ok=True)


    fig = plt.figure(figsize=(7, 6))
    fig.suptitle(f"Average Tokens Usage across {len(file_names)} {data_type.capitalize()} items")
    for i,stratifier in enumerate(stratifiers):
            y_data = None
            for file_name in file_names:
                df = pd.read_csv(f"data/{data_type}/result/tokens_usage/{file_name}_result.csv")
                x_data, y = get_tokens_usage_data(df, stratifier, relevance)
                y= np.array(y)
                if y_data is None:
                    y_data = y
                else:
                    y_data += np.array(y)


            y_data=y_data/len(file_names)

            print(stratifier, [float(x) for x in y_data])


            # Plot the data on the current subplot
            plt.plot(x_data, y_data, label=stratifier, linestyle='--', marker='o')

            plt.xticks(x_data)
            plt.ylabel("tokens usage")
            plt.xlabel("sample size")

            # Adjust the layout so subplots don't overlap
            plt.tight_layout()


            #Customize the subplot (optional)

    plt.legend()
    plt.savefig(f"plots/{data_type}/{data_type}_tokens_usage.png")
    # Display the plot
    plt.show()
    plt.close()

def get_ratio_data(df, stratifier,words, metric, relevance):

    grouped = df.groupby(["stratifier", "relevance","words"])
    data = grouped.get_group((stratifier, relevance,words))
    data_sorted = data.sort_values(by='sample')
    x_data = data_sorted['tokens'].values
    y_data = data_sorted[metric].values

    return x_data, y_data

def graph_ratio(file_names, data_type, stratifiers, metric, words=150, relevance=False):
    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)


    #plot_dir = f"plots/{file_name}"

    # Creazione della directory di output
    #os.makedirs(plot_dir, exist_ok=True)


    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Impact of Tokens Usage on Similarity Score across {len(file_names)} {data_type.capitalize()} items")
    for i,stratifier in enumerate(stratifiers):
            y_data = None
            for file_name in file_names:
                df1 = pd.read_csv(f"data/{data_type}/result/report/{file_name}_report.csv", converters={metric: literal_eval})
                df2 = pd.read_csv(f"data/{data_type}/result/tokens_usage/{file_name}_result.csv")
                merged_df = pd.merge(df1, df2, on=["stratifier","sample","relevance"])
                x_data, y = get_ratio_data(merged_df, stratifier, words, metric, relevance)
                y = np.array(y)
                if y_data is None:
                    y_data = y
                else:
                    y_data += np.array(y)

            y_data=y_data/len(file_names)

            x_data = x_data[:6]
            y_data = y_data[:6]




            # Plot the data on the current subplot
            plt.plot(x_data, y_data, label=stratifier, linestyle='--', marker='o')

            #ax.set_ylim([0, 1])
            plt.ylabel("Cosine Similarity")
            plt.xlabel("tokens usage")

    plt.legend()
    # Adjust the layout so subplots don't overlap
    plt.tight_layout()

    plt.savefig(f"plots/{data_type}/{data_type}_result_ratio_{metric}.png")
    # Display the plot
    plt.show()
    plt.close()

if __name__ == "__main__":
    #file_name = "Toothbrush"
    #file_names = ["Toothbrush","Filament", "Bike","EspressoMachine","ApplePen","Thermostat","Monitor","ToothbrushHeads","Helmet","Gloves"]#,"Filament", "Bike", "EspressoMachine","ApplePen","Thermostat", "Monitor","ToothbrushHeads","Helmet","Gloves"
    #file_names = ["014", "036", "019", "061", "082", "104", "133", "135", "144", "147"]
    file_names = ["arizona", "georgia", "michigan", "nevada"]

    data_type = "elezioni_politiche"
    stratifiers=["Knapsack","KullLei","KDE","Relevance","Random"]
    metrics=["f1","precision","recall","jaccard"]

    metric="f1"
    os.chdir('../')


    #graph_topics_coverage(file_names, data_type, stratifiers, metric, relevance=True)

    metric="embedding_sim"
    #graph_report(file_names, data_type, stratifiers, metric, words=150, relevance=True)

    #graph_cluster(file_names, data_type, stratifiers)

    graph_tokens_usage(file_names, data_type, stratifiers)

    #graph_ratio(file_names, data_type, stratifiers, metric,relevance=True)