import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.StratifierCommonFunctions import compute_distribution_by_sum_perc, compute_distribution_by_sum_counts, \
    compute_target_counts, validate_user_DIM


def generateHistogramPlots(file_name, data_type, df, DIM):
    """
    Genera un unico grafico con una matrice di istogrammi per tutte le dimensioni.

    Args:
        file_name (str): Nome del file di output.
        df (DataFrame): DataFrame contenente i dati.
        DIM (list of lists): Dimensioni e relative classi.
    """

    # Validazione delle dimensioni
    validate_user_DIM(DIM, df)

    # Lista di 10 colori di base
    base_colors = [mcolors.to_rgb(c) for c in list(mcolors.TABLEAU_COLORS.values())]



    # Calcola le percentuali di distribuzione per ogni classe in ogni dimensione
    distr_type=[]
    distr_type.append(("perc",compute_distribution_by_sum_perc(df, DIM)))
    distr_type.append(("counts",compute_distribution_by_sum_counts(df, DIM)))

    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)

    if len(file_name.split("_"))==4:
        name_dataset, method, _, sample_size = file_name.split("_")
        plot_dir = f"plots/{data_type}/{name_dataset}/sample_{sample_size}/{method}"
    else:
        plot_dir = f"plots/{data_type}/{file_name}/original"

    # Creazione della directory di output
    os.makedirs(plot_dir, exist_ok=True)


    # Definizione del layout della matrice
    n_dims = len(DIM)
    n_cols = n_dims  # Numero di colonne nella matrice
    n_rows = 1

    for name_distr,distr in distr_type:
        # Creazione della figura
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6), constrained_layout=True)
        fig.suptitle(f"{file_name}_dimensions_hist_{name_distr}", fontsize=16)
        axes = axes.flatten()  # Appiattisce l'array per un accesso semplice

        # Loop attraverso tutte le dimensioni
        for i, dim_classes in enumerate(DIM):
            dim_name = f"dim{i + 1}"
            classes = list(distr[i].keys())
            counts = list(distr[i].values())

            # Seleziona l'asse corrente
            ax = axes[i]
            ax.set_ylim([0, 1])
            # Creazione dell'istogramma
            x_indices = np.arange(len(classes))
            alphas = np.linspace(0.9, 0.4, len(counts))  # Da 0.4 a 1 in base al numero di barre
            r,g,b= base_colors[i]
            colors = [(r, g, b, alpha) for alpha in alphas]  # Gradiente di 'steelblue'

            # Disegna le barre con il gradiente
            bars = ax.bar(x_indices, counts, color=colors)

            # Titoli, etichette e ticks
            ax.set_title(f"Histogram of assigned classes in {dim_name}")
            ax.set_xticks(x_indices)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.set_ylabel("Perc. of reviews")

            # Aggiungi percentuali sopra ogni barra
            for idx, bar in enumerate(bars):
                pct = counts[idx]*100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),  # Offset sopra la barra
                    f"{pct:.1f}%",  # Percentuale con 1 decimale
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='black'
                )


        plt.tight_layout()
        # Salvataggio del grafico
        plt.savefig(f"{plot_dir}/{file_name}_dimensions_hist_{name_distr}.svg", format="svg")
        plt.close()
        #print(f"Combined dimension histograms saved as '{combined_file}'.")