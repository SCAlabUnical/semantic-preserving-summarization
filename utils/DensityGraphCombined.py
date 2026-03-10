import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils.StratifierCommonFunctions import validate_user_DIM


def generateDensityPlots(file_name, data_type, df, DIM):
    """
    Genera un unico grafico per ogni dimensione combinando istogrammi e densità.

    Args:
        file_name (str): Nome del file di output.
        df (DataFrame): DataFrame contenente i dati.
        DIM (list of lists): Dimensioni e relative classi.
    """
    # Creazione della directory di output
    os.makedirs("plots", exist_ok=True)


    if len(file_name.split("_")) == 4:
        name_dataset, method, _, sample_size = file_name.split("_")
        plot_dir = f"plots/{data_type}/{name_dataset}/sample_{sample_size}/{method}"
    else:
        plot_dir = f"plots/{data_type}/{file_name}/original"

    # Creazione della directory di output
    os.makedirs(plot_dir, exist_ok=True)

    # Validazione delle dimensioni
    validate_user_DIM(DIM, df)


    # Intervallo standard per l'asse x
    x_range = (0, 1)
    x = np.linspace(x_range[0], x_range[1], 500)
    # Creazione di una matrice di sottografici
    n_cols = max([len(x) for x in DIM])
    n_rows = len(DIM)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4*n_rows), constrained_layout=True)
    fig.suptitle(f"{file_name}_dim_combined", fontsize=16)
    first=True
    # Loop attraverso tutte le dimensioni
    for dim_index, dim_classes in enumerate(DIM):
        dim_name = f"Dimension {dim_index + 1}"


        # Pre-calcolo dei limiti massimi per gli assi y
        max_counts = 0
        max_density = 0
        for class_name in dim_classes:
            class_probs = df[class_name].values
            kde = gaussian_kde(class_probs)
            density = kde(x)
            max_counts = max(max_counts, np.histogram(class_probs, bins=20)[0].max())
            max_density = max(max_density, density.max())



        for i, class_name in enumerate(dim_classes):
            class_probs = df[class_name].values
            kde = gaussian_kde(class_probs)
            density = kde(x)

            ax = axes[dim_index][i]

            # Istogramma
            ax.hist(class_probs, bins=20, range=x_range, color='lightgray', edgecolor='black', alpha=0.5, label='Histogram (Counts) 'if first else "")
            ax.set_ylim(0, max_counts)



            ax.set_xlabel(f'Probability {class_name}')
            ax.set_ylabel('Counts')

            # Densità
            ax2 = ax.twinx()
            ax2.plot(x, density, color='blue', linewidth=2, label='Density' if first else "")
            ax2.set_ylabel('Density', color='blue')
            ax2.set_ylim(0, max_density)

            # Titolo del sottografico
            ax.set_title(f"Distribution for {class_name} in {dim_name}")
            ax.grid(True)
            fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
            first=False

        # Disabilita gli assi inutilizzati (se ci sono)
        for j in range(len(DIM[dim_index]),n_cols):
            axes[dim_index][j].axis('off')

    # Salva il grafico combinato
    fig.tight_layout()  # Adjust layout for clarity
    plt.savefig(f"{plot_dir}/{file_name}_dim_combined.svg", format="svg")
    plt.close()

