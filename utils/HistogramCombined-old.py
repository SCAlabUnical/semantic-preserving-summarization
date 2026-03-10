import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def combine_images_horizontally(image_files, output_file):
    """
    Combines multiple images horizontally into a single image.
    """
    images = [Image.open(img) for img in image_files]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    combined_image = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    combined_image.save(output_file)


# Validate if all column names in user_DIM exist in df_full
def validate_user_DIM(user_DIM, df_full):
    """
    Checks if all column names in user_DIM exist in df_full.

    Args:
        user_DIM (list of lists): Dimensions and associated classes.
        df_full (DataFrame): Full data.

    Returns:
        bool: True if all columns are valid, False otherwise.
        list: Missing column names.
    """
    # Flatten user_DIM to get a single list of column names
    all_columns = [col for dim in user_DIM for col in dim]

    # Find missing columns
    missing_columns = [col for col in all_columns if col not in df_full.columns]

    if missing_columns:
        print("Validation failed. Missing columns:", missing_columns)
        return False, missing_columns
    else:
        print("Validation successful. All columns in user_DIM are present.")
        return True, []


def generateHistogramPlots(file_name, df, DIM):
    # Validate user_DIM
    is_valid, missing_columns = validate_user_DIM(DIM, df)

    # 1) Assign rows to the class with the highest value in each dimension
    assigned_classes_per_dim = []
    for i, dim_classes in enumerate(DIM):
        dim_name = f"dim{i+1}"
        assign_col = f"assigned_{dim_name}"

        valid_cols = [c for c in dim_classes if c in df.columns]

        sub_probs = df[valid_cols]
        assigned_class_series = sub_probs.idxmax(axis=1)
        df[assign_col] = assigned_class_series
        assigned_classes_per_dim.append(assign_col)

    # 2) Count rows assigned to each class in each dimension
    dimension_counts = {}
    overall_max_count = 0
    for i, dim_classes in enumerate(DIM):
        dim_name = f"dim{i+1}"
        assign_col = f"assigned_{dim_name}"
        class_count_map = df[assign_col].value_counts().to_dict()

        counts_in_order = [class_count_map.get(c, 0) for c in dim_classes]
        dimension_counts[dim_name] = (dim_classes, counts_in_order)

        local_max = max(counts_in_order) if counts_in_order else 0
        overall_max_count = max(overall_max_count, local_max)

    # 3) Plot one bar chart per dimension
    os.makedirs("plots", exist_ok=True)

    dimension_plot_files = []
    for i, dim_classes in enumerate(DIM):
        dim_name = f"dim{i+1}"
        classes, counts = dimension_counts[dim_name]

        # Fixed figure size and alignment
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        x_indices = np.arange(len(classes))
        bars = ax.bar(x_indices, counts, color='steelblue')

        # Title, ticks, labels
        ax.set_title(f"Histogram of assigned classes in {dim_name}", pad=10)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel("Number of Reviews")
        ax.set_ylim([0, overall_max_count * 1.1])  # Consistent y-scale

        # Add % labels on bars
        total_in_dim = sum(counts)
        for idx, bar in enumerate(bars):
            count_val = counts[idx]
            pct = (count_val / total_in_dim) * 100 if total_in_dim > 0 else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (overall_max_count * 0.01),
                f"{pct:.1f}%",
                ha='center',
                va='bottom',
                fontsize=8,
                color='black'
            )

        # Adjust layout and save plot
        plt.subplots_adjust(bottom=0.2)  # Ensure space for labels
        plot_name = f"plots/{file_name}_part_{i}.png"
        plt.savefig(plot_name, bbox_inches="tight", dpi=300, pad_inches=0.2)
        plt.close()
        dimension_plot_files.append(plot_name)

    # 4) Combine histograms horizontally
    combined_file = f"plots/{file_name}_dimensions_hist.png"
    combine_images_horizontally(dimension_plot_files, combined_file)
    print(f"Combined dimension histogram: {combined_file}")



#at the end dlete all intermediate files
plots_folder = "plots"
for filename in os.listdir(plots_folder):
    if "_part" in filename:
        file_path = os.path.join(plots_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            #print(f"Deleted: {file_path}")