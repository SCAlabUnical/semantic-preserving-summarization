import scipy

def compute_distribution_by_sum_perc(df, DIM):
    distribution_percentages = []

    for dim_index, classes in enumerate(DIM):
        # Calculate the total sum for all classes in this dimension
        total_sum = df[classes].sum().sum()

        # Calculate and store the distribution for each class
        distribution_percentages.append({cls: df[cls].sum() / total_sum for cls in classes})

    return distribution_percentages


def compute_distribution_by_sum_counts(df, DIM):
    distributions_count = []  # Store distributions for each dimension

    for dim_index, classes in enumerate(DIM):

        # Find the index of the class with the highest probability in each row for this dimension
        highest_class_counts = df[classes].idxmax(axis=1).value_counts()

        # Calculate counts and distribution for each class
        counts = [highest_class_counts.get(col, 0) for col in classes]
        total = sum(counts)
        distribution = {classes[i]: counts[i] / total for i in range(len(classes))}

        # Store the distribution for the current dimension
        distributions_count.append(distribution)

    return distributions_count

def print_distribution(distribution):
    for i, d in enumerate(distribution):
        # Print the distribution for the current dimension
        print(f"dim{i} distribution:", d)
    print("\n")


def compute_target_counts(distribution_percentages, DIM, M):
    """
    Calculate target counts for each class, ensuring the sum of target counts per dimension is exactly M.
    """
    target_counts = {}

    for dim_index, classes in enumerate(DIM):

        # Step 1: Calculate preliminary target counts as integers
        preliminary_counts = {cls: distribution_percentages[dim_index][cls] * M for cls in classes}
        int_counts = {cls: int(count) for cls, count in preliminary_counts.items()}

        # Step 2: Calculate remaining counts needed to reach M
        total_int_counts = sum(int_counts.values())
        remainder = M - total_int_counts

        # Step 3: Sort classes by the fractional part of their preliminary counts
        fractional_parts = {cls: preliminary_counts[cls] - int_counts[cls] for cls in classes}
        sorted_classes = sorted(fractional_parts, key=fractional_parts.get, reverse=True)

        # Step 4: Distribute the remainder to classes with the largest fractional parts
        for i in range(remainder):
            int_counts[sorted_classes[i]] += 1

        # Store the adjusted counts for the current dimension
        target_counts.update(int_counts)

    return target_counts

def kl_divergence(a, b):
    #return sum(a[i] * np.log(a[i] / b[i]) for i in range(len(a)))
    return sum(scipy.special.kl_div(a,b))


def validate_user_DIM(user_DIM, df_full):
    """
    Verifica se tutte le colonne in user_DIM esistono nel DataFrame.

    Args:
        user_DIM (list of lists): Dimensioni e relative classi.
        df_full (DataFrame): DataFrame contenente i dati.

    Returns:
        bool: True se tutte le colonne sono valide, False altrimenti.
        list: Colonne mancanti.
    """
    all_columns = [col for dim in user_DIM for col in dim]
    missing_columns = [col for col in all_columns if col not in df_full.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in the DataFrame: {missing_columns}")


def log_progress(i,M):
    print("", end=f"\rProgress: {i+1}/{M}")
    if i+1 == M:
        print("\r", end="")



def remove_outliers(df, topic_id="topicID"):
    return df[df[topic_id] != -1].reset_index(drop=True)


"""
def calculate_distribution_count(dataset, DIM):
    distributions = {}  # Store distributions for each dimension

    for dim_index, classes in enumerate(DIM):
        # Determine the actual column names for the classes in this dimension
        class_columns = [f"{cls}" for cls in classes]

        # Find the index of the class with the highest probability in each row for this dimension
        highest_class_counts = dataset[class_columns].idxmax(axis=1).value_counts()

        # Calculate counts and distribution for each class
        counts = [highest_class_counts.get(col, 0) for col in class_columns]
        total = sum(counts)
        distribution = {classes[i]: counts[i] / total for i in range(len(classes))}

        # Store the distribution for the current dimension
        distributions[f"dim{dim_index + 1}"] = distribution

    return distributions

def calculate_distribution_percentages(df, DIM):
    distribution_percentages = {}

    for dim_index, classes in enumerate(DIM):
        # Determine the column names for the classes in this dimension
        class_columns = [f"{cls}" for cls in classes]

        # Calculate the total sum for all classes in this dimension
        total_sum = df[class_columns].sum().sum()

        # Calculate and store the distribution for each class
        distribution_percentages[f"dim{dim_index + 1}"] = {
            cls: df[cls].sum() / total_sum for cls in class_columns
        }
    return distribution_percentages
"""