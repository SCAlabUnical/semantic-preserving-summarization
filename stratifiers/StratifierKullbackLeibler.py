import numpy as np
import pandas as pd

from utils.StratifierCommonFunctions import compute_target_counts, kl_divergence, print_distribution, \
    compute_distribution_by_sum_perc, log_progress


class StratifierKullbackLeibler:

    def __init__(self, k=40, use_relevance_score=False, alpha=0.6, debug=False):
        self.k = k
        self.use_relevance_score = use_relevance_score
        self.scoring_function = lambda x,y,i: min((alpha+0.05*i),0.85)*x + (1-min((alpha+0.05*i),0.85))*(1-y) if use_relevance_score else x
        self.debug = debug

    def stratify(self,df, DIM, M):

        # Display initial dataset distribution
        if self.debug:
            print("Initial Dataset Distribution:")



        distribution_percentages = compute_distribution_by_sum_perc(df, DIM)
        target_distribution = [list(d.values()) for d in distribution_percentages]

        # Define target counts based on calculated distribution
        target_counts = compute_target_counts(distribution_percentages, DIM, M)

        w = []
        for x in DIM:
            w.append(1 / len(x))

        if self.debug:
            print("\n")
            print(target_counts)

        if self.use_relevance_score:
            df_sorted = df.sort_values(by='relevance_score', ascending=False)
        else:
            # Step 1: Calculate combined scores for each row
            df['score'] = 0
            for classes in DIM:
                num_classes = len(classes)
                max_variance = (num_classes - 1) / (num_classes ** 2)

                # Calculate confidence
                class_probs = df[classes].values
                actual_variance = np.var(class_probs, axis=1)
                confidence = np.clip(actual_variance / max_variance, 0, 1)
                df['score'] += confidence

            df['score'] /= len(DIM)

            # Step 2: Sort rows by score in descending order
            df_sorted = df.sort_values(by='score', ascending=False)

        # Step 3: Greedy selection with constraint checking
        selected_sample = []
        current_counts = {cls: 0 for dim in DIM for cls in dim}  # Initialize counts for each class
        # print(current_counts)

        idx_selected = set()

        for i in range(M):
            valid_sample = []

            for idx, row in df_sorted.iterrows():
                if idx not in idx_selected:
                    valid = True  # Assume the row is valid for selection

                    # Track classes selected for this row
                    selected_classes = set()

                    # Check each dimension for the class with the highest probability
                    for dim_index, classes in enumerate(DIM):
                        # Find the class with the highest probability in this dimension
                        max_class = max(classes, key=lambda cls: row.get(f"{cls}", 0))

                        # Check if adding this class would exceed the target count
                        if current_counts[max_class] < target_counts[max_class]:
                            selected_classes.add(max_class)  # Mark as selected for this row
                        else:
                            valid = False  # Set valid to False if it exceeds the target
                            break  # No need to check further if one condition fails

                    # Add row if all conditions are met across dimensions
                    if valid:
                        valid_sample.append((idx, row, selected_classes))

                    # Stop if sample size reached
                    if len(valid_sample) ==  self.k :
                        break


            if valid_sample:
                # find the best tuple that minimize the kullback-leibler divergence on the valid sample
                kl_scores =[]
                relevance_scores=[]
                for _, x, _ in valid_sample:
                    selected_sample_copy = selected_sample.copy()
                    selected_sample_copy.append(x)
                    sample_df = pd.DataFrame(selected_sample_copy)
                    sample_distribution = compute_distribution_by_sum_perc(sample_df, DIM)
                    kl = 0
                    for index_dim in range(len(target_distribution)):
                        kl += w[index_dim] * kl_divergence(target_distribution[index_dim], list(sample_distribution[index_dim].values()))

                    kl_scores.append(kl)
                    relevance_scores.append(x["relevance_score"])


                kl_scores = np.array(kl_scores)
                kl_scores/= np.max(kl_scores)

                relevance_scores = np.array(relevance_scores)

                valid_sample_score = self.scoring_function(kl_scores, relevance_scores,i)

                idx, row, selected_classes = valid_sample[np.argmin(valid_sample_score)]

                selected_sample.append(row)
                idx_selected.add(idx)
                # Update current counts for each selected class
                for cls in selected_classes:
                    current_counts[cls] += 1
                    # current_counts[cls] += row.get(f"{cls}", 0)
                # print(str(row))
            log_progress(i,M)



        # Convert selected sample to DataFrame
        sample_df = pd.DataFrame(selected_sample)

        # Final Adjustment: Add or remove reviews to meet the exact sample size
        if len(sample_df) < M:
            # Add reviews with the next highest confidence scores
            remaining_rows = df_sorted.loc[~df_sorted.index.isin(sample_df.index)]
            for idx, row in remaining_rows.iterrows():
                sample_df = pd.concat([sample_df, pd.DataFrame([row])])
                if len(sample_df) == M:
                    break

        if self.debug:
            print_distribution(compute_distribution_by_sum_perc(sample_df, DIM))
        #print(time.time()-start_time)
        return sample_df