import numpy as np
from itertools import combinations
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, homogeneity_score, completeness_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
from itertools import zip_longest
from sklearn.metrics import adjusted_mutual_info_score
from collections import defaultdict
import math

import warnings

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples",
    category=UserWarning,
    module="sklearn.metrics.cluster._supervised"
)

def merge_intent_dfs(test_df, target_df):

    test_df = pd.DataFrame(test_df)
    target_df = pd.DataFrame(target_df)

    target_df.columns = ["index", "intent", "count", "message"]

    df_merged = test_df.merge(target_df, left_on=["conversation", "index"], right_on=["message", "index"], how="inner", suffixes=("_true", "_predicted"))
    df_merged = df_merged.dropna(subset=["index", "message"])

    # Drop NA;s
    df_merged = df_merged.dropna(subset=["intent_predicted"])

    return df_merged

def safe_label_encode(series):
    mask = series.notna()
    encoded = pd.Series([np.nan] * len(series), index=series.index)

    le = LabelEncoder()
    encoded[mask] = le.fit_transform(series[mask])

    return encoded

def encode_intent_labels(df_merged):
    # Encode ground truth intents (each cluster gets a unique ID)
    df_merged["true_cluster_id"] = safe_label_encode(df_merged["intent_true"])

    # Encode predicted intents separately (different numbers, same grouping)
    df_merged["predicted_cluster_id"] = safe_label_encode(df_merged["intent_predicted"])

    true_clusters = df_merged["true_cluster_id"].values
    predicted_clusters = df_merged["predicted_cluster_id"].values

    return true_clusters, predicted_clusters

class IntentClusteringEvaluator:
    def __init__(self, ground_truth, predicted, messages, true_intents, predicted_intents, stratified=False):
        self.ground_truth = ground_truth
        self.predicted = predicted
        self.messages = messages
        self.true_intents = true_intents
        self.predicted_intents = predicted_intents
        self.stratified = stratified

    def extract_errors(self):
        """Extracts examples of false splits and false merges."""
        false_splits = []
        false_merges = []
        n = len(self.ground_truth)
        
        for i, j in combinations(range(n), 2):
            same_true = (self.ground_truth[i] == self.ground_truth[j])
            same_pred = (self.predicted[i] == self.predicted[j])
            
            if same_true and not same_pred:
                false_splits.append((i, j))
            elif not same_true and same_pred:
                false_merges.append((i, j))
        
        return false_splits, false_merges

    def find_interesting_error_examples(self, top_n=20, stratified=False):
        """Samples up to `top_n` examples, optionally stratified by intent/cluster."""
        false_splits, false_merges = self.extract_errors()

        if not stratified:
            # Original behavior: random sampling

            num_each = top_n // 2

            sampled_splits = random.sample(false_splits, min(num_each, len(false_splits)))
            sampled_merges = random.sample(false_merges, min(num_each, len(false_merges)))

        else:
            # Stratified behavior
            def stratified_sample(pairs, label_source, max_total):
                buckets = defaultdict(list)
                for i, j in pairs:
                    key = label_source[i]
                    buckets[key].append((i, j))

                samples = []
                bucket_items = list(buckets.items())
                random.shuffle(bucket_items)  # <-- shuffle bucket order

                n_buckets = len(bucket_items)
                per_bucket = max(1, max_total // n_buckets)

                for _, bucket in bucket_items:
                    sampled = random.sample(bucket, min(len(bucket), per_bucket))
                    samples.extend(sampled)
                    if len(samples) >= max_total:
                        break

                return samples[:max_total]

            sampled_splits = stratified_sample(false_splits, self.true_intents, top_n // 2)
            sampled_merges = stratified_sample(false_merges, self.predicted_intents, top_n // 2)

        # Interleave examples
        interleaved = list(zip_longest(sampled_splits, sampled_merges))
        interesting_examples = []

        for split_pair, merge_pair in interleaved:
            for pair, error_type in [(split_pair, "false_split"), (merge_pair, "false_merge")]:
                if pair is None:
                    continue
                i, j = pair

                if error_type == "false_split":
                    str_pair = "Message: " + self.messages[i] + " AND " + self.messages[j] + " should have the same intent: " + self.true_intents[i]
                elif error_type == "false_merge":
                    str_pair = "Message: " + self.messages[i] + " AND " + self.messages[j] + " should have had seperate intents: " + self.true_intents[i] + " AND " + self.true_intents[j]

                # str_pair = (
                #     self.messages[i] + ", true intent: " + self.true_intents[i] + ", predicted intent: " + self.predicted_intents[i],
                #     self.messages[j] + ", true intent: " + self.true_intents[j] + ", predicted intent: " + self.predicted_intents[j]
                # )

                # interesting_examples.append((*str_pair, error_type))

                interesting_examples.append((str_pair, error_type))
                if len(interesting_examples) >= top_n:
                    break
            if len(interesting_examples) >= top_n:
                break

        return interesting_examples, false_splits, false_merges

    def compute_metrics(self, true_labels, pred_labels):
        true_labels = np.asarray(true_labels)
        pred_labels = np.asarray(pred_labels)
        N = len(true_labels)
        if N == 0:
            return {}
        # Get unique classes and clusters and their counts
        classes, class_counts = np.unique(true_labels, return_counts=True)
        clusters, cluster_counts = np.unique(pred_labels, return_counts=True)
        num_classes = len(classes)
        num_clusters = len(clusters)
        # Contingency table: counts of intersections between each predicted cluster and each true class
        class_index = {cl: idx for idx, cl in enumerate(classes)}
        cluster_index = {clust: idx for idx, clust in enumerate(clusters)}
        contingency = np.zeros((num_clusters, num_classes), dtype=int)
        for t, p in zip(true_labels, pred_labels):
            contingency[cluster_index[p], class_index[t]] += 1
        # Compute purity
        correct_per_cluster = contingency.max(axis=1)  # for each cluster, count of most common true label
        purity = correct_per_cluster.sum() / N
        # Compute NCA (Normalized Clustering Accuracy)
        if num_classes > 0:
            purity_per_cluster = correct_per_cluster / cluster_counts  # purity of each cluster
            avg_purity = purity_per_cluster.mean()
            baseline = 1.0 / num_classes  # expected purity if cluster assignments were uniformly random
            NCA = (avg_purity - baseline) / (1.0 - baseline) if (1.0 - baseline) > 0 else 1.0
        else:
            NCA = 1.0
        # Compute Adjusted Mutual Information (AMI)
        AMI = adjusted_mutual_info_score(true_labels, pred_labels)
        # Compute pairwise precision, recall, F1
        # Count true pairs (pairs of messages with same true label)
        true_pairs = 0
        for count in class_counts:
            if count > 1:
                true_pairs += math.comb(count, 2)
        # Count predicted pairs (pairs in same cluster)
        pred_pairs = 0
        for count in cluster_counts:
            if count > 1:
                pred_pairs += math.comb(count, 2)
        # Count correctly clustered pairs (pairs that are same true label and same cluster)
        correct_pairs = 0
        for i in range(num_clusters):
            # sum of n_ij choose 2 for each true label j in cluster i
            for count in contingency[i]:
                if count > 1:
                    correct_pairs += math.comb(count, 2)
        precision = correct_pairs / pred_pairs if pred_pairs > 0 else 1.0  # if no predicted pairs, precision = 1 by convention (no false positives)
        recall = correct_pairs / true_pairs if true_pairs > 0 else 1.0    # if no true pairs (all singleton classes), recall = 1 (trivial perfect clustering)
        pairwise_F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "purity": purity,
            "NCA": NCA,
            "AMI": AMI,
            "pairwise_precision": precision,
            "pairwise_recall": recall,
            "pairwise_F1": pairwise_F1
        }


    def evaluate(self):
        """Computes all evaluation metrics and extracts key errors."""
        ari = adjusted_rand_score(self.ground_truth, self.predicted)
        nmi = normalized_mutual_info_score(self.ground_truth, self.predicted)
        fmi = fowlkes_mallows_score(self.ground_truth, self.predicted)

        interesting_examples, false_splits, false_merges = self.find_interesting_error_examples(top_n=20, stratified=self.stratified)

        true_amount = len(set(self.ground_truth))
        predicted_amount = len(set(self.predicted))

        metrics = self.compute_metrics(self.ground_truth, self.predicted)
        
        return {
            "Clustering Metrics": {"Purity":metrics["purity"], "NCA":metrics["NCA"], "AMI":metrics["AMI"], "pairwise_precision":metrics["pairwise_precision"],
                                   "pairwise_recall": metrics["pairwise_recall"], "pairwise_F1":metrics["pairwise_F1"], "ARI": ari, "NMI": nmi, "FMI": fmi},
            "Errors": {"Interesting Examples": interesting_examples, "False Splits": len(false_splits), "False Merges": len(false_merges)},
            "Amounts": {"True Amount": true_amount, "Predicted Amount":predicted_amount}
        }
