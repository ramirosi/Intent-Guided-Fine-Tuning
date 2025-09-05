# ===================== Prompt Templates (unchanged) =====================

lamarckian_prompt_template = """I gave a friend an instruction and some inputs. \
The friend read the instruction and wrote an output for every one of the inputs. \
Give back ONLY instruction that was given such that it can be used as a prompt.

Here are the input-output pairs:
## Example ##
{pairs}
"""

feedback_examiner_prompt_template = """You are a quick improver. Given an existing prompt and a series of cases where it made mistakes. \
Look through each case carefully and identify what is causing the mistakes. Based on these observations, output ways to improve the prompt based on the mistakes.
## Existing Prompt ##
{existing_prompt}
## Cases where it gets wrong:##
{wrong_cases}
ways to improve the existing prompt based on observations of the mistakes in the cases above are:"""

feedback_improver_prompt_template = """You are a quick improver. Given an existing prompt and feedback on how it should improve. Create an improved version based on the feedback.
## Existing Prompt ##
{existing_prompt}
## Feedback##
{feedback}
## Improved Prompt##"""

eda_prompt_template = """You are a mutator. Given a series of prompts, your task is to generate another prompt with the same semantic meaning and intentions.
## Existing Prompts ##
{existing_prompts}
The newly mutated prompt is:"""

eda_index_prompt_template = """You are a mutator. Given a series of prompts, your task is to generate another prompt with the same semantic meaning and intentions. \
The series of prompts are ranked by their quality from best to worst.
## Existing Prompts ##
{existing_prompts}
The newly mutated prompt is:"""

crossover_prompt_template = """You are a mutator who is familiar with the concept of cross-over in genetic algorithm, namely combining the genetic information of two parents to generate new offspring. \
Given two parent prompts, you will perform a cross-over to generate an offspring prompt that covers the same semantic meaning as both parents.

RETURN ONLY THE NEW PROMPT.

# Example
Parent prompt 1: Now you are a categorizer, your mission is to ascertain the sentiment of the provided text, either favorable or unfavorable

Parent prompt 2: Assign a sentiment label to the given sentence from ['negative','positive'] and return only the label without any other text.

Offspring prompt: Your mission is to ascertain the sentiment of the provided text and assign a sentiment label from ['negative','positive'].
## Given ##
Parent prompt 1: {prompt1}
Parent prompt 2: {prompt2}
Offspring prompt:"""

semantic_prompt_template = """You are a mutator. Given a prompt, your task is to generate another prompt with the same semantic meaning and intentions.
# Example:
current prompt: Your mission is to ascertain the sentiment of the provided text and assign a label from ['negative','positive'].
mutated prompt: Determine the sentiment of the given sentence and assign a label from ['negative','positive'].
Given:
current prompt: {existing_prompt}
mutated prompt:"""


# ===================== Imports =====================

from GIFA.generate_intent_full import IntentGenerator
from GPFA.evaluate_intent import encode_intent_labels, merge_intent_dfs
from GPFA.evaluate_intent import IntentClusteringEvaluator
from sklearn.metrics.pairwise import cosine_distances
from GPFA.update_prompt import IntentPromptUpdater
from scipy.spatial.distance import hamming
from GIFA.gpt_api import call_ADA_003
from GIFA.gpt_api import call_GPT
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from pathlib import Path
import hashlib, json


# ===================== Small helpers =====================

def find_random_pairs(num_candidates, top_n=5):
    all_pairs = [(i, j) for i in range(num_candidates) for j in range(i + 1, num_candidates)]
    random.shuffle(all_pairs)
    return [(i, j, 0.0) for i, j in all_pairs[:top_n]]

def fast_pairwise_matrix(labels):
    labels = np.array(labels)
    return (labels[:, None] == labels[None, :]).astype(int)

def correctness_vector(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    valid_mask = ~np.isnan(true_labels) & ~np.isnan(pred_labels)
    true_labels_clean = true_labels[valid_mask]
    pred_labels_clean = pred_labels[valid_mask]
    if len(true_labels_clean) < 2:
        return np.array([])
    true_mat = fast_pairwise_matrix(true_labels_clean)
    pred_mat = fast_pairwise_matrix(pred_labels_clean)
    correctness = (true_mat == pred_mat).astype(int)
    triu_indices = np.triu_indices_from(correctness, k=1)
    return correctness[triu_indices]

def compute_all_correctness_vectors(true_labels_list, pred_labels_list):
    return [correctness_vector(true, pred) for true, pred in zip(true_labels_list, pred_labels_list)]

def find_most_different_pairs(vectors, results, alpha=0.5, top_n=5):
    n = len(vectors)
    pairs = []
    hamming_list, quality_list = [], []
    for i in range(n):
        for j in range(i + 1, n):
            min_len = min(len(vectors[i]), len(vectors[j]))
            vi = vectors[i][:min_len]
            vj = vectors[j][:min_len]
            hd = hamming(vi, vj)
            si = results[i]["Clustering Metrics"]["AMI"] * results[i]["Clustering Metrics"]["NCA"]
            sj = results[j]["Clustering Metrics"]["AMI"] * results[j]["Clustering Metrics"]["NCA"]
            q = (si + sj) / 2
            pairs.append((i, j, hd, q))
            hamming_list.append(hd)
            quality_list.append(q)
    min_hd, max_hd = (min(hamming_list), max(hamming_list)) if hamming_list else (0, 1)
    min_q, max_q = (min(quality_list), max(quality_list)) if quality_list else (0, 1)
    scored = []
    for i, j, hd, q in pairs:
        hd_n = (hd - min_hd) / (max_hd - min_hd) if max_hd > min_hd else 0
        q_n  = (q  - min_q ) / (max_q  - min_q ) if max_q  > min_q  else 0
        scored.append((i, j, alpha * hd_n + (1 - alpha) * q_n))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


# ===================== SEE with "read-if-present, else-generate" dummy mode =====================

class SEE:
    """
    dummy_mode=True:
      - Prefer cached artifacts in dummy_folder.
      - If an expected file is missing, run the real path (GPT/ADA/IntentGenerator) ONCE,
        save the artifact to dummy_folder, then continue. Next run will be fully cached.
      - We do not persist 'train' (feedback) GIFA to disk to avoid PID coupling; only VAL GIFA
        are cached as GIFA/prompt_{pid:03d}.csv.
    """
    def __init__(self, train_df, val_df, label_intent_df, population_size=5, method="general", search_method="combination",
                 top_n=2, top_k=2, init_temperature=0.7, val_samples=5, feedback_samples=5, K1=1, K2=2,
                 crossover_types=["error", "error"], lagged_feedback=True, show_tqdm=False,
                 warm_start=False, warm_start_prompt_list=None,
                 dummy_mode=False, dummy_folder="dumm"):
        if warm_start_prompt_list is None:
            warm_start_prompt_list = []
        self.train_df = train_df
        self.val_df = val_df
        self.label_intent_df = label_intent_df

        self.population_size = int(population_size)
        self.method = method
        self.search_method = search_method
        self.top_n = top_n
        self.top_k = top_k
        self.init_temperature = init_temperature
        self.val_samples = val_samples
        self.feedback_samples = feedback_samples
        self.K1 = K1
        self.K2 = K2
        self.crossover_types = [t.lower() for t in crossover_types]
        self.lagged_feedback = lagged_feedback
        self.show_tqdm = show_tqdm
        self.warm_start = warm_start
        self.warm_start_prompt_list = warm_start_prompt_list

        self.intent_df_list = []
        self.history = pd.DataFrame(columns=[
            "prompt_id", "step", "prompt", "parent_idx", "parent_idxs", "feedback",
            "score_ami", "score_nca", "score_f1", "ami_x_nca",
            "true_clusters", "predicted_clusters"])
        self.prompt_counter = 0

        # Caching dirs
        self.dummy_mode = dummy_mode
        self.root = Path(dummy_folder)
        self.dir_gifa    = self.root / "GIFA"
        self.dir_prompts = self.root / "prompts"
        self.dir_updates = self.root / "updated_prompts"
        self.dir_embed   = self.root / "embeddings"
        self.file_history = self.root / "history.csv"
        self.file_embed_index = self.dir_embed / "index.csv"

        # Ensure dir structure
        self.dir_gifa.mkdir(parents=True, exist_ok=True)
        self.dir_prompts.mkdir(parents=True, exist_ok=True)
        self.dir_updates.mkdir(parents=True, exist_ok=True)
        self.dir_embed.mkdir(parents=True, exist_ok=True)
        if not self.file_embed_index.exists():
            pd.DataFrame(columns=["prompt_id","hash","prompt"]).to_csv(self.file_embed_index, index=False)

    # ---------- path helpers ----------
    def _gifa_csv_val(self, pid: int) -> Path:
        return self.dir_gifa / f"prompt_{pid:03d}.csv"
    def _prompt_txt(self, pid: int) -> Path:
        return self.dir_prompts / f"prompt_{pid:03d}.txt"
    def _upd_dyn_txt(self, pid: int) -> Path:
        return self.dir_updates / f"prompt_{pid:03d}_updated_dynamic.txt"
    def _upd_stat_txt(self, pid: int) -> Path:
        return self.dir_updates / f"prompt_{pid:03d}_updated_static.txt"

    def _save_history(self):
        try:
            self.history.to_csv(self.file_history, index=False)
        except Exception:
            pass

    # ---------- embeddings (read if present, else call ADA and save) ----------
    def _hash_prompt(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    def _embed_path(self, prompt_text: str) -> Path:
        return self.dir_embed / f"{self._hash_prompt(prompt_text)}.json"
    def _save_embed_index_row(self, prompt_id: int, prompt_text: str):
        try:
            df = pd.read_csv(self.file_embed_index)
            h = self._hash_prompt(prompt_text)
            if not ((df["hash"] == h) & (df["prompt_id"] == prompt_id)).any():
                df.loc[len(df)] = {"prompt_id": prompt_id, "hash": h, "prompt": prompt_text}
                df.to_csv(self.file_embed_index, index=False)
        except Exception:
            pass
    def _get_or_create_embedding(self, prompt_text: str):
        p = self._embed_path(prompt_text)
        if self.dummy_mode and p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        # fallback: call ADA and save
        emb = call_ADA_003(prompt_text)
        try:
            p.write_text(json.dumps(emb))
        except Exception:
            pass
        return emb

    # ---------- logging ----------
    def _log_prompt(self, step, prompt, result, true_clusters, predicted_clusters,
                    parent_idx=None, parent_idxs=None, feedback=None):
        row = {
            "prompt_id": self.prompt_counter,
            "step": step,
            "prompt": prompt,
            "parent_idx": parent_idx,
            "parent_idxs": parent_idxs,
            "feedback": feedback,
            "score_ami": result["Clustering Metrics"]["AMI"],
            "score_nca": result["Clustering Metrics"]["NCA"],
            "score_f1": result["Clustering Metrics"]["pairwise_F1"],
            "ami_x_nca": result["Clustering Metrics"]["AMI"] * result["Clustering Metrics"]["NCA"],
            "true_clusters": true_clusters,
            "predicted_clusters": predicted_clusters,
        }
        self.history.loc[len(self.history)] = row
        self.prompt_counter += 1
        self._save_history()

    # ---------- evaluation (read val CSV if present, else generate & save) ----------
    def _evaluate_prompt(self, df, prompt, num_samples=50, train=False):
        pid = self.prompt_counter

        # Always compute a static prompt string (local only)
        updater = IntentPromptUpdater(prompt)
        updated_static_prompt = updater.create_base_static_prompt()

        # Save prompt texts if not already present
        if not train:
            pt = self._prompt_txt(pid)
            if not pt.exists():
                try:
                    pt.write_text(str(prompt))
                except Exception:
                    pass
            ust = self._upd_stat_txt(pid)
            if not ust.exists():
                try:
                    ust.write_text(str(updated_static_prompt))
                except Exception:
                    pass
            # dynamic file mirrors evaluated prompt
            udt = self._upd_dyn_txt(pid)
            if not udt.exists():
                try:
                    udt.write_text(str(prompt))
                except Exception:
                    pass

        # Read cached VAL intents if present (dummy)
        csv_path = self._gifa_csv_val(pid)
        if self.dummy_mode and (not train) and csv_path.exists():
            intents_df = pd.read_csv(csv_path)
        else:
            # fallback: run generator and (if not train) save
            generator = IntentGenerator(
                updated_static_prompt, api_time=3, n_samples=num_samples,
                method=self.method, top_k=self.top_k, top_n=self.top_n,
                search=True, show_tqdm=self.show_tqdm
            )
            intents_df, _ = generator.run_intent_generation(df)
            if not train:
                try:
                    intents_df.to_csv(csv_path, index=False)
                except Exception:
                    try:
                        intents_df.to_pandas().to_csv(csv_path, index=False)  # in case it's polars
                    except Exception:
                        pass

        if not train:
            self.intent_df_list.append(intents_df)

        df_pred = intents_df.to_pandas() if hasattr(intents_df, "to_pandas") else intents_df
        df_merged = merge_intent_dfs(self.label_intent_df, df_pred)
        true_clusters, predicted_clusters = encode_intent_labels(df_merged)
        evaluator = IntentClusteringEvaluator(
            true_clusters, predicted_clusters,
            list(df_merged["message"]),
            list(df_merged["intent_true"]),
            list(df_merged["intent_predicted"])
        )
        results = evaluator.evaluate()
        return results, predicted_clusters, true_clusters

    # ---------- Phase 0 ----------
    def _initialize_population(self):
        candidates, results = [], []
        predicted_clusters_total, true_clusters_total = [], []

        if self.warm_start:
            base_candidates = self.warm_start_prompt_list
        else:
            base_candidates = []
            if self.dummy_mode:
                # try to read prompts; if missing, generate and save
                for k in range(self.population_size):
                    pid = self.prompt_counter + k
                    p = self._prompt_txt(pid)
                    if p.exists():
                        base_candidates.append(p.read_text().strip())
                    else:
                        # fallback: Lamarckian via GPT and save
                        sample_df = self.label_intent_df.sample(n=min(10, len(self.label_intent_df)))
                        inp = sample_df['conversation'].tolist()
                        out = sample_df['intent'].tolist()
                        pair_lines = "\n".join(f"{i} -> {o}" for i, o in zip(inp, out))
                        query = lamarckian_prompt_template.format(pairs=pair_lines)
                        inferred_prompt = call_GPT(query, temperature=self.init_temperature).strip()
                        if not inferred_prompt:
                            inferred_prompt = "Classify the message into intents."
                        p.write_text(inferred_prompt)
                        base_candidates.append(inferred_prompt)
            else:
                for _ in range(self.population_size):
                    sample_df = self.label_intent_df.sample(n=min(10, len(self.label_intent_df)))
                    inp = sample_df['conversation'].tolist()
                    out = sample_df['intent'].tolist()
                    pair_lines = "\n".join(f"{i} -> {o}" for i, o in zip(inp, out))
                    query = lamarckian_prompt_template.format(pairs=pair_lines)
                    inferred_prompt = call_GPT(query, temperature=self.init_temperature).strip()
                    if inferred_prompt and inferred_prompt not in base_candidates:
                        base_candidates.append(inferred_prompt)

        # Evaluate each on VAL and log
        for prompt in base_candidates:
            _ = self._get_or_create_embedding(prompt)
            self._save_embed_index_row(self.prompt_counter, prompt)

            res, predicted_clusters, true_clusters = self._evaluate_prompt(
                self.val_df, prompt, num_samples=self.val_samples, train=False
            )
            self._log_prompt(step="lamarckian", prompt=prompt, result=res,
                             true_clusters=true_clusters, predicted_clusters=predicted_clusters)
            candidates.append(prompt)
            results.append(res)
            predicted_clusters_total.append(predicted_clusters)
            true_clusters_total.append(true_clusters)

        return candidates, results, predicted_clusters_total, true_clusters_total

    # ---------- Phase 1 (Feedback) ----------
    def _local_feedback_mutation(self, candidates, results, predicted_clusters, true_clusters):
        previous_candidates = candidates
        previous_results = results
        all_candidates = list(candidates)
        all_results = list(results)
        all_predicted_clusters = list(predicted_clusters)
        all_true_clusters = list(true_clusters)

        for k in range(self.K1):
            print(f"Feedback loop: {k}")
            current_candidates, current_results = [], []

            for i in range(len(previous_candidates)):
                parent_prompt = previous_candidates[i]

                if self.dummy_mode:
                    # Prefer prepared updated prompt; if missing, compute and save it.
                    expected_pid = self.prompt_counter  # child pid to be logged next
                    upd_dyn_path = self._upd_dyn_txt(expected_pid)
                    if upd_dyn_path.exists():
                        updated_dynamic_prompt = upd_dyn_path.read_text().strip()
                    else:
                        # fallback: run train eval -> updater -> save
                        train_result, _, _ = self._evaluate_prompt(
                            df=self.train_df, prompt=parent_prompt,
                            num_samples=min(self.feedback_samples, self.val_samples), train=True
                        )
                        updater = IntentPromptUpdater(parent_prompt, train_result)
                        if self.lagged_feedback:
                            _, updated_dynamic_prompt = updater.create_feedback_updated_prompt(
                                fast=True, temperature=self.init_temperature
                            )
                        else:
                            _, updated_dynamic_prompt = updater.create_updated_prompt(fast=True)
                        upd_dyn_path.write_text(updated_dynamic_prompt)
                else:
                    # real path
                    train_result, _, _ = self._evaluate_prompt(
                        df=self.train_df, prompt=parent_prompt,
                        num_samples=min(self.feedback_samples, self.val_samples), train=True
                    )
                    updater = IntentPromptUpdater(parent_prompt, train_result)
                    if self.lagged_feedback:
                        _, updated_dynamic_prompt = updater.create_feedback_updated_prompt(
                            fast=True, temperature=self.init_temperature
                        )
                    else:
                        _, updated_dynamic_prompt = updater.create_updated_prompt(fast=True)

                _ = self._get_or_create_embedding(updated_dynamic_prompt)
                self._save_embed_index_row(self.prompt_counter, updated_dynamic_prompt)

                val_result, pred_c, true_c = self._evaluate_prompt(
                    df=self.val_df, prompt=updated_dynamic_prompt, num_samples=self.val_samples, train=False
                )

                self._log_prompt(step=f"feedback_{k}", prompt=updated_dynamic_prompt, result=val_result,
                                 true_clusters=true_c, predicted_clusters=pred_c, parent_idx=i)

                current_candidates.append(updated_dynamic_prompt)
                current_results.append(val_result)

                all_candidates.append(updated_dynamic_prompt)
                all_results.append(val_result)
                all_predicted_clusters.append(pred_c)
                all_true_clusters.append(true_c)

            previous_candidates = current_candidates
            previous_results = current_results

        return all_candidates, all_results, all_predicted_clusters, all_true_clusters

    # ---------- Phase 2 (Crossover) ----------
    def _crossover_mutation(self, candidates, results, predicted_clusters, true_clusters, type="error", k2_round=0):
        type = type.lower()
        if self.dummy_mode:
            pairs = find_random_pairs(len(candidates), top_n=self.population_size)
        else:
            if type == "error":
                vectors = compute_all_correctness_vectors(true_clusters, predicted_clusters)
                pairs = find_most_different_pairs(vectors, results, top_n=self.population_size)
            elif type == "ada":
                embs = [self._get_or_create_embedding(p) for p in candidates]
                valid = [i for i, e in enumerate(embs) if isinstance(e, (list, tuple)) and len(e) > 0]
                if len(valid) < 2:
                    pairs = find_random_pairs(len(candidates), top_n=self.population_size)
                else:
                    embs_mat = np.array([embs[i] for i in valid], dtype=np.float32)
                    dist_sub = cosine_distances(embs_mat)
                    n = len(candidates)
                    dist_matrix = np.zeros((n, n), dtype=np.float32)
                    for a, ia in enumerate(valid):
                        for b, ib in enumerate(valid):
                            dist_matrix[ia, ib] = dist_sub[a, b]
                    # reuse quality-aware selection
                    pairs = find_most_different_pairs(dist_matrix, results, top_n=self.population_size)
            else:
                pairs = find_random_pairs(len(candidates), top_n=self.population_size)

        local_candidates, local_results = [], []
        local_predicted_clusters, local_true_clusters = [], []
        base_pid = self.prompt_counter

        for off_idx, (i, j, score) in enumerate(pairs):
            if self.dummy_mode:
                expected_pid = base_pid + off_idx
                upd_dyn_path = self._upd_dyn_txt(expected_pid)
                if upd_dyn_path.exists():
                    inferred_prompt = upd_dyn_path.read_text().strip()
                else:
                    # fallback: generate offspring and save
                    prompt_tmpl = crossover_prompt_template.format(prompt1=candidates[i], prompt2=candidates[j])
                    inferred_prompt = call_GPT(prompt_tmpl).strip()
                    upd_dyn_path.write_text(inferred_prompt)
            else:
                updated_prompt = crossover_prompt_template.format(prompt1=candidates[i], prompt2=candidates[j])
                inferred_prompt = call_GPT(updated_prompt).strip()

            _ = self._get_or_create_embedding(inferred_prompt)
            self._save_embed_index_row(self.prompt_counter, inferred_prompt)

            result, pred_c, true_c = self._evaluate_prompt(
                df=self.val_df, prompt=inferred_prompt, num_samples=self.val_samples, train=False
            )

            self._log_prompt(step=f"crossover_{type}_{k2_round}", prompt=inferred_prompt, result=result,
                             true_clusters=true_c, predicted_clusters=pred_c, parent_idxs=(i, j))

            local_candidates.append(inferred_prompt)
            local_results.append(result)
            local_predicted_clusters.append(pred_c)
            local_true_clusters.append(true_c)

        total_candidates = candidates + local_candidates
        total_results = results + local_results
        total_predicted_clusters = list(predicted_clusters) + local_predicted_clusters
        total_true_clusters      = list(true_clusters)      + local_true_clusters
        return total_candidates, total_results, total_predicted_clusters, total_true_clusters

    # ---------- Orchestration ----------
    def _warm_start_population(self):
        candidates, results = [], []
        predicted_clusters_total, true_clusters_total = [], []
        for prompt in self.warm_start_prompt_list:
            _ = self._get_or_create_embedding(prompt)
            self._save_embed_index_row(self.prompt_counter, prompt)
            res, pred_c, true_c = self._evaluate_prompt(self.val_df, prompt, num_samples=self.val_samples, train=False)
            self._log_prompt(step="warm_start", prompt=prompt, result=res,
                             true_clusters=true_c, predicted_clusters=pred_c)
            candidates.append(prompt)
            results.append(res)
            predicted_clusters_total.append(pred_c)
            true_clusters_total.append(true_c)
        return candidates, results, predicted_clusters_total, true_clusters_total

    def run(self):
        if not self.warm_start:
            self.candidates, self.results, self.predicted_clusters, self.true_clusters = self._initialize_population()
        else:
            self.candidates, self.results, self.predicted_clusters, self.true_clusters = self._warm_start_population()

        if self.K1 != 0:
            self.candidates, self.results, self.predicted_clusters, self.true_clusters = self._local_feedback_mutation(
                self.candidates, self.results, self.predicted_clusters, self.true_clusters
            )

        k2_round = 0
        if self.K2 != 0:
            for crossover_type in self.crossover_types:
                self.candidates, self.results, self.predicted_clusters, self.true_clusters = self._crossover_mutation(
                    self.candidates, self.results, self.predicted_clusters, self.true_clusters,
                    type=crossover_type, k2_round=k2_round
                )
                k2_round += 1

        return self.candidates, self.results, self.predicted_clusters, self.true_clusters
