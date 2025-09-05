from GIFA.generate_intent_full import IntentGenerator
from GPFA.evaluate_intent import encode_intent_labels, merge_intent_dfs
from GPFA.evaluate_intent import IntentClusteringEvaluator
from GPFA.update_prompt import IntentPromptUpdater
import numpy as np
import pandas as pd
import random
from pathlib import Path


class greedy_prompt_optimizer:
    def __init__(self, dynamic_prompt, train_df, val_df, label_intent_df, population_size=5, num_samples=10, method="general", 
                 top_k=2, top_n=2, init_temperature = 0.7, lagged_feedback = True, 
                 type="greedy", show_tqdm=False,
                 # NEW:
                 dummy_mode: bool = False,
                 dummy_folder: str = "gpo_dummy"):
        
        self.dynamic_prompt = dynamic_prompt

        self.train_df = train_df
        self.val_df = val_df
        self.label_intent_df = label_intent_df

        self.population_size = population_size
        self.num_samples = num_samples
        self.method = method
        self.top_k = top_k
        self.top_n = top_n

        self.init_temperature = init_temperature
        self.lagged_feedback = lagged_feedback

        self.show_tqdm = show_tqdm
        self.type = type

        self.prompt_counter = 0

        self.history = pd.DataFrame(columns=[
            "prompt_id", "step", "prompt", "parent_idx", "parent_idxs", "feedback",
            "score_ami", "score_nca", "score_f1", "ami_x_nca",
            "true_clusters", "predicted_clusters"])
        
        self.intent_df_list = []
        self.global_results = []

        # NEW: simple dummy cache + artifact folders
        self.dummy_mode = dummy_mode
        self.dummy_root = Path(dummy_folder)
        self.dummy_gifa = self.dummy_root / "GIFA"                 # CSVs per prompt id
        self.prompts_dir = self.dummy_root / "prompts"             # dynamic prompt per iter
        self.updated_dir = self.dummy_root / "updated_prompts"     # updated_dynamic/static per iter
        self.hist_file = self.dummy_root / "history.csv"
        # Create roots early (ok even in online mode; it's harmless)
        self.dummy_gifa.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.updated_dir.mkdir(parents=True, exist_ok=True)

    def _gifa_path(self, prompt_id: int) -> Path:
        return self.dummy_gifa / f"prompt_{prompt_id:03d}.csv"

    def _prompt_paths(self, prompt_id: int):
        base = f"prompt_{prompt_id:03d}"
        return (
            self.prompts_dir / f"{base}.txt",
            self.updated_dir / f"{base}_updated_dynamic.txt",
            self.updated_dir / f"{base}_updated_static.txt",
        )

    def _save_history_csv(self):
        try:
            self.history.to_csv(self.hist_file, index=False)
        except Exception:
            pass

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
        # NEW: persist log each step
        self._save_history_csv()

    def _evaluate_prompt(self, df, prompt, num_samples = 50, train = False):
        """
        Minimal dummy-mode behavior:
        - Always build updated_static_prompt via IntentPromptUpdater(prompt).create_base_static_prompt()
        - If dummy_mode and cached CSV exists for current prompt_id: load it and skip generation
        - Else: run generation as-is; if dummy_mode, save CSV to cache
        """
        # Always construct base static (pure, no GPT call)
        updater = IntentPromptUpdater(prompt)
        updated_static_prompt = updater.create_base_static_prompt()

        # NEW: figure out this iteration's prompt_id before logging increments it
        prompt_id = self.prompt_counter
        cache_path = self._gifa_path(prompt_id)

        if self.dummy_mode and cache_path.exists():
            # Load cached intents; keep it simple => pandas
            full_intent_df = pd.read_csv(cache_path)
            unique_intents = sorted(full_intent_df["intent"].dropna().unique()) if "intent" in full_intent_df.columns else []
        else:
            # Run the original pipeline unchanged
            try:
                generator = IntentGenerator(updated_static_prompt, api_time=3, n_samples=num_samples, method = self.method,
                                         top_k=self.top_k, top_n=self.top_n, search=True, show_tqdm=self.show_tqdm)
                full_intent_df, unique_intents = generator.run_intent_generation(df)
            except:
                generator = IntentGenerator(updated_static_prompt, api_time=3, num_samples=num_samples, method = self.method,
                                            top_k=self.top_k, top_n=self.top_n, search=True, show_tqdm=self.show_tqdm)
                full_intent_df, unique_intents = generator.run_intent_generation(df)

            # NEW: if dummy mode, save result for future runs
            if self.dummy_mode:
                self.dummy_gifa.mkdir(parents=True, exist_ok=True)
                try:
                    # Polars or pandas: try both
                    if hasattr(full_intent_df, "write_csv"):
                        full_intent_df.write_csv(str(cache_path))
                    else:
                        full_intent_df.to_csv(cache_path, index=False)
                except Exception:
                    try:
                        full_intent_df.to_pandas().to_csv(cache_path, index=False)
                    except Exception:
                        pass

        # Keep original bookkeeping
        if not train:
            self.intent_df_list.append(full_intent_df)

        # For safety, ensure df_merged uses pandas
        if hasattr(full_intent_df, "to_pandas"):
            full_intent_df_pd = full_intent_df.to_pandas()
        else:
            full_intent_df_pd = full_intent_df

        df_merged = merge_intent_dfs(self.label_intent_df, full_intent_df_pd)
        true_clusters, predicted_clusters = encode_intent_labels(df_merged)

        evaluator = IntentClusteringEvaluator(true_clusters, predicted_clusters, list(df_merged["message"]), 
                                            list(df_merged["intent_true"]), list(df_merged["intent_predicted"]))
        results = evaluator.evaluate()

        return results, predicted_clusters, true_clusters
            
    def run_gpo(self):

        # Setup starting prompt
        updater = IntentPromptUpdater(self.dynamic_prompt)
        updated_static_prompt = updater.create_base_static_prompt()
        updated_dynamic_prompt = self.dynamic_prompt

        best_prompt_index = 0 # Always starts with improving itself 

        for iter in range(int(self.population_size)):

            # Compute paths for THIS iteration (before any increments)
            prompt_id = self.prompt_counter
            gifa_csv   = self._gifa_path(prompt_id)
            p_txt, upd_dyn_txt, upd_stat_txt = self._prompt_paths(prompt_id)

            # If we are in dummy mode AND all artifacts already exist for this prompt id,
            # skip LLM prompt updates and reuse the saved prompts.
            use_cache = self.dummy_mode and gifa_csv.exists() and upd_dyn_txt.exists() and upd_stat_txt.exists() and p_txt.exists()

            # Instantiate the updater and generate the updated prompt
            if iter > 0 and not use_cache:

                # Choose best prompt based on a Metric.
                ami_nca_product = [result["Clustering Metrics"]["AMI"] * result["Clustering Metrics"]["NCA"] for result in self.global_results]

                if self.type == "greedy":
                    best_prompt_index = np.argmax(ami_nca_product)

                    print(f"best index found: {best_prompt_index}, with product {ami_nca_product[best_prompt_index]}")

                    updated_dynamic_prompt = self.history["prompt"][best_prompt_index]
                    updater = IntentPromptUpdater(updated_dynamic_prompt, self.global_results[best_prompt_index])

                elif self.type == "random":
                    best_prompt_index = random.sample(range(0,len(self.history)), 1)[0]

                    print(f"random index found: {best_prompt_index}, with product {ami_nca_product[best_prompt_index]}")

                    updated_dynamic_prompt = self.history["prompt"][best_prompt_index]
                    updater = IntentPromptUpdater(updated_dynamic_prompt, self.global_results[best_prompt_index])

                # Check if the feedback for the new prompt should be lagged or not
                if self.lagged_feedback:
                    updated_static_prompt, updated_dynamic_prompt = updater.create_feedback_updated_prompt(fast=True, temperature=self.init_temperature)
                else:
                    updated_static_prompt, updated_dynamic_prompt = updater.create_updated_prompt(fast=True)

                # NEW: save prompts for this iteration
                try:
                    p_txt.write_text(updated_dynamic_prompt)
                    upd_dyn_txt.write_text(updated_dynamic_prompt)
                    upd_stat_txt.write_text(updated_static_prompt)
                except Exception:
                    pass

            elif use_cache:
                # NEW: load saved prompts; keeps traversal identical without GPT calls
                try:
                    updated_dynamic_prompt = upd_dyn_txt.read_text()
                    updated_static_prompt  = upd_stat_txt.read_text()
                except Exception:
                    # Fallback: at least keep base-static from current dynamic
                    updater = IntentPromptUpdater(updated_dynamic_prompt)
                    updated_static_prompt = updater.create_base_static_prompt()

            else:
                # First iteration or online mode without cache: ensure we save the starting prompts too
                try:
                    p_txt.write_text(updated_dynamic_prompt)
                    upd_dyn_txt.write_text(updated_dynamic_prompt)
                    upd_stat_txt.write_text(updated_static_prompt)
                except Exception:
                    pass

            print("########################################")
            print(f"###### Currently at Iteration {iter} ######")
            print("########################################")

            print(f"Updated Prompt: {updated_dynamic_prompt}")

            results, predicted_clusters, true_clusters = self._evaluate_prompt(self.val_df, updated_static_prompt, num_samples = self.num_samples, train = False)

            self.global_results.append(results)

            self._log_prompt(iter, updated_dynamic_prompt, results, true_clusters, predicted_clusters, feedback = self.lagged_feedback, parent_idx = best_prompt_index)

        return self.history
