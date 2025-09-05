from tqdm import tqdm
from GPFA.see_algorithm import SEE
import numpy as np
import pandas as pd
import time
from pathlib import Path

# -------------------------------
# Utilities
# -------------------------------

def see_run_history(ph_object):
    """Collect a flat per-prompt history from a SEE run, and append
    all IntentGenerator DataFrame columns (per-step lists).
    """
    history = ph_object.history.copy()
    if len(ph_object.intent_df_list) == 0:
        return history
    # Convert any polars DFs to pandas
    intent_dfs = [df.to_pandas() if hasattr(df, "to_pandas") else df for df in ph_object.intent_df_list]
    column_names = intent_dfs[0].columns

    for col in column_names:
        col_data = [df[col].tolist() if col in df.columns else [] for df in intent_dfs]
        history[col] = col_data

    return history


# -------------------------------
# Hyperband Wrapper with Dummy Mode
# -------------------------------

class hb_see:
    def __init__(
        self,
        train_df,
        val_df,
        label_intent_df,
        schedule,
        method: str = "general",
        search_method: str = "combination",
        top_n: int = 2,
        top_k: int = 2,
        init_temperature: float = 0.7,
        feedback_samples: int = 5,
        K1: int = 1,
        K2: int = 2,
        crossover_type = ("error", "error"),
        lagged_feedback: bool = True,
        show_tqdm: bool = False,
        filename: str | None = None,
        # dummy controls
        dummy_mode: bool = False,
        dummy_root: str = "hyperband_see"
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.label_intent_df = label_intent_df

        self.schedule = schedule

        self.method = method
        self.search_method = search_method
        self.top_n = top_n
        self.top_k = top_k

        self.init_temperature = init_temperature
        self.feedback_samples = feedback_samples
        self.K1 = K1
        self.K2 = K2
        self.crossover_type = list(crossover_type)
        self.lagged_feedback = lagged_feedback

        self.show_tqdm = show_tqdm
        self.filename = filename

        # dummy mode
        self.dummy_mode = dummy_mode
        self.dummy_root = Path(dummy_root)
        if self.dummy_mode:
            self.dummy_root.mkdir(parents=True, exist_ok=True)

        # initialize/restore aggregate CSV
        try:
            if self.filename is not None and Path(self.filename).exists():
                self.total_df = pd.read_csv(self.filename)
            elif self.dummy_mode and (self.dummy_root / "total_history.csv").exists():
                self.total_df = pd.read_csv(self.dummy_root / "total_history.csv")
            else:
                self.total_df = pd.DataFrame()
        except Exception:
            self.total_df = pd.DataFrame()

    def _run_dir(self, bracket_idx: int, round_idx: int) -> Path:
        return self.dummy_root / f"bracket_{bracket_idx:02d}" / f"round_{round_idx:02d}"

    def _save_total(self):
        if self.filename is not None:
            self.total_df.to_csv(self.filename, index=False)
        if self.dummy_mode:
            (self.dummy_root / "total_history.csv").parent.mkdir(parents=True, exist_ok=True)
            self.total_df.to_csv(self.dummy_root / "total_history.csv", index=False)

    def run_hyperband(self):
        prompts = []  # warm-start pool carried across rounds within a bracket

        for idx, tournament in enumerate(self.schedule["multiplicity"]):
            for r in tqdm(range(tournament), desc=f"Bracket {idx+1}/{len(self.schedule['multiplicity'])}"):
                start = time.time()

                population_size = int(self.schedule["x_list"][r + idx])
                val_samples = int(self.schedule["R_list"][r + idx])

                if r == 0:
                    warm_start = False
                    warm_start_prompt_list = []
                else:
                    warm_start = True
                    warm_start_prompt_list = prompts

                # Build / reuse run directory for this bracket/round
                run_dir = self._run_dir(idx + 1, r + 1)
                if self.dummy_mode:
                    run_dir.mkdir(parents=True, exist_ok=True)

                # If in dummy mode and the run already exists with a cached run_history, skip heavy work
                cached_run_history = run_dir / "run_history.csv"
                if self.dummy_mode and cached_run_history.exists():
                    run_df = pd.read_csv(cached_run_history)
                else:
                    # Each SEE gets its own nested dummy folder to stash its artifacts
                    ph = SEE(
                        self.train_df,
                        self.val_df,
                        self.label_intent_df,
                        population_size=population_size,
                        method=self.method,
                        search_method=self.search_method,
                        top_n=self.top_n,
                        top_k=self.top_k,
                        init_temperature=self.init_temperature,
                        val_samples=val_samples,
                        K1=self.K1,
                        feedback_samples=self.feedback_samples,
                        K2=self.K2,
                        crossover_types=self.crossover_type,
                        lagged_feedback=self.lagged_feedback,
                        show_tqdm=self.show_tqdm,
                        warm_start=warm_start,
                        warm_start_prompt_list=warm_start_prompt_list,
                        # pass through dummy configuration
                        dummy_mode=self.dummy_mode,
                        dummy_folder=str(run_dir),
                    )
                    ph.run()

                    # Collect per-run history + attach intent columns
                    run_df = see_run_history(ph)

                    # Persist run-level history for fast reuse
                    try:
                        run_df.to_csv(cached_run_history, index=False)
                    except Exception:
                        pass

                # Annotate with hyperband metadata
                run_df["population_size"] = population_size
                run_df["val_samples"] = val_samples
                run_df["tournament"] = idx + 1
                run_df["round"] = r + 1

                # Select top prompts based on performance for warm-starting next round
                # (fall back safely if metric is missing)
                metric = run_df.get("ami_x_nca", pd.Series([-np.inf] * len(run_df)))
                top_indices = metric.fillna(-np.inf).to_numpy().argsort()[::-1][: max(1, int(population_size / 2))]
                prompts = run_df["prompt"].iloc[top_indices].tolist()

                # Archive results globally
                self.total_df = pd.concat([self.total_df, run_df], axis=0, ignore_index=True)
                self._save_total()

                elapsed_time = time.time() - start
                print(
                    f"Finished Hyperband (bracket {idx+1}, round {r+1}) in {elapsed_time:.1f}s | "
                    f"pop={population_size}, val_samples={val_samples}"
                )

        return self.total_df
