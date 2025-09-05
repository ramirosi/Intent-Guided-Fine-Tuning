import logging
import random
import re
import signal
import time
import os
from typing import Any, List, Tuple, Optional

import polars as pl
from tqdm import tqdm

from GIFA.gpt_api import call_GPT
from GIFA.intent_description import DescriptionProcessor
from GIFA.retrieval.history_retrieval import IntentRetriever

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Constants ---
CHUNK_SIZE = 250
MAX_RETRIES = 6
TIMEOUT_S = 90
MAX_SEARCH_INTENTS = 50


class TimeoutException(Exception):
    pass


def _timeout_supported() -> bool:
    return hasattr(signal, "SIGALRM")


def _run_with_timeout(fn, seconds: int = TIMEOUT_S):
    if _timeout_supported():
        def _handler(signum, frame):
            raise TimeoutException()
        original = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original)
    start = time.monotonic()
    result = fn()
    if time.monotonic() - start > seconds:
        raise TimeoutException()
    return result


class IntentGenerator:
    def __init__(
        self,
        base_prompt: str,
        *,
        n_samples: Optional[int] = None,
        seed: int = 2025,
        api_time: int = 7,
        method: str = "general",
        top_k: int = 3,
        top_n: int = 2,
        search: bool = True,
        show_tqdm: bool = True,
        dummy_mode: bool = False,             # <-- NEW
        dummy_file: str = "dummy_intents.csv" # <-- NEW
    ) -> None:
        self.base_prompt = base_prompt
        self.n_samples = n_samples
        self.api_time = api_time
        self.method = method
        self.top_k = top_k
        self.top_n = top_n
        self.search = search
        self.show_tqdm = show_tqdm
        self.dummy_mode = dummy_mode
        self.dummy_file = dummy_file

        self.rng = random.Random(seed)
        self.next_line = method in {"general", "example"}

        self.description = DescriptionProcessor()
        self.intent_retrieval: IntentRetriever | None = None
        self.full_intent_df: pl.DataFrame = pl.DataFrame()
        self.unique_intents: List[str] = []

        logger.info("✅ IntentGenerator initialized")

    def create_intent_prompt(self, intents: list[str] | str, trunc: str, message_count: int) -> str:
        if intents == "None":
            intent_list = "None"
        elif self.next_line:
            intent_list = ", \n".join(intents)
        else:
            intent_list = ", ".join(intents)
        return self.base_prompt.format(intent_list=intent_list, message_count=message_count, trunc=trunc).strip()

    def parse_intent_output(self, messages: List[str], llm_output: str, *, index_value: List[int]) -> pl.DataFrame:
        valid_pattern = r"^[a-zA-Z0-9_'\-]+$"

        def _clean(text: str) -> str:
            text = "\n".join(l for l in text.split("\n") if l.strip())
            return text.replace("#", "")

        text = _clean(llm_output)

        intents_data = []
        count = 0
        for line in text.split("\n"):
            line = line.strip()
            if "Coupled intents:" in line:
                continue
            if line.startswith("-") or re.match(r"^\d+\.", line):
                line = line.lstrip("-").strip()
                line = re.sub(r"^\d+\.\s*", "", line)
                if count >= len(index_value):
                    break
                intent_str = line[1:-1].strip() if line.startswith("[") and line.endswith("]") else line
                validated_intent = intent_str if re.fullmatch(valid_pattern, intent_str) else None
                intents_data.append((index_value[count], validated_intent, count))
                count += 1

        df = pl.DataFrame(intents_data, schema=[("index", pl.Int64), ("intent", pl.Utf8), ("count", pl.Int64)], orient="row") if intents_data else pl.DataFrame(schema=["index", "intent", "count"])
        if len(df) == len(messages):
            df = df.with_columns(pl.Series("messages", messages))
        return df

    @staticmethod
    def format_messages(messages: List[str]) -> str:
        return "\n".join(f"{i+1}. {msg}" for i, msg in enumerate(messages))

    @staticmethod
    def replace_numbered_lines(text: str) -> str:
        return re.sub(r"^\d+\.", "-", text, flags=re.MULTILINE)

    @staticmethod
    def remove_semicolon(text: str) -> str:
        return "\n".join(re.sub(r":[^,]+", "", l) if l.strip().startswith("-") else l for l in text.split("\n"))

    def sample_intents(self, indices: List[int]) -> list[str] | str:
        if not self.unique_intents:
            return "None"
        return self.intent_retrieval.retrieve_top_intents(indices, top_k=self.top_k)

    def _try_generate_intents(self, conv_idx: int, indices: List[int], messages: List[str]) -> Tuple[pl.DataFrame | None, int]:
        total = len(messages)
        msg_str = self.format_messages(messages)
        tokens_used = 0

        for attempt in range(MAX_RETRIES):
            def _call():
                prompt = self.create_intent_prompt(self.sample_intents(indices), msg_str, total)
                temp = 0.7 if attempt < 4 else 0.9
                logger.debug(f"📝 Prompt for conv {conv_idx} (attempt {attempt+1}):\n{prompt}")
                return call_GPT(prompt, temperature=temp, return_token_count=True)

            try:
                result, tok = _run_with_timeout(_call, TIMEOUT_S)
                logger.debug(f"🤖 GPT output (conv {conv_idx}, attempt {attempt+1}):\n{result}")
                tokens_used += tok
                cleaned = self.remove_semicolon(self.replace_numbered_lines(result))
                df = self.parse_intent_output(messages, cleaned, index_value=indices)
                if len(df) != total:
                    logger.warning("⚠️ Parsed row count mismatch (conv %s, attempt %s)", conv_idx, attempt + 1)
                    continue
                if self.method == "general":
                    _, delta = self.description.update(df, column="intent")
                    tokens_used += delta
                return df, tokens_used
            except Exception as e:
                logger.exception("🔥 Error during intent generation (conv %s, attempt %s): %s", conv_idx, attempt + 1, e)

            time.sleep(self.api_time)
        return None, tokens_used

    def run_intent_generation(self, conv_split):
        logger.info("🚀 Starting intent generation...")

        # --- DUMMY MODE SHORTCUT ---
        if self.dummy_mode and os.path.exists(self.dummy_file):
            logger.info(f"📂 Dummy mode: loading saved intents from {self.dummy_file}")
            self.full_intent_df = pl.read_csv(self.dummy_file)
            self.unique_intents = self.full_intent_df["intent"].unique().to_list()
            return self.full_intent_df, self.unique_intents

        # --- normal flow ---
        conv_df = pl.from_pandas(conv_split) if not isinstance(conv_split, pl.DataFrame) else conv_split
        conv_df = conv_df.sort("conversation_number")
        grouped = conv_df.group_by("conversation_number").agg(pl.col("index"), pl.col("conversation"))

        if self.n_samples and self.n_samples < len(grouped):
            sample_ids = self.rng.sample(range(len(grouped)), self.n_samples)
            grouped = grouped[sample_ids]
            logger.info(f"🔎 Sampled {self.n_samples} conversations")

        all_indices = grouped["index"].to_list()
        all_convs = grouped["conversation"].to_list()
        flat_idx = [i for sub in all_indices for i in sub]
        flat_conv = [m for sub in all_convs for m in sub]

        logger.info("⚙️ Initializing IntentRetriever...")
        self.intent_retrieval = IntentRetriever(indices=flat_idx, conversation_list=flat_conv)
        logger.info("✅ IntentRetriever initialized.")

        conv_iterator = tqdm(
            enumerate(zip(all_indices, all_convs)),
            total=len(all_convs),
            desc="Conversations",
            ncols=80,
            leave=True
        )

        for i, (idx_list, msg_list) in conv_iterator:
            df_result, _ = self._try_generate_intents(i, idx_list, msg_list)
            if df_result is not None:
                self.full_intent_df = pl.concat([self.full_intent_df, df_result])
                self.intent_retrieval.update(df_result.to_pandas())
                self.unique_intents = self.full_intent_df["intent"].unique().to_list()

        logger.info("🎉 Intent generation complete.")

        # --- SAVE RESULTS IN DUMMY MODE ---
        if self.dummy_mode:
            self.full_intent_df.write_csv(self.dummy_file)
            logger.info(f"💾 Saved dummy results to {self.dummy_file}")

        return self.full_intent_df, self.unique_intents

