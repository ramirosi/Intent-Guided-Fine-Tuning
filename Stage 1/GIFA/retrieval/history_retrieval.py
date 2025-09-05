import pandas as pd
import re
from GIFA.retrieval.ada_fast import ADA03


class IntentRetriever:
    """Intent retriever that relies **exclusively** on semantic search (ADA‑03)."""

    def __init__(
        self,
        indices: list[int],
        conversation_list: list[str],
    ) -> None:
        self.indices = indices
        self.conversations_raw = self._split_conversation(conversation_list)
        self.message_pos_map = {idx: pos for pos, idx in enumerate(indices)}
        self.ada03 = ADA03(self.conversations_raw, self.indices)

        self.full_df: pd.DataFrame = pd.DataFrame()
        self.intent_dict: dict[str, set[int]] = {}
        self.index_to_intent_map: dict[int, str] = {}

    @staticmethod
    def _split_conversation(messages: list[str]) -> list[str]:
        return [re.sub(r"^(assistant:|customer:)\s*", "", m).strip() for m in messages if m.strip()]

    def update(self, new_df: pd.DataFrame) -> None:
        if new_df.empty:
            return

        self.full_df = pd.concat([self.full_df, new_df], ignore_index=True)

        for _, row in new_df.iterrows():
            intent, index = row["intent"], row["index"]
            if pd.notna(intent) and pd.notna(index):
                self.index_to_intent_map[int(index)] = str(intent)
                self.intent_dict.setdefault(str(intent), set()).add(int(index))

        self.ada03.int_set = set(self.index_to_intent_map)

    def retrieve_top_intents(
        self,
        indices: list[int],
        top_k: int = 2,
        *,
        preprocess: bool = False,
    ) -> list[str]:
        if not self.index_to_intent_map:
            return []

        retrieved_intents: set[str] = set()

        for msg_id in indices:
            msg_pos = self.message_pos_map.get(msg_id)
            if msg_pos is None:
                continue

            top_indices, _ = self.ada03.create_ranking(msg_id, top_k)

            for idx in top_indices:
                intent = self.index_to_intent_map.get(idx)
                if intent and "," not in intent:
                    retrieved_intents.add(intent)

        return list(retrieved_intents)
