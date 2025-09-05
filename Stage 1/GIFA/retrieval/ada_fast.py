import numpy as np
import pandas as pd
import time
import requests
import json
import os
from tqdm import tqdm
import faiss  # Import faiss
from typing import Optional, List, Any

class ADA03:
    def __init__(self, conversations_list, indices, csv_path="github_code/GIFA/retrieval/embeddings.csv"):
        self.corpus = conversations_list
        self.indices = indices
        self.csv_path = csv_path
        self.int_set = set()  # Will be synced by the retriever

        self.message_pos_map = {index_val: i for i, index_val in enumerate(indices)}
        self.embeddings_df = self.load_and_generate_embeddings()

        self.faiss_index = None
        self.faiss_id_to_original_index = []

        if not self.embeddings_df.empty:
            self._build_faiss_index()

    def call_ADA_003(self, msg):
        """Calls ADA-003 API and returns the embedding as a list."""

        # FILL IN WITH YOUR OWN API OR OTHER EMBEDDER MODEL
        url = ""
        headers = ""
        data = {"input": [msg], "dimensions": 1024}  # ensure consistent size
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"Error in API call: {e}")
            return []

    def _safe_parse_embedding(self, x: Any):
        """Parse embedding cell to a Python list[float] if it's a JSON string."""
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                # last-resort: handle malformed JSON-ish strings
                import ast
                return ast.literal_eval(x)
        return x  # already a list

    def _coerce_list_float(self, v: Any) -> Optional[List[float]]:
        """Ensure the value is a flat list of floats; return None if not coercible."""
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 1:
                    return arr.tolist()
            except Exception:
                return None
        return None

    def load_and_generate_embeddings(self):
        """
        Loads embeddings from a CSV file. If any are missing for the given
        corpus, fetch them using the ADA API and update the CSV.
        Also ensures unique indices and consistent embedding dimensions.
        """
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            # parse strings -> lists safely
            if "embedding" in df.columns:
                df["embedding"] = df["embedding"].apply(self._safe_parse_embedding)
        else:
            print("No existing embeddings found. Creating new ones...")
            df = pd.DataFrame(columns=["index", "embedding"])

        # De-duplicate by index (keep the latest)
        if not df.empty and "index" in df.columns:
            df = df.drop_duplicates(subset=["index"], keep="last")

        # Coerce to flat list[float] and drop invalid rows
        df["embedding"] = df["embedding"].apply(self._coerce_list_float)
        df = df.dropna(subset=["embedding"]).reset_index(drop=True)

        # Optional: enforce a single embedding dimension across DF
        if not df.empty:
            lengths = df["embedding"].apply(lambda e: len(e))
            # Choose the most common length
            target_dim = lengths.mode().iloc[0]
            df = df[lengths == target_dim].copy()

        existing_indices = set(df["index"].tolist()) if "index" in df.columns else set()
        missing_indices = [idx for idx in self.indices if idx not in existing_indices]

        if missing_indices:
            print(f"Fetching {len(missing_indices)} missing embeddings...")
            new_embeddings = []
            for idx in tqdm(missing_indices, desc="Fetching Embeddings"):
                time.sleep(0.5)
                pos = self.message_pos_map.get(idx)
                if pos is None:
                    continue

                embed = self.call_ADA_003(self.corpus[pos])
                if embed:
                    new_embeddings.append({"index": idx, "embedding": embed})

            if new_embeddings:
                new_df = pd.DataFrame(new_embeddings)
                df = pd.concat([df, new_df], ignore_index=True)
                # Re-apply de-dup and dimension filter after adding
                df = df.drop_duplicates(subset=["index"], keep="last")
                df["embedding"] = df["embedding"].apply(self._coerce_list_float)
                df = df.dropna(subset=["embedding"]).reset_index(drop=True)
                if not df.empty:
                    lengths = df["embedding"].apply(lambda e: len(e))
                    target_dim = lengths.mode().iloc[0]
                    df = df[lengths == target_dim].copy()

                df.to_csv(self.csv_path, index=False)
                print(f"Updated embeddings saved to {self.csv_path}.")

        if "index" in df.columns and not df.empty:
            df.set_index("index", inplace=True, drop=False)
        return df

    def _build_faiss_index(self):
        """Builds a faiss index from the embeddings in the DataFrame."""
        unique_df = self.embeddings_df[~self.embeddings_df.index.duplicated(keep='first')]

        # Build a clean matrix
        mat = []
        ids = []
        for idx, row in unique_df.iterrows():
            emb = row["embedding"]
            # ensure proper dtype/shape
            if isinstance(emb, (list, tuple, np.ndarray)):
                arr = np.asarray(emb, dtype=np.float32)
                if arr.ndim == 1:
                    mat.append(arr)
                    ids.append(idx)

        if not mat:
            self.faiss_index = None
            self.faiss_id_to_original_index = []
            return

        embeddings = np.vstack(mat).astype('float32')
        faiss.normalize_L2(embeddings)
        dimensions = embeddings.shape[1]

        self.faiss_index = faiss.IndexFlatIP(dimensions)  # IP = Inner Product (cosine when normalized)
        self.faiss_id_to_original_index = ids
        self.faiss_index.add(embeddings)

    def query_embedding(self, int_index) -> Optional[List[float]]:
        """Retrieves a single embedding list for a given index, robust to duplicates."""
        try:
            rows = self.embeddings_df.loc[int_index, "embedding"]
        except KeyError:
            return None

        # If duplicates exist, .loc returns a Series; take the last one
        if isinstance(rows, pd.Series):
            rows = rows.iloc[-1]

        emb = self._coerce_list_float(rows)
        return emb

    def create_ranking(self, query_int_number, top_k):
        """
        Ranks indices using FAISS, constrained to search only within
        self.int_set, using cosine similarity.
        """
        if self.faiss_index is None or not self.int_set:
            return [], []

        q_embed = self.query_embedding(query_int_number)
        if q_embed is None:
            return [], []

        query_vector = np.asarray(q_embed, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)

        original_index_to_faiss_id = {idx: i for i, idx in enumerate(self.faiss_id_to_original_index)}
        searchable_faiss_ids = [original_index_to_faiss_id[idx] for idx in self.int_set if idx in original_index_to_faiss_id]

        if not searchable_faiss_ids:
            return [], []

        selector = faiss.IDSelectorArray(np.array(searchable_faiss_ids, dtype='int64'))

        distances, faiss_ids = self.faiss_index.search(
            query_vector,
            top_k,
            params=faiss.SearchParameters(sel=selector)
        )

        retrieved_indices = [self.faiss_id_to_original_index[i] for i in faiss_ids[0] if i != -1]
        final_indices = [idx for idx in retrieved_indices if idx != query_int_number]

        return final_indices, []
