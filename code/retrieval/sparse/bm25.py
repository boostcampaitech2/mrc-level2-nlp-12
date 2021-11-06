import os
import time
import json
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

# https://github.com/dorianbrown/rank_bm25
from rank_bm25 import BM25Plus
from datasets import Dataset

from utils.wiki_split import context_split

@contextmanager
def timer(name: str):
    """
    Summary:
        contextmanager to measure run time.

    Args:
        name (str): name to display.
    """

    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class Bm25Retriever:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        split_context_path: Optional[str] = "wiki_split.json"
    ) -> NoReturn:
        """
        Summary:
            Retriever that carries out sparse retrieval with BM25+ algorithm.

        Args:
            tokenize_fn (:func:): function to tokenize texts. e.g. `lambda x: x.split(' ')`, `Huggingface Tokenizer`, `konlpy.tag의 Mecab`
            data_path (Optional[str]): data path. Defaults to "../data/".
            context_path (Optional[str]): context file path. Defaults to "wikipedia_documents.json".
            split_context_path (Optional[str]) : split context file path. Defaults to "wiki_split.json"
        """
    
        self.tokenize_fn = tokenize_fn
        self.bm25 = None
        self.data_path = data_path
        self.context_path = context_path
        self.split_context_path = split_context_path
        
        wiki_data_path = os.path.join(data_path, context_path)
        wiki_df = pd.read_json(wiki_data_path, orient='index')
        self.contexts = wiki_df['text'].tolist()
        self.split_indices = None
        
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        self._get_sparse_embedding()

    def _get_sparse_embedding(self) -> NoReturn:
        """
        Summary:
            make sparse embedding, pickle, and save bm25 object. If a sparse embedding file already exists, then load the file.
        """
        
        bm25_name = f"bm25.bin"
        bm25_path = os.path.join(self.data_path, bm25_name)

        wiki_data_path = os.path.join(self.data_path, self.split_context_path)
        assert os.path.exists(wiki_data_path), f"Run wiki split first."

        wiki_json = None
        with open(wiki_data_path, "r", encoding="utf-8") as f:
            wiki_json = json.load(f)
        self.split_indices = [int(idx[:idx.index("_")]) if "_" in idx else int(idx) for idx in wiki_json]

        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build sparse embedding")
            with timer('bm25'):
                tokenized_contexts = [self.tokenize_fn(wiki_json[c]["text"]) for c in wiki_json]
                self.bm25 = BM25Plus(tokenized_contexts, k1=1.2, b=0.75)
            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("Embedding pickle saved.")
   
    def retrieve(self, query_dataset: Dataset, topk: Optional[int] = 1) -> pd.DataFrame:
        """
        Summary:
            retrieve documents relevant to given query dataset.

        Args:
            query_or_dataset (Dataset): query dataset.
            topk (Optional[int]): the number of documents in order of relevance. Defaults to 1.

        Returns:
            cqas (pd.DataFrame): retrieved documents.
        """

        if isinstance(query_dataset, Dataset):
            total = []
            doc_indices = None

            with timer("bulk query by exhaustive search"):
                _, doc_indices = self._get_relevant_doc_bulk(query_dataset["question"], topk=topk)

            for idx, example in enumerate(tqdm(query_dataset, desc="results to df: ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                # if validation dataset is given
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
    
    def _get_relevant_doc_bulk(self, queries: List[str], topk: Optional[int] = 1) -> Tuple[List[float], List[int]]:
        """
        Summary:
            get a bulk of documents relevant to given query set.

        Args:
            queries (List[str]): query list.
            topk (Optional[int]): the number of documents in order of relevance. Defaults to 1.

        Returns:
            doc_scores, doc_indices (Tuple[List[float], List[int]]): topk documents' scores and indices.
        """

        # bm25 search
        result = [self.bm25.get_scores(self.tokenize_fn(q)) for q in queries]
        assert (
            np.sum(result) != 0
        ), f"Given query does not match any of the documents."

        if not isinstance(result, np.ndarray):
            result = np.array(result)

        doc_scores = []
        doc_indices = []
        for queried_doc in result:
            candidates = {}
            for i, score in enumerate(queried_doc):
                candidates[i] = score
            
            candidates = sorted(candidates.items(), reverse=True, key = lambda item: item[1])

            scores = []
            indices = []
            for idx, score in candidates:
                if self.split_indices[idx] not in indices:
                    indices.append(self.split_indices[idx])
                    scores.append(score)
                
                if len(indices) == topk and len(scores) == topk:
                    break
            
            doc_scores.append(scores)
            doc_indices.append(indices)     
        print(f'doc indices and scores shape : ({len(doc_scores)}, {len(doc_scores[0])})')
        
        return doc_scores, doc_indices
