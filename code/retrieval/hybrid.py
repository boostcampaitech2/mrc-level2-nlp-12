import os
import time
from typing import NoReturn, Optional, List
from tqdm import tqdm
from contextlib import contextmanager

import pandas as pd
import numpy as np
import torch

from datasets import Dataset

from retrieval.sparse import (
    es_dfr,
    TfidfRetriever,
    Bm25Retriever,
    EsBm25Retriever,
)
from retrieval.dense import (
    st,
    dpr,
)
from arguments import DPRArguments

@contextmanager
def timer(name):
    """
    Summary:
        contextmanager to measure run time.

    Args:
        name (str): name to display.
    """

    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class HybridRetriever:
    def __init__(
        self,
        tokenize_fn,
        sparse_retrieval: Optional[str] = "BM25",
        dense_retrieval: Optional[str] = "DPR",
        dpr_args: Optional[DPRArguments] = None,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Summary:
            Retriever combined by Sparse Retriever and Dense Retriever.

        Args:
            tokenize_fn (:func:): function to tokenize texts. e.g. `lambda x: x.split(' ')`, `Huggingface Tokenizer`, `konlpy.tag의 Mecab`
            sparse_retrieval (Optional[str]): sparse retrieval type. Defaults to "BM25".
            dense_retrieval (Optional[str]): dense retrieval type. Defaults to "DPR".
            dpr_args (Optional[DPRArguments]): if dense_retrieval set to 'DPR', then need this arg. Defaults to None.
            data_path (Optional[str]): data path. Defaults to "../data/".
            context_path (Optional[str]): context file path. Defaults to "wikipedia_documents.json".
        """

        self.sparse_retriever = None
        self.dense_retriever = None
    
        if sparse_retrieval == "TFIDF":
            self.sparse_retriever = TfidfRetriever(
                tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
            )
            self.sparse_retriever.get_sparse_embedding()
        elif sparse_retrieval == "BM25":
            self.sparse_retriever = Bm25Retriever(tokenize_fn=tokenize_fn)
        elif sparse_retrieval == "ES_BM25":
            self.sparse_retriever = EsBm25Retriever()
        elif sparse_retrieval == "ES_DFR":
            self.sparse_retriever = es_dfr.DFRRetriever()
            self.sparse_retriever._proc_init()
        
        if dense_retrieval == "DPR":
            self.dense_retriever = dpr.DensePassageRetrieval(dpr_args)
            self.dense_retriever.load_passage_embedding()
        elif dense_retrieval == "ST":
            self.dense_retriever = st.STRetriever()
            self.dense_retriever._proc_init()

        print("==================== HYBRID RETRIEVAL ====================")
        print(f"SPARSE RETRIEVAL : {sparse_retrieval}")
        print(f"DENSE RETRIEVAL : {sparse_retrieval}")
        
        wiki_data_path = os.path.join(data_path, context_path)
        wiki_df = pd.read_json(wiki_data_path, orient='index')
        wiki_df = wiki_df.drop_duplicates(subset=['text'], keep='first')
        self.contexts = wiki_df['text'].tolist()
        
        print(f"Lengths of unique contexts : {len(self.contexts)}")

    def _get_relevant_doc_bulk(
        self,
        queries: list,
        topk: Optional[int] = 1,
        alpha: Optional[float] = 0.1,
        normalization: bool = False,
        weight_on_dense: bool = False,
    ) -> List[int]:
        """
        Summary:
            get a bulk of documents relevant to given query set.

        Args:
            queries (List[str]): query list.
            topk (Optional[int]): the number of documents in order of relevance. Defaults to 1.
            alpha (Optional[float]): weight rate. Defaults to 0.1.
            normalization (bool): normalize sparse and dense retrieval scores. Defaults to False.
            weight_on_dense (bool): apply weight on dense. Defaults to False.

        Returns:
            doc_indices (List[int]): topk documents' indices.
        """

        with timer("SPARSE SEARCH..."):    
            dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(queries=queries, topk=topk)
        with timer("DENSE SEARCH..."):
            sparse_scores, sparse_indices = self.sparse_retriever._get_relevant_doc_bulk(queries=queries, topk=topk)
        print("=== DENSE SHAPE ===")
        print(dense_scores[0])
        print(f'dense score: ({len(dense_scores)}, {len(dense_scores[0])})')
        print(f'dense indices: ({len(dense_indices)}, {len(dense_indices[0])})')
        print("=== SPARSE SHAPE ===")
        print(sparse_scores[0])
        print(f'sparse score: ({len(sparse_scores)}, {len(sparse_scores[0])})')
        print(f'sparse indices: ({len(sparse_indices)}, {len(sparse_indices[0])})')
        
        # https://www.koreascience.or.kr/article/JAKO201511758627325.pdf
        # https://github.com/castorini/pyserini
        doc_indices = []
        for query_idx in range(len(queries)):
            min_sparse_score = np.min(np.float((sparse_scores[query_idx])))
            max_sparse_score = np.max(np.float((sparse_scores[query_idx])))
            min_dense_score = np.min(dense_scores[query_idx].item() if torch.is_tensor(dense_scores[query_idx]) \
                                else dense_scores[query_idx])
            max_dense_score = np.max(dense_scores[query_idx].item() if torch.is_tensor(dense_scores[query_idx]) \
                                else dense_scores[query_idx])

            sparse_dict = {
                idx:score for idx, score in zip(sparse_indices[query_idx], sparse_scores[query_idx])
            }
            dense_dict = {
                idx:score for idx, score in zip(dense_indices[query_idx], dense_scores[query_idx])
            }

            hybrid_result = []
            for doc_idx in set(sparse_dict.keys()) | set(dense_dict.keys()):
                sparse_score = 0
                dense_score = 0
                
                if torch.is_tensor(doc_idx):
                    doc_idx = doc_idx.item()
                else:
                    doc_idx = np.int(doc_idx)

                if doc_idx in sparse_dict:
                    sparse_score = np.float(sparse_dict[doc_idx])
                if doc_idx in dense_dict:
                    if torch.is_tensor(dense_dict[doc_idx]):
                        dense_score = dense_dict[doc_idx].item()    
                    dense_score = dense_dict[doc_idx]
                
                if sparse_score == 0:
                    sparse_score = min_sparse_score
                if dense_score == 0:
                    dense_score = min_dense_score

                if normalization:
                    sparse_score = (sparse_score - (min_sparse_score + max_sparse_score) / 2.) \
                                    / (max_sparse_score - min_sparse_score)
                    dense_score = (dense_score - (min_dense_score + max_dense_score) / 2.) \
                                    / (max_dense_score - min_dense_score)

                score = alpha * sparse_score + dense_score if not weight_on_dense else sparse_score + alpha * dense_score

                if doc_idx >= len(self.contexts):
                    continue
                hybrid_result.append((score, doc_idx))
            
            doc_indices.append([idx for _, idx in sorted(hybrid_result, reverse=True)[:topk]])
        
        print(f'doc indices shape : ({len(doc_indices)}, {len(doc_indices[0])})')

        return doc_indices

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

        total = []
        with timer("query hybrid search..."):
            doc_indices = self._get_relevant_doc_bulk(queries=query_dataset["question"], topk=topk)

        for idx, example in enumerate(tqdm(query_dataset, desc="results to df: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": " ".join(
                    [self.contexts[int(pid)] for pid in doc_indices[idx]]
                ) if topk > 1 else self.contexts[int(doc_indices[idx][0])]
            }
            # if validation dataset is given
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas
