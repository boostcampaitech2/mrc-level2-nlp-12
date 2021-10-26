import os
import json
import datasets
import time
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from rank_bm25 import (  # https://github.com/dorianbrown/rank_bm25
    BM25Okapi,
    BM25L,
    BM25Plus,
)

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DPRDataset:
    def __init__(self, wiki_data_path, datasets_path):
        self.wiki_data_path = wiki_data_path
        self.datasets_path = datasets_path
        self.wiki = None
        self.dataset = None

    def load_wiki_data(self, path=None):
        if path == None:
            path = self.wiki_data_path

        with open(path, "r") as f:
            self.wiki = json.load(f)

    def load_train_data(self, path=None):
        if self.dataset == None:
            if path == None:
                path = self.datasets_path
            if os.path.isdir(path):
                self.dataset = datasets.load_from_disk(path)
            else:
                self.dataset = datasets.load_dataset(path)
        train_datasets = self.dataset["train"]
        return train_datasets

    def load_valid_data(self, path=None):
        if self.dataset == None:
            if path == None:
                path = self.datasets_path
            if os.path.isdir(path):
                self.dataset = datasets.load_from_disk(path)
            else:
                self.dataset = datasets.load_dataset(path)
        valid_datasets = self.dataset["validation"]
        return valid_datasets


class BM25Data:
    def __init__(
        self,
        tokenize_fn,
        wiki_data=None,
        data_path: Optional[str] = "../../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.
        """
        self.data_path = data_path
        # with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
        #     wiki = json.load(f)

        self.wiki = wiki_data
        self.contexts = [v["text"] for v in self.wiki.values()]
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # BM25
        self.tokenize_fn = tokenize_fn
        self.bm25 = None

    def get_sparse_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고
            bm25를 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        # Pickle을 저장합니다.
        bm25_name = f"bm25.bin"
        bm25_path = os.path.join(self.data_path, bm25_name)

        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            with timer("bm25"):
                tokenized_contexts = [self.tokenize_fn(c) for c in self.contexts]
                self.bm25 = BM25Plus(tokenized_contexts, k1=1.2, b=0.75)
            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1,
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.bm25 is not None, f"get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            # print("[Search query]\n", query_or_dataset, "\n")

            # for i in range(topk):
            #     print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
            #     print(self.contexts[doc_indices[i]])
            # print(doc_indices)
            return (
                doc_scores,
                [self.contexts[doc_indices[i]] for i in range(topk)],
                doc_indices,
            )

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("bulk query by exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        # with timer("single query by exhaustive search"):
        result = self.bm25.get_scores(self.tokenize_fn(query))
        assert (
            np.sum(result) != 0
        ), f"오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]

        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        result = [self.bm25.get_scores(self.tokenize_fn(q)) for q in queries]

        if not isinstance(result, np.ndarray):
            result = np.array(result)

        doc_scores = np.partition(result, -k)[:, -k:][:, ::-1]
        ind = np.argsort(doc_scores, axis=-1)[:, ::-1]
        doc_scores = np.sort(doc_scores, axis=-1)[:, ::-1]
        doc_indices = np.argpartition(result, -k)[:, -k:][:, ::-1]
        r, c = ind.shape
        ind = ind + np.tile(np.arange(r).reshape(-1, 1), (1, c)) * c
        doc_indices = doc_indices.ravel()[ind].reshape(r, c)

        return doc_scores, doc_indices

