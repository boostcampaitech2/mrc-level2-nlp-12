import os
import json
import time
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore # https://github.com/deepset-ai/haystack

from elasticsearch import Elasticsearch

from datasets import Dataset

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


class EsBm25Retriever:
    def __init__(
        self,
        index: Optional[str] = "wikipedia",
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Summary:
            Retriever that carries out sparse retrieval with BM25_okapi algorithm using elasticsearch.

        Args:
            index (Optional[str]): elasticsearch index name. Defaults to "wikipedia".
            data_path (Optional[str]): data path. Defaults to "../data/".
            context_path (Optional[str]): context file path. Defaults to "wikipedia_documents.json".
        """

        wiki_data_path = os.path.join(data_path, context_path)
        wiki_df = pd.read_json(wiki_data_path, orient='index')
        self.contexts = wiki_df['text'].tolist()
        self.document_store = None

        if Elasticsearch('localhost:9200').indices.exists(index):
            self.document_store = ElasticsearchDocumentStore(
                host="localhost",
                port=9200,
                index=index,
                create_index=False,
            )
        else:
            self._create_index(index)
            self._store_data(wiki_df)
            
    def _create_index(self, index: str) -> NoReturn:
        """
        Summary:
            create elasticsearch index.

        Args:
            index (str): elasticsearch index name.
        """

        custom_mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "my_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer",
                            "decompound_mode": "mixed",
                            "filter": [
                                "nori_readingform", # 한자 -> 한글로 번역
                                "nori_number", # 한글 -> 숫자 ex) 영영칠 -> 7
                                "nori_posfilter", # 품사 제거 https://coding-start.tistory.com/167
                            ]
                        }
                    },
                    "filter":{
                        "nori_posfilter":{
                            "type":"nori_part_of_speech",
                            "stoptags":[
                                "E",
                                "IC",
                                "J",
                                "MAG",
                                "MM",
                                "NA",
                                "NR",
                                "SC",
                                "SE",
                                "SF",
                                "SH",
                                "SL",
                                "SN",
                                "SP",
                                "SSC",
                                "SSO",
                                "SY",
                                "UNA",
                                "UNKNOWN",
                                "VA",
                                "VCN",
                                "VCP",
                                "VSV",
                                "VV",
                                "VX",
                                "XPN",
                                "XR",
                                "XSA",
                                "XSN",
                                "XSV"
                            ]
                        }
                    }
                },
                "similarity": {
                    "bm25": {
                        "type": "BM25",
                        "b": 0.75,
                        "k1": 1.2,
                    }
                }
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "my_analyzer",
                        "similarity": "bm25"
                    }
                }
            }
        }

        self.document_store = ElasticsearchDocumentStore(
            host="localhost",
            port=9200,
            index=index,
            custom_mapping=custom_mapping,
            create_index=True
        )

    def _store_data(
        self, wiki_df: pd.DataFrame
    ) -> NoReturn:
        """
        Summary:
            store data in elasticsearch.

        Args:
            wiki_df (pd.DataFrame): wikipedia data in shape of pandas dataframe.
        """
        wiki_df = wiki_df.drop_duplicates(subset=['text'], keep='first')

        wiki = []
        for doc in wiki_df.iterrows():
            element = {
                "id": str(doc["doucment_id"]),
                "text": doc["text"],
                "meta": {
                    "name": doc["title"]
                }
            }
            wiki.append(element)

        self.document_store.write_documents(wiki)

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

        doc_scores, doc_indices = [], []
        for query in queries:
            results = self.document_store.query(query=query, top_k=topk)

            candidates = {}
            for query_result in results:
                doc_idx = query_result.to_dict()['id']
                candidates[doc_idx] = query_result.to_dict()['score']

            candidates = sorted([(score, idx) for idx, score in candidates.items()])[::-1]
            candidates = np.array(candidates)

            doc_scores.append(candidates[:, 0][:topk])
            doc_indices.append(candidates[:, 1][:topk])
        
        print(f'doc indices and scores shape : ({len(doc_scores)}, {len(doc_scores[0])})')
        
        return doc_scores, doc_indices

    def retrieve(self, query_dataset: Dataset, topk: Optional[int] = 1,) -> pd.DataFrame:
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
        doc_indices = None

        with timer("query elasticsearch"):
            _, doc_indices = self._get_relevant_doc_bulk(query_dataset["question"], topk)

        print(doc_indices[0])

        for idx, example in enumerate(tqdm(query_dataset, desc="results to df: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": " ".join(
                    [self.contexts[int(d_idx)] for d_idx in doc_indices[idx]]
                ) if topk > 1 else self.contexts[int(doc_indices[idx][0])]
            }
            # if validation dataset is given
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas
