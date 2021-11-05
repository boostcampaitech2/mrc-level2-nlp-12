import os
import json
import pprint
import time
import pandas as pd

from contextlib import contextmanager
from typing import Optional
from elasticsearch import Elasticsearch
from tqdm import tqdm
from .. import preprocess


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# https://lucene.apache.org/core/8_9_0/analyzers-nori/org/apache/lucene/analysis/ko/POS.Tag.html
# https://www.notion.so/ES-24cfe4e24deb45aca4a10dc806882a65
# https://www.elastic.co/guide/index.html
class DFRRetriever():
    '''A class for providing elasticsearch (DFR) retrieval
    '''
    def __init__(self):
        self.index_name = 'dev' # change if you want
        self.index_settings = self._set_index_setting()
        self.contexts = None
        self.es = None

    def _proc_init(self):
        '''
        A function for proceeding whole indexing process
        '''
        self._set_context()
        self._set_instance()
        docs = self._convert_context_to_dict() # wiki
        self._set_index()
        self._proc_indexing(docs)

    def _set_context(self,
                     data_path: Optional[str] = "../data/",
                     context_path: Optional[str] = "wikipedia_documents.json"):
        '''
        A function for loading context from wikipedia
        '''
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )

        # optional
        # preprocessing = preprocess.Preprocess(self.contexts, ['russian', 'arabic'])
        # preprocessing._proc_preprocessing()
        # self.contexts = preprocessing.sents
        # print('--- Preprocessed Contexts Lengths ---')
        # print(len(self.contexts))

    def _set_index_setting(self):
        '''
        A function for setting indexing method (customization recommended)
        '''
        return {
            "settings": {
                "index": {
                    "similarity": {
                        "my_similarity": {
                            "type": "DFR", # DFR similarity
                            "basic_model": "g",
                            "after_effect": "l",
                            "normalization": "h2",
                            "normalization.h2.c": "3.0"
                        }
                    },
                    "analysis": {
                        "analyzer": {
                            "my_analyzer": {
                                "tokenizer": "tokenizer_discard_punctuation_false",
                                "filter": [
                                    "nori_number", "pos_stop_sp"
                                ],
                            }
                        },
                        "tokenizer": {
                            "tokenizer_discard_punctuation_false": {
                                "type": "nori_tokenizer",
                                "discard_punctuation": "false"
                            }
                        },
                        "filter": {
                            "pos_stop_sp": {
                                "type": "nori_part_of_speech",
                                "stoptags": ["J"]
                            }
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "my_analyzer",
                        "search_analyzer": "my_analyzer",
                        "similarity": "my_similarity"
                    }
                }
            }
        }

    def _set_instance(self):
        '''
        A function for initiating elasticsearch
        '''
        try:
            self.es.transport.close()
        except:
            pass
        self.es = Elasticsearch()
        print('--- Check Info---')
        print(self.es.info())

    def _convert_context_to_dict(self):
        '''
        A function for converting contexts to dictionary
        '''
        docs = dict()
        for i in range(1, len(self.contexts) + 1):
            docs[i] = {
                "content": self.contexts[i - 1]
            }
        return docs

    def _set_index(self):
        '''
        A function for creating index
        '''
        if self.es.indices.exists(self.index_name):
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(index=self.index_name, body=self.index_settings)

    def _proc_indexing(self, docs):
        '''
        A function for indexing process

        Args:
            docs (dict): consists of contexts
        '''
        for i, doc in tqdm(docs.items()):
            self.es.index(index=self.index_name, id=i, body=doc)

        print('--- Check Indexing Result ---')
        d = self.es.get(index=self.index_name, id=1)
        pprint.pprint(d)

    def _get_analyzed_query_result(self, query):
        '''
        A function for checking analyzed query result

        Args:
            query (str): question query
        '''
        res = self.es.indices.analyze(
            index=self.index_name,
            body={
                "analyzer": "my_analyzer",
                "text": query
            })
        print('--- Check Query Analyzed Result ---')
        pprint.pprint(res)

    def _search_docs(self, query, topk=5):
        '''
        A function for searching docs from query

        Args:
            query (str): question query
        '''
        res = self.es.search(index=self.index_name, q=query, size=topk)
        print('--- Check Search Result ---')
        pprint.pprint(res)

    def _get_relevant_doc_bulk(self, query_or_dataset, topk):
        '''
        A function for getting relevant docs
        '''
        doc_indices = []
        for i in range(len(query_or_dataset)):
            res = self.es.search(index=self.index_name, q=query_or_dataset[i], size=topk)
            doc_indices.append(self._get_doc_ids(res))

        print('--- Check Doc Indices ---')
        print(doc_indices[0])
        return doc_indices

    def _get_doc_ids(self, res):
        '''
        A function for extracting doc ids from result of ES search

        Args:
            res (dict) : a raw result
        '''
        docs = res['hits']['hits']
        doc_indices = [int(tmp['_id']) - 1 for tmp in docs]
        return doc_indices

    def retrieve(self, query_or_dataset, topk=5):
        '''
        A function for retrieving docs

        Args:
            query_or_dataset (dataset): question queries
            topk (int): a number of searching docs (default 5)
        '''
        total = []
        with timer("query dfr search"):
            doc_indices = self._get_relevant_doc_bulk(query_or_dataset['question'], topk=topk)
        for idx, example in enumerate(tqdm(query_or_dataset, desc="dfr retrieval: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context_id": doc_indices[idx],
                "context": " ".join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)
        df = pd.DataFrame(total)
        return df
