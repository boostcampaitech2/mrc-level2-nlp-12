import os
import json
import pprint
import time
import pandas as pd

from datasets import (
    Value,
    Features,
    Dataset,
    DatasetDict,
)
from preprocess import *
from contextlib import contextmanager
from typing import Optional
from elasticsearch import Elasticsearch
from tqdm import tqdm

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# https://www.notion.so/ES-24cfe4e24deb45aca4a10dc806882a65
class EsRetrieval():
    '''A class for providing ES-based retrieval
    '''
    def __init__(self, args):
        self.index_name = args.index_name
        self.index_settings = self._create_index_settings()
        self.contexts = None # default from wiki
        self.es = None

    def _proc_basic_setting(self):
        '''
        A function for proceeding whole indexing process
        '''
        self._set_contexts()
        self._create_es_instance()
        docs = self._convert_contexts_to_dict()
        self._create_index()
        self._proc_indexing(docs)

    def _set_contexts(self,
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
        stopword_df = pd.read_table('korean_stopwords.txt', delimiter='\t', encoding='utf-8', header=None)
        stopword = stopword_df[0].tolist()
        preprocess = Preprocess(self.contexts, ['russian', 'arabic'], stopwords=stopword)
        preprocess.proc_preprocessing()
        self.contexts = preprocess.sents
        print('--- Preprocessed Contexts Lengths ---')
        print(len(self.contexts))

    def _create_index_settings(self):
        '''
        A function for setting indexing method (customization recommended)
        '''
        return {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": {
                            "my_analyzer": {
                                "tokenizer": "tokenizer_discard_punctuation_false",
                                "filter": [ # filter enroll
                                    "nori_number", "pos_stop_sp"
                                ],
                            }
                        },
                        "tokenizer": { # define custom tokenizer
                            "tokenizer_discard_punctuation_false": {
                                "type": "nori_tokenizer",
                                "discard_punctuation": "false" # whether discard punct or not
                            }
                        },
                        "filter": { # define custom filter
                            # https://lucene.apache.org/core/8_9_0/analyzers-nori/org/apache/lucene/analysis/ko/POS.Tag.html
                            "pos_stop_sp": {
                                "type": "nori_part_of_speech",
                                "stoptags": ["J", "NA", "SSO", "SSC", "SH", "SL", "SY"] # tags should be removed
                            }
                        }
                    }
                }
            },
            "mappings": {
                "properties": { # define docs format
                    "content": {
                        "type": "text",
                        "analyzer": "my_analyzer",
                        "search_analyzer": "my_analyzer",
                    }
                }
            }
        }

    def _create_es_instance(self):
        '''
        A function for initiating ES
        '''
        try:
            self.es.transport.close()
        except:
            pass
        self.es = Elasticsearch()
        print('--- Check ES Info---')
        print(self.es.info())

    def _convert_contexts_to_dict(self):
        '''
        A function for converting contexts to dictionary
        '''
        docs = dict()
        for i in range(1, len(self.contexts) + 1):
            docs[i] = {
                "content": self.contexts[i - 1]
            }
        return docs

    def _create_index(self):
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

        print('--- Check Result ---')
        d = self.es.get(index=self.index_name, id=1) # first doc id = 1 (caution)
        pprint.pprint(d)

    def _check_query_analyzed_result(self, q):
        '''
        A function for checking query-analyzed result

        Args:
            q (str): question query
        '''
        res = self.es.indices.analyze(
            index=self.index_name,
            body={
                "analyzer": "my_analyzer",
                "text": q
            })
        print('--- Check Query Analyzed Result ---')
        pprint.pprint(res)

    def _search_docs(self, q, topk=5):
        '''
        A function for searching docs from query

        Args:
            q (str): question query
            topk (int): a number of searching docs
        '''
        res = self.es.search(index=self.index_name, q=q, size=topk) # topk 5
        print('--- Check Search Result ---')
        pprint.pprint(res)

    def _get_relevant_doc_bulk(self, query_or_dataset, topk=5):
        '''
        A function for getting relevant docs on queries

        Args:
            query_or_dataset (dataset): queries
            topk (int): default 5
        '''
        doc_indices = []
        for i in range(len(query_or_dataset)):
            res = self.es.search(index=self.index_name, q=query_or_dataset[i], size=topk)  # topk 5
            doc_indices.append(self._get_doc_ids(res))

        print('--- Check Doc Indices by ST ---')
        print(doc_indices[0])
        return doc_indices

    def _get_doc_ids(self, res):
        '''
        A function for extracting doc ids from result of ES search

        Args:
            res : raw result of ES search
        '''
        docs = res['hits']['hits']
        doc_indices = [int(tmp['_id']) - 1 for tmp in docs] # -1 => ES index to real doc index
        return doc_indices

    def _retrieve(self, query_or_dataset, topk=5):
        '''
        A function for retrieving docs based on topk scores

        Args:
            query_or_dataset (dataset): queries
        '''
        total = []
        with timer("query elastic search"):
            doc_indices = self._get_relevant_doc_bulk(query_or_dataset['question'], topk=topk)
        for idx, example in enumerate(tqdm(query_or_dataset, desc="ES retrieval: ")):
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
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets