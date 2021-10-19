import os
import json
import pprint

from typing import Optional
from elasticsearch import Elasticsearch
from tqdm import tqdm

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
                                    "pos_stop_sp", "nori_number"
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
                            "pos_stop_sp": {
                                "type": "nori_part_of_speech",
                                "stoptags": ["SP"] # tags should be removed
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
        for i in range(1, len(self.contexts[:5000]) + 1):
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

    def _retrieve(self):
        pass # keep to be continued


