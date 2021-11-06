import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import time

from tqdm import tqdm
from contextlib import contextmanager
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import load_from_disk

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# https://www.notion.so/sentence-transformers-v2-ff935f530e494aa183c4e0560a1f8ffe
# https://www.sbert.net/docs/training/overview.html
class STRetriever():
    '''A class for training encoder and retrieve docs based on sentence-transformers

    Args:
        model_name (str): a model name of encoder
        num_neg (int): a number of negative sampling
    '''
    def __init__(self, model_name='klue/bert-base', num_neg=30):
        self.test_queries, self.wiki_contexts = None, None
        self.model, self.q_embedding, self.p_embedding = None, None, None
        self.train_df, self.train_queries, self.train_contexts, self.train_dataloader = None, None, None, None

        self.num_neg = num_neg
        self.model_name = model_name
        self.p_with_neg = []

    def _proc_init(self):
        self._set_train_dataset()
        self._set_negative_sampling()
        self._set_train_dataloader()
        self._train()
        self._proc_embedding()

    def _set_train_dataset(self,
                           train_data_path='/opt/ml/data/train_dataset/'):
        '''
        A function for loading train dataset to train encoder model

        Args:
            train_data_path (str): a path of train dataset
        '''
        dataset = load_from_disk(train_data_path)
        document_ids, answers, contexts, contexts, questions, titles, ids = [], [], [], [], [], [], []

        for ex in dataset['train']:
            ids.append(ex['id'])
            answers.append(ex['answers'])
            contexts.append(ex['context'])
            questions.append(ex['question'])
            titles.append(ex['title'])
            document_ids.append(ex['document_id'])

        self.train_df = pd.DataFrame({
            "doc_id": document_ids,
            "question": questions,
            "answer": answers,
            "context": contexts,
            "title": titles,
            "id": ids
        })

        self.train_queries = self.train_df['question'].tolist()
        self.train_contexts = self.train_df['context'].tolist()

    def _set_negative_sampling(self):
        '''
        A function for sampling negative sentences randomly
        '''
        corpus = np.array(self.train_contexts)
        for c in self.train_contexts:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.num_neg)
                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    self.p_with_neg.append(c)
                    self.p_with_neg.extend(p_neg)
                    break

        print("[Query]")
        print(self.train_df["question"])

        print("\n[Positive Context]")
        print(self.p_with_neg[31]) # a positive context for train_queries[1] when num_neg=30

        print("\n[Negative Context]")
        print(self.p_with_neg[32])
        print(self.p_with_neg[33])

    def _set_train_dataloader(self):
        '''
        A function for getting train dataloader | positive 1 + negative num_neg
        '''
        train_examples = []
        for i in range(len(self.p_with_neg)):
            if i % (self.num_neg + 1) == 0:
                train_examples += [InputExample(texts=[self.train_queries[i // (self.num_neg + 1)],
                                                       self.p_with_neg[i]], label=0.9)] # label => similarity
            else:
                train_examples += [InputExample(texts=[self.train_queries[i // (self.num_neg + 1)],
                                                       self.p_with_neg[i]], label=0.1)]
        self.train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

        print('--- Train Dataloader Lengths ---')
        print(len(self.train_dataloader))

    def _train(self):
        '''
        A function for training new encoder model
        '''

        # model consists of two parts | word_embedding + pooling_model
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        # bias=False
        d1 = models.Dense(word_embedding_model.get_word_embedding_dimension(),
                          120, # query => short
                          bias=False,
                          activation_function=nn.Identity())
        d2 = models.Dense(word_embedding_model.get_word_embedding_dimension(),
                          512, # context => long
                          bias=False,
                          activation_function=nn.Identity())
        asym_model = models.Asym({'QRY': [d1], 'DOC': [d2]})

        # (doc1, doc2) semantic search => symmetric (recommended by documentation)
        # (query, docs) retrieval => asymmetric (recommended by documentation)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, asym_model])
        train_loss = losses.CosineSimilarityLoss(self.model)
        self.model.fit(train_objectives=[(self.train_dataloader, train_loss)], epochs=1, warmup_steps=100)

    def _set_test_dataset(self,
                          test_data_path='/opt/ml/data/test_dataset/',
                          wiki_data_path='/opt/ml/data/wikipedia_documents.json'):
        '''
        A function for loading test dataset to embed
        '''
        dataset = load_from_disk(test_data_path)
        test_df = pd.DataFrame({"question": dataset['validation']['question'], "id": dataset['validation']['id']})
        self.test_queries = test_df['question'].tolist()

        wiki_df = pd.read_json(wiki_data_path, orient='index')
        wiki_df.drop_duplicates(subset=['text'], inplace=True)
        self.wiki_contexts = wiki_df['text'].tolist()

    def _proc_embedding(self):
        '''
        A function for embedding test queries and wiki contexts
        '''
        self._set_test_dataset()

        if self.model is None:
            raise ValueError(' [ Model Not Exist ] 학습된 모델이 없습니다.')

        # embedding
        self.q_embedding = self.model.encode(self.test_queries)
        self.p_embedding = self.model.encode(self.wiki_contexts)

        print('--- Check Embedding Shapes ---')
        print('q_embedding: ', self.q_embedding.shape)
        print('p_embedding: ', self.p_embedding.shape)

        # save
        with open('/opt/ml/code/p_embedding.bin', 'wb') as f:
            pickle.dump(self.p_embedding, f)

        with open('/opt/ml/code/q_embedding.bin', 'wb') as f:
            pickle.dump(self.q_embedding, f)

    def _get_relevant_doc_bulk(self,
                               topk=30,
                               q_embedding_path='/opt/ml/code/q_embedding.bin',
                               p_embedding_path='/opt/ml/code/p_embedding.bin'):
        '''
        A function for getting relevant docs
        '''
        if self.q_embedding is None or self.p_embedding is None:
            print('--- Check Saved Embedding Files ---')
            if os.path.isfile(q_embedding_path) and os.path.isfile(p_embedding_path):
                with open(q_embedding_path, 'rb') as f:
                    self.q_embedding = pickle.load(f)
                with open(p_embedding_path, 'rb') as f:
                    self.p_embedding = pickle.load(f)
            else:
                raise ValueError(' [ Embedding Files Not Found ] 저장된 embedding 파일이 없습니다.')

        hits = self.q_embedding @ self.p_embedding.T # dot product
        hits = torch.argsort(hits, dim=1, descending=True)

        doc_indices = []
        for i in range(len(hits)):
            tmp = []
            for j in range(topk):
                tmp += [hits[i][j]]
            doc_indices.append(tmp)

        print('--- Check Doc Indices---')
        print(doc_indices[0])
        return doc_indices

    def retrieve(self, query_or_dataset, topk=5):
        '''
        A function for retrieving docs

        Args:
            query_or_dataset (dataset): question queries
            topk (int): a number of searching docs (default 5)
        '''
        total = []
        with timer("query sentence-transformers search"):
            doc_indices = self._get_relevant_doc_bulk(topk=topk)
        for idx, example in enumerate(tqdm(query_or_dataset, desc="sentence-transformers retrieval: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context_id": doc_indices[idx],
                "context": " ".join(
                    [self.wiki_contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)
        df = pd.DataFrame(total)
        return df
