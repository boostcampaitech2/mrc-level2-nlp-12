import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import pickle
import os
import time

from datasets import (
    Value,
    Features,
    Dataset,
    DatasetDict,
)
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

# [docs] https://www.sbert.net/docs/training/overview.html
# [experiment] https://www.notion.so/sentence-transformers-v2-ff935f530e494aa183c4e0560a1f8ffe
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)

class StRetrieval():
    '''A class for training encoder and retrieve docs based on sentence-transformers
    '''
    def __init__(self, model_name='klue/bert-base', num_neg=30):
        set_seed(42)

        # train fields
        self.train_df = None
        self.q_list = None
        self.p_list = None
        self.train_dataloader = None

        # test and wiki fields
        self.query_or_dataset = None
        self.contexts = None

        self.model = None
        self.q_embedding = None
        self.p_embedding = None

        self.num_neg = num_neg
        self.model_name = model_name
        self.p_with_neg = []

    def _load_train_dataset(self, DATASET_PATH = '/opt/ml/data/train_dataset/'):
        '''
        A function for loading train dataset to train encoder model

        Args:
            DATASET_PATH (str): a path of train dataset
        '''
        dataset = load_from_disk(DATASET_PATH)
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

        self.q_list = self.train_df['question'].tolist()
        self.p_list = self.train_df['context'].tolist()

    def _set_negative_samples(self):
        '''
        A function for sampling negative sentences randomly
        '''
        corpus = np.array(self.p_list)
        for c in self.p_list:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.num_neg) # random choice
                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    self.p_with_neg.append(c)
                    self.p_with_neg.extend(p_neg)
                    break

        # pos and neg check
        print("[Query Given]")
        print(self.train_df["question"])

        print("\n[Positive Context]")
        print(self.p_with_neg[31]) # if num_neg=30 and idx=31 => a positive context for second query (q_list[1])

        print("\n[Negative Context]")
        print(self.p_with_neg[32])
        print(self.p_with_neg[33])

    def _set_train_dataloader(self):
        '''
        A function for getting train dataloader (positive sentences 1 + negative sentences num_neg)
        '''
        train_examples = []
        for i in range(len(self.p_with_neg)):
            # pos
            if i % (self.num_neg + 1) == 0:
                train_examples += [InputExample(texts=[self.q_list[i // (self.num_neg + 1)],
                                                       self.p_with_neg[i]], label=0.9)] # label => similarity
            # neg
            else:
                train_examples += [InputExample(texts=[self.q_list[i // (self.num_neg + 1)],
                                                       self.p_with_neg[i]], label=0.1)]
        self.train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8) # default

        print('--- Train Dataloader Lengths ---')
        print(len(self.train_dataloader))

    def _train(self):
        '''
        A function for training new encoder model
        '''

        # model consists of two parts => word_embedding + pooling_model
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        # bias=False
        d1 = models.Dense(word_embedding_model.get_word_embedding_dimension(),
                          512, # todo question lengths might be short
                          bias=False,
                          activation_function=nn.Identity())
        d2 = models.Dense(word_embedding_model.get_word_embedding_dimension(),
                          512,
                          bias=False,
                          activation_function=nn.Identity())
        asym_model = models.Asym({'QRY': [d1], 'DOC': [d2]})

        # (doc1, doc2) semantic search => symmetric (recommended based on documentation)
        # (query, docs) retrieval => asymmetric (recommended based on documentation)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, asym_model])
        train_loss = losses.CosineSimilarityLoss(self.model) # default
        self.model.fit(train_objectives=[(self.train_dataloader, train_loss)], epochs=1, warmup_steps=100) # default

        # todo save model => to hub, to local

    def _load_test_dataset(self):
        '''
        A function for loading test dataset to embed
        '''
        dataset = load_from_disk('/opt/ml/data/test_dataset/')
        test_df = pd.DataFrame({"question": dataset['validation']['question'], "id": dataset['validation']['id']})
        self.query_or_dataset = test_df['question'].tolist()

        wiki_data_path = '/opt/ml/data/wikipedia_documents.json'
        wiki_df = pd.read_json(wiki_data_path, orient='index')
        wiki_df.drop_duplicates(subset=['text'], inplace=True)
        self.contexts = wiki_df['text'].tolist()

    def _encode(self):
        '''
        A function for encoding (embedding) questions from test_dataset and passages from wikipedia
        '''
        self._load_test_dataset()

        if self.model is None:
            raise ValueError(' [ Model Not Exist ] 학습된 모델이 없습니다.')

        self.q_embedding = self.model.encode(self.query_or_dataset)
        self.p_embedding = self.model.encode(self.contexts)

        print('--- Check Embedding Shapes ---')
        print('q_embedding: ', self.q_embedding.shape)
        print('p_embedding: ', self.p_embedding.shape)

        # save embedding bin files
        with open('/opt/ml/code/p_embedding.bin', 'wb') as f: # path might be different (caution)
            pickle.dump(self.p_embedding, f)

        with open('/opt/ml/code/q_embedding.bin', 'wb') as f:
            pickle.dump(self.q_embedding, f)

    def _get_relevant_doc_bulk(self, topk=30):
        '''
        A function for retrieving passages by scores from dot product of q_embedding, p_embedding
        '''
        if self.q_embedding is None or self.p_embedding is None:
            print('--- Check Saved Embedding Files ---')
            if os.path.isfile('/opt/ml/code/q_embedding.bin') and os.path.isfile('/opt/ml/code/p_embedding.bin'):
                with open('/opt/ml/code/q_embedding.bin', 'rb') as f:
                    self.q_embedding = pickle.load(f)
                with open('/opt/ml/code/p_embedding.bin', 'rb') as f:
                    self.p_embedding = pickle.load(f)
            else:
                raise ValueError(' [ Embedding Files Not Found ] 저장된 embedding file 이 없습니다.')

        hits = self.q_embedding @ self.p_embedding.T # dot product
        hits = torch.argsort(hits, dim=1, descending=True)

        doc_indices = []
        for i in range(len(hits)):
            tmp = []
            for j in range(topk):
                tmp += [hits[i][j]]
            doc_indices.append(tmp)

        print('--- Check Doc Indices by ST ---')
        print(doc_indices[0])
        return doc_indices

    def _retrieve(self, query_or_dataset, topk=30):
        '''
        A function for retrieving docs based on topk scores

        Args:
            query_or_dataset (dataset): queries
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
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        print('--- Check Retrieve Result ---')
        print(total[0])

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