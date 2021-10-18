import os
import os.path as path
import json
import torch
import numpy as np
import pickle
import torch.nn.functional as F
import pandas as pd
import random

from fuzzywuzzy import fuzz
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm, trange
from transformers import TrainingArguments, BertPreTrainedModel, BertModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_from_disk, load_dataset
from datasets import Sequence, Value, Features, DatasetDict, Dataset

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)

class BertEncoder(BertPreTrainedModel):
    '''A class for encoding questions and passages
    '''

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # embedded vec
        pooled_output = outputs[1]
        return pooled_output

class Retrieval:
    '''A common class for all retrieval methods
    '''

    def __init__(self, args, tokenizer):
        set_seed(42)
        self.args = args
        self.tokenizer = tokenizer

        with open(os.path.join("/opt/ml/", "data", "wikipedia_documents.json"), "r") as f:
            wiki = json.load(f)

        # wiki context
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.context_ids = list(dict.fromkeys([v["document_id"] for v in wiki.values()]))

    def _load_encoder(self):
        p_encoder = BertEncoder.from_pretrained(self.args.model_checkpoint)
        q_encoder = BertEncoder.from_pretrained(self.args.model_checkpoint)

        # q_encoder, p_encoder => 인코더 bin 파일 존재 확인
        if path.isfile(path.join(self.args.q_encoder_path, 'pytorch_model.bin')) and path.isfile(path.join(self.args.p_encoder_path, 'pytorch_model.bin')):
            print('--- Load Encoders from Local ---')
            q_encoder = BertEncoder.from_pretrained(self.args.q_encoder_path)
            p_encoder = BertEncoder.from_pretrained(self.args.p_encoder_path)
        return q_encoder, p_encoder

    def _load_dataset(self):
        if self.args.is_dpr:
            print('--- Squad Kor Dataset Ready ---')
            dataset = load_dataset('squad_kor_v1')
            train_dataset = dataset['validation'] # 5774 개
        else:
            dataset_path = path.join(self.args.train_data_dir, self.args.train_data_name)
            train_dataset = load_from_disk(dataset_path)
            train_dataset = train_dataset['train']

        # negative in-batch ready
        corpus = list(set([example["context"] for example in train_dataset]))
        corpus = np.array(corpus)
        p_with_neg = []
        for c in train_dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.args.num_neg)
                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        q_seqs = self.tokenizer(
            train_dataset["question"],
            padding="max_length",
            truncation=True,
            # max_length=512,
            return_tensors="pt"
        )
        p_seqs = self.tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            # max_length=512,
            return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, self.args.num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, self.args.num_neg + 1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, self.args.num_neg + 1, max_len)

        # question and passage concat for training
        final_train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        return final_train_dataset

class DprRetrieval(Retrieval):
    '''A class for retrieving passages based on DPR method

    Args:
        args (RetrievalArguments): Arguments for retrieval process
    '''

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.p_embedding = None
        self.q_embedding = None
        self.q_encoder = None

    def proc_embedding(self):
        if path.isfile('question_embedding.bin'):
            with open('question_embedding.bin', "rb") as file:
                self.q_embedding = pickle.load(file)

        if path.isfile('passage_embedding.bin'):
            with open('passage_embedding.bin', "rb") as file:
                self.p_embedding = pickle.load(file)

        if self.q_embedding is None and self.p_embedding is None:
            print('--- Question Embedding and Passage Embedding Start ---')
            q_encoder, p_encoder = self._load_encoder()
            train_dataset = self._load_dataset()

            args = TrainingArguments(
                output_dir="dense_retrieval",
                evaluation_strategy="epoch",
                learning_rate=self.args.lr,
                per_device_train_batch_size=self.args.train_batch_size,
                per_device_eval_batch_size=self.args.eval_batch_size,
                num_train_epochs=self.args.epochs,
                weight_decay=0.01,
                gradient_accumulation_steps=1,  # 메모리 효율
            )

            p_encoder, q_encoder = self._train(args, train_dataset, p_encoder, q_encoder, self.args.num_neg)

            with torch.no_grad():
                p_encoder.eval()

                # passage embedding
                p_embedding = []
                for passage in tqdm.tqdm(self.contexts):  # passages from wiki
                    passage = self.tokenizer(
                        passage,
                        padding="max_length",
                        truncation=True,
                        # max_length=512,
                        return_tensors="pt"
                    ).to("cuda")
                    p_emb = p_encoder(**passage).to("cpu").detach().numpy()
                    p_embedding.append(p_emb)
                p_embedding = np.array(p_embedding).squeeze()

            # passage embedding save
            self.p_embedding = p_embedding
            with open('passage_embedding.bin', "wb") as file:
                pickle.dump(self.p_embedding, file)

            self.q_encoder = q_encoder

    def _train(self, args, train_dataset, p_encoder, q_encoder, num_neg=2):
        batch_size = args.per_device_train_batch_size
        p_encoder.to('cuda')
        q_encoder.to('cuda')

        # Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
            {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        global_step = 0

        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:

            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    p_encoder.train()
                    q_encoder.train()

                    targets = torch.zeros(batch_size).long()  # positive example
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (num_neg + 1), -1).to(args.device)
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    del batch
                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = p_encoder(**p_inputs)
                    # (batch_size, emb_dim)
                    q_outputs = q_encoder(**q_inputs)

                    p_outputs = p_outputs.view(batch_size, -1, num_neg + 1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    q_encoder.zero_grad()
                    p_encoder.zero_grad()

                    global_step += 1
                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

        # 인코더 저장
        q_encoder.save_pretrained('q_encoder/')
        p_encoder.save_pretrained('p_encoder/')
        return p_encoder, q_encoder

    def get_relevant_doc_bulk(self, queries, topk=1):
        if self.q_embedding is None: # embedding file doesn't exist
            print('--- Question Embedding Start ---')
            self.q_encoder.eval()
            self.q_encoder.cuda()

            with torch.no_grad():
                q_seqs_val = self.tokenizer(
                    queries,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to("cuda")
                q_embedding = self.q_encoder(**q_seqs_val)
                q_embedding.squeeze_()
                self.q_embedding = q_embedding.cpu().detach().numpy()

            # question embedding save
            with open('question_embedding.bin', "wb") as file:
                pickle.dump(self.q_embedding, file)

        # p_embedding: numpy, q_embedding: numpy
        result = torch.matmul(self.q_embedding, self.p_embedding.T)
        print('--- Sim Result ---')
        print(result.shape)

        # doc_indices = np.argsort(result, axis=1)[:, -topk:][:, ::-1]
        tmp_indices = torch.argsort(result, dim=1, descending=True).squeeze()

        doc_indices = []
        for i in range(tmp_indices.size()[0]):
            tmp = []
            for j in range(topk):
                tmp += [tmp_indices[i][j]]
            doc_indices.append(tmp)

        doc_scores = []
        for i in range(len(doc_indices)):
            doc_scores.append(result[i][[doc_indices[i]]])
        return doc_scores, doc_indices

    def retrieve(self, query_or_dataset, topk=1):
        total = []
        alpha = 2 # 중복 방지
        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            query_or_dataset["question"], topk=max(40 + topk, alpha * topk)
        )
        for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieval: ")):

            doc_scores_topk = [doc_scores[idx][0]]
            doc_indices_topk = [doc_indices[idx][0]]

            pointer = 1

            while len(doc_indices_topk) != topk:
                is_non_duplicate = True
                new_text_idx = doc_indices[idx][pointer]
                new_text = self.contexts[new_text_idx]

                for d_id in doc_indices_topk:
                    if fuzz.ratio(self.contexts[d_id], new_text) > 65:
                        is_non_duplicate = False
                        break

                if is_non_duplicate:
                    doc_scores_topk.append(doc_scores[idx][pointer])
                    doc_indices_topk.append(new_text_idx)

                pointer += 1
                if pointer == max(40 + topk, alpha * topk):
                    break

            assert len(doc_indices_topk) == topk, "중복 없는 topk 추출을 위해 alpha 값을 증가시켜 주세요."

            for doc_id in range(topk):
                doc_idx = doc_indices_topk[doc_id]
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": self.context_ids[doc_idx],  # retrieved id
                    "context": self.contexts[doc_idx],  # retrieved passage
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original passage
                    tmp["answers"] = example["answers"]  # original answer
                total.append(tmp)

        df = pd.DataFrame(total)
        print(df[:10])

        if self.args.predict is True:
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                    "context_id": Value(dtype="int32", id=None),
                }
            )
        else:
            f = Features(
                {
                    "answers": Sequence(
                        feature={
                            "text": Value(dtype="string", id=None),
                            "answer_start": Value(dtype="int32", id=None)},
                        length=-1,
                        id=None,
                    ),
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                    "original_context": Value(dtype="string", id=None),
                    "context_id": Value(dtype="int32", id=None),
                }
            )

        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets