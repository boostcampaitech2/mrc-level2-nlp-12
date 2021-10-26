from transformers import (
    TrainingArguments,
    BertPreTrainedModel,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel,
    AutoConfig,
)
import torch.nn as nn
from torch.utils.data import TensorDataset
import os.path as path
import numpy as np
from dpr_dataset import BM25Data
from contextlib import contextmanager
import time
import tqdm


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class Encoder(nn.Module):
    def __init__(self, model_checkpoint):
        super().__init__()
        self.model_checkpoint = model_checkpoint
        config = AutoConfig.from_pretrained(model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint, config=config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        return pooled_output


class BertEncoder(BertPreTrainedModel):
    """A class for encoding questions and passages
    """

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        # embedded vec
        pooled_output = outputs[1]
        return pooled_output

    # DPRetrieval arguments list
    """
    q_encoder_path
    p_encoder_path
    model_checkpoint
    neg_strategy
    num_neg
    """


class DPRetrieval:
    """A common class for all retrieval methods
    """

    def __init__(self, args, tokenizer, wiki_data, train_data, eval_data):
        # set_seed(42)
        self.args = args
        self.tokenizer = tokenizer
        self.wiki = wiki_data
        self.train_dataset = train_data
        self.eval_dataset = eval_data
        self.contexts = [v["text"] for v in self.wiki.values()]
        self.context_ids = [v["document_id"] for v in self.wiki.values()]
        self.bm25 = BM25Data(tokenizer.tokenize, wiki_data)

    def _load_encoder(self):
        # q_encoder, p_encoder => 인코더 bin 파일 존재 확인
        if path.isfile(
            path.join(self.args.q_encoder_path, "pytorch_model.bin")
        ) and path.isfile(path.join(self.args.p_encoder_path, "pytorch_model.bin")):
            print("--- Load Encoders from Local ---")
            q_encoder = Encoder(self.args.q_encoder_path)
            p_encoder = Encoder(self.args.p_encoder_path)
        else:
            print("--- Load Encoders from Server ---")
            q_encoder = Encoder(self.args.model_checkpoint)
            p_encoder = Encoder(self.args.model_checkpoint)

        return q_encoder, p_encoder

    def _load_dataset(self):
        # negative in-batch ready
        # corpus = list(set([example["context"] for example in self.train_dataset]))
        # corpus = np.array(corpus)
        corpus = np.array(self.train_dataset["context"])
        cor_titles = np.array(self.train_dataset["title"])
        p_with_neg = []
        titles = []
        if self.args.neg_strategy == "random":
            for t, c in zip(self.train_dataset["title"], self.train_dataset["context"]):
                titles.append(t)
                while True:
                    neg_idxs = np.random.randint(len(corpus), size=self.args.num_neg)
                    if not c in corpus[neg_idxs]:
                        p_neg = corpus[neg_idxs]
                        p_neg_title = cor_titles[neg_idxs]
                        p_with_neg.append(c)
                        p_with_neg.extend(p_neg)
                        titles.extend(p_neg_title)
                        break

        elif self.args.neg_strategy == "BM_Gold":
            with timer("BM Gold neg sampling"):
                self.bm25.get_sparse_embedding()
                for t, c, q in tqdm.tqdm(
                    zip(
                        self.train_dataset["title"],
                        self.train_dataset["context"],
                        self.train_dataset["question"],
                    )
                ):
                    titles.append(t)
                    while True:
                        neg_idxs = np.random.randint(
                            len(corpus), size=self.args.num_neg - 1
                        )
                        if not c in corpus[neg_idxs]:
                            p_neg = corpus[neg_idxs]
                            p_with_neg.append(c)
                            p_neg_title = cor_titles[neg_idxs]
                            # BM25 top1 context 넣기

                            bm_score, bm_context, bm_indice = self.bm25.retrieve(q, 50)
                            i = 0
                            while True:
                                assert i < 50, f"bm25의 top50 까지 {c}와 같아 에러 발생"
                                if bm_context[i] != c:
                                    p_with_neg.append(bm_context[i])
                                    titles.append(self.wiki[str(bm_indice[i])]["title"])
                                    break
                                i += 1
                            p_with_neg.extend(p_neg)
                            titles.extend(p_neg_title)
                            break

        # t_seqs = self.tokenizer(
        #     self.train_dataset["title"],
        #     padding="max_length",
        #     truncation=True,
        #     max_length=512,
        #     return_tensors="pt",
        # )

        q_seqs = self.tokenizer(
            self.train_dataset["question"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        p_seqs = self.tokenizer(
            titles,
            p_with_neg,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(
            -1, self.args.num_neg + 1, max_len
        )
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, self.args.num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, self.args.num_neg + 1, max_len
        )

        # question and passage concat for training
        final_train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
            # t_seqs["input_ids"],
            # t_seqs["attention_mask"],
            # t_seqs["token_type_ids"],
        )
        return final_train_dataset

    def _load_eval_dataset(self):

        # negative in-batch ready
        # corpus = list(set([example["context"] for example in self.eval_dataset]))
        # corpus = np.array(corpus)
        corpus = np.array(self.eval_dataset["context"])
        cor_titles = np.array(self.eval_dataset["title"])
        titles = []
        p_with_neg = []
        if self.args.neg_strategy == "random":
            for t, c in zip(self.eval_dataset["title"], self.eval_dataset["context"]):
                titles.append(t)
                while True:
                    neg_idxs = np.random.randint(len(corpus), size=self.args.num_neg)
                    if not c in corpus[neg_idxs]:
                        p_neg = corpus[neg_idxs]
                        p_neg_title = cor_titles[neg_idxs]
                        p_with_neg.append(c)
                        p_with_neg.extend(p_neg)
                        titles.extend(p_neg_title)
                        break

        elif self.args.neg_strategy == "BM_Gold":
            self.bm25.get_sparse_embedding()
            for t, c, q in zip(
                self.eval_dataset["title"],
                self.eval_dataset["context"],
                self.eval_dataset["question"],
            ):
                titles.append(t)
                while True:
                    neg_idxs = np.random.randint(
                        len(corpus), size=self.args.num_neg - 1
                    )
                    if not c in corpus[neg_idxs]:
                        p_neg = corpus[neg_idxs]
                        p_with_neg.append(c)
                        p_neg_title = cor_titles[neg_idxs]
                        # BM25 top1 context 넣기
                        bm_score, bm_context, bm_indice = self.bm25.retrieve(q, 50)
                        i = 0
                        while True:
                            assert i < 50, f"bm25의 top50 까지 {c}와 같아 에러 발생"
                            if bm_context[i] != c:
                                p_with_neg.append(bm_context[i])
                                titles.append(self.wiki[str(bm_indice[i])]["title"])
                                break
                            i += 1
                        p_with_neg.append(bm_context)
                        p_with_neg.extend(p_neg)
                        titles.extend(p_neg_title)
                        break

        # t_seqs = self.tokenizer(
        #     self.eval_dataset["title"],
        #     padding="max_length",
        #     truncation=True,
        #     max_length=512,
        #     return_tensors="pt",
        # )

        q_seqs = self.tokenizer(
            self.eval_dataset["question"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        p_seqs = self.tokenizer(
            titles,
            p_with_neg,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(
            -1, self.args.num_neg + 1, max_len
        )
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, self.args.num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, self.args.num_neg + 1, max_len
        )

        # question and passage concat for training
        final_valid_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        return final_valid_dataset
