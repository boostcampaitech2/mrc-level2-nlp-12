import os.path as path
import os
from .dpr_model import DPRetrieval
import pickle
import torch
import tqdm
import numpy as np
import wandb
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from contextlib import contextmanager
import time

"""DPREmbedding Agumentation list
train
num_neg
use_wandb
"""


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DPRTrainer(DPRetrieval):
    def __init__(self, args, tokenizer, wiki_data, train_data, eval_data):
        super().__init__(args, tokenizer, wiki_data, train_data, eval_data)
        self.p_embedding = None
        self.q_embedding = None
        self.best_acc = 0
        self.p_encoder = None
        self.q_encoder = None

    def load_passage_embedding(self):
        """ 저장되어있는 passage_embedding을 불러옴
        """
        if path.isfile(path.join(self.args.p_encoder_path, "passage_embedding.bin")):
            with open(
                path.join(self.args.p_encoder_path, "passage_embedding.bin"), "rb"
            ) as file:
                self.p_embedding = pickle.load(file)

    def set_embedding(self, queries, q_encoder=None, p_encoder=None):
        """주어진 q_encoder, p_encoder를 바탕으로 question과 passage를 embedding

        Args:
            queries (list): questions
            q_encoder (nn.Module, optional): question encoder. Defaults to None.
            p_encoder (nn.Module, optional): passage encoder. Defaults to None.
        """
        print("--- Question Embedding and Passage Embedding Start ---")
        if q_encoder == None and p_encoder == None:
            q_encoder, p_encoder = self._load_encoder()
        print("--- Passage Embedding Start ---")
        with torch.no_grad():
            p_encoder.eval()
            p_encoder.cuda()
            # passage embedding
            p_embedding = []
            for passage in tqdm.tqdm(self.contexts):  # passages from wiki
                passage = self.tokenizer(
                    passage,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                p_emb = p_encoder(**passage).to("cpu").detach().numpy()
                p_embedding.append(p_emb)
            p_embedding = np.array(p_embedding).squeeze()

        # passage embedding save
        self.p_embedding = p_embedding
        self.p_encoder = p_encoder

        if not self.args.best_save:
            if not path.isdir(self.args.p_encoder_path):
                os.mkdir(self.args.p_encoder_path)

            with open(
                path.join(self.args.p_encoder_path, "passage_embedding.bin"), "wb"
            ) as file:
                pickle.dump(self.p_embedding, file)

        print("--- Question Embedding Start ---")
        with torch.no_grad():
            q_encoder.eval()
            q_encoder.cuda()
            q_seqs_val = self.tokenizer(
                queries,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            q_embedding = q_encoder(**q_seqs_val)
            q_embedding.squeeze_()

        self.q_embedding = q_embedding.cpu().detach().numpy()
        self.q_encoder = q_encoder

        # question embedding save
        if not path.isdir(self.args.q_encoder_path):
            os.mkdir(self.args.q_encoder_path)
        with open("question_embedding.bin", "wb") as file:
            pickle.dump(self.q_embedding, file)

    def train(self, args, **kwargs):
        if self.args.use_wandb:
            self.init_wandb(**kwargs)
            self.run.config.update(self.args)
            self.log_table = []
        if self.q_encoder == None or self.p_encoder == None:
            self.q_encoder, self.p_encoder = self._load_encoder()

        p_encoder, q_encoder = self._train(
            args,
            self._load_dataset(),
            self.p_encoder,
            self.q_encoder,
            self.args.num_neg,
        )

    def _train(self, args, train_dataset, p_encoder, q_encoder, num_neg=2):
        metric_key_prefix = "train"
        batch_size = args.per_device_train_batch_size
        p_encoder.to("cuda")
        q_encoder.to("cuda")

        # Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm.trange(int(args.num_train_epochs), desc="Epoch")
        losses = 0
        for _ in train_iterator:
            with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    if global_step % args.eval_steps == 0 and global_step != 0:
                        self.eval(p_encoder, q_encoder, global_step)
                    p_encoder.train()
                    q_encoder.train()

                    targets = torch.zeros(batch_size).long()  # positive example
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0]
                        .view(batch_size * (num_neg + 1), -1)
                        .to(args.device),
                        "attention_mask": batch[1]
                        .view(batch_size * (num_neg + 1), -1)
                        .to(args.device),
                        "token_type_ids": batch[2]
                        .view(batch_size * (num_neg + 1), -1)
                        .to(args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device),
                    }

                    del batch
                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = p_encoder(**p_inputs)
                    # (batch_size, emb_dim)
                    q_outputs = q_encoder(**q_inputs)

                    p_outputs = torch.transpose(
                        p_outputs.view(batch_size, num_neg + 1, -1), 1, 2
                    )
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)

                    tepoch.set_postfix(loss=f"{str(loss.item())}")
                    losses += loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    q_encoder.zero_grad()
                    p_encoder.zero_grad()
                    if global_step % args.logging_steps == 0 and global_step != 0:
                        loss_mean = losses / global_step
                        losses = 0
                        metric = {"loss": loss_mean}
                        for key in list(metric.keys()):
                            if not key.startswith(f"{metric_key_prefix}_"):
                                metric[f"{metric_key_prefix}_{key}"] = metric.pop(key)
                        if self.args.use_wandb:
                            self.run.log(metric)
                    global_step += 1
                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

        self.eval(p_encoder, q_encoder, global_step)
        return p_encoder, q_encoder

    def eval(self, p_encoder, q_encoder, global_step):

        print("-" * 15 + " evaluation start " + "-" * 15)
        metric_key_prefix = "eval"
        datasets = self.eval_dataset

        top_k = self.args.eval_topk

        queries = datasets["question"]
        ground_truth = datasets["document_id"]
        self.set_embedding(queries, q_encoder, p_encoder)
        doc_scores, doc_indices = self.get_relevant_doc_bulk(queries, top_k)

        assert len(ground_truth) == len(doc_indices), "GT와 임베딩 된 doc의 수가 다름"

        cnt = 0
        for i, ans in enumerate(ground_truth):
            if ans in doc_indices[i]:
                cnt += 1
        n = random.randint(0, len(queries))
        if self.args.use_wandb:
            self.log_table.append(
                [
                    global_step,
                    ground_truth[n],
                    queries[n],
                    self.contexts[ground_truth[n]],
                    self.contexts[doc_indices[n][0]],
                ]
            )
            self.text_table = wandb.Table(
                columns=["global_step", "doc_id", "question", "ground_truth", "top1"],
                data=self.log_table,
            )
        acc = cnt / len(doc_indices) * 100
        metric = {"accuarcy": acc}

        for key in list(metric.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metric[f"{metric_key_prefix}_{key}"] = metric.pop(key)
        print(f"top_k : {top_k}, accuracy : {acc}")

        if self.args.use_wandb:
            self.run.log(metric)
            self.run.log({"valid_samples": self.text_table})

        # 인코더 저장
        if not path.isdir(self.args.p_encoder_path):
            os.makedirs(self.args.p_encoder_path)
        if not path.isdir(self.args.q_encoder_path):
            os.makedirs(self.args.q_encoder_path)

        if self.args.best_save:
            if self.best_acc < acc:
                self.best_acc = acc
                torch.save(
                    p_encoder.state_dict(),
                    os.path.join(self.args.p_encoder_path, f"p_encoder.pt"),
                )
                torch.save(
                    q_encoder.state_dict(),
                    os.path.join(self.args.q_encoder_path, f"q_encoder.pt"),
                )
                print("--- save Encoder ---")
                if not path.isdir(self.args.p_encoder_path):
                    os.mkdir(self.args.p_encoder_path)
                with open(
                    path.join(self.args.p_encoder_path, "passage_embedding.bin"), "wb"
                ) as file:
                    pickle.dump(self.p_embedding, file)
                print("--- save p_embedding ---")
        else:
            torch.save(
                p_encoder.state_dict(),
                os.path.join(self.args.p_encoder_path, f"p_encoder.pt"),
            )
            torch.save(
                q_encoder.state_dict(),
                os.path.join(self.args.q_encoder_path, f"q_encoder.pt"),
            )

        return acc

    def get_relevant_doc_bulk(self, queries, topk=1):
        if self.q_embedding is None:  # embedding file doesn't exist
            print("--- Question Embedding Start ---")
            self.q_encoder.eval()
            self.q_encoder.cuda()

            with torch.no_grad():
                q_seqs_val = self.tokenizer(
                    queries,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                q_embedding = self.q_encoder(**q_seqs_val)
                q_embedding.squeeze_()
                self.q_embedding = q_embedding.cpu().detach().numpy()

            # question embedding save
            with open("question_embedding.bin", "wb") as file:
                pickle.dump(self.q_embedding, file)

        # p_embedding: numpy, q_embedding: numpy
        print("--- Sim Result ---")
        result = torch.matmul(
            torch.tensor(self.q_embedding), torch.tensor(self.p_embedding.T)
        )
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

    def init_wandb(self, entity, project, runname):
        # self.run = wandb.init(project="T2050-retrieval-dev", entity="bc-ai-it-mrc", name="metric_text")
        self.run = wandb.init(project=project, entity=entity, name=runname)

