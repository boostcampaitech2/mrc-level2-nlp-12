from .dpr_train_utils import DPRTrainer
from .dpr_dataset import DPRDataset
from .dpr_model import Encoder
from typing import List, Tuple, NoReturn, Any, Optional, Union
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer
import os


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DensePassageRetrieval(DPRTrainer):
    """Passage retrieve for the question through the trained p,q encoder
    """

    def __init__(self, args) -> None:
        dpr_tokenizer = AutoTokenizer.from_pretrained(
            args.model_checkpoint, use_fast=True,
        )
        dpr_datasets = DPRDataset(
            "/opt/ml/data/wikipedia_documents.json", "/opt/ml/data/test_dataset"
        )
        wiki_dataset = dpr_datasets.load_wiki_data()
        if "q_encoder.pt" in os.listdir(self.args.q_encoder_path):
            self.q_encoder = Encoder(args.model_checkpoint)
            self.q_encoder.load_state_dict(
                torch.load(args.q_encoder_path + "/q_encoder.pt")
            )
        else:
            raise f"q_encoder.pt 파일이 {self.args.q_encoder_path} 폴더 안에 있지 않습니다."

        super().__init__(args, dpr_tokenizer, wiki_dataset, None, None)

    def retrieve(self, query_or_dataset, topk=1):
        """Method to find passage as much as specified as topk for query_or_dataset

        Args:
            query_or_dataset (str or dataset): question
            topk (int, optional): the number of retrieve passages. Defaults to 1.

        Returns:
            df (pandas.DataFrame): Dataframe composed of retrieval data
        """
        total = []
        alpha = 2  # 중복 방지
        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            query_or_dataset["question"], topk=topk
        )
        doc_indices = np.array(doc_indices)
        doc_indices = doc_indices.tolist()

        for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieval: ")):

            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": doc_indices[idx],
                "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        df = pd.DataFrame(total)

        return df

    def get_relevant_doc_bulk(self, queries, topk=1):

        print("--- Question Embedding Start ---")

        with torch.no_grad():
            self.q_encoder.eval()
            self.q_encoder.cuda()
            q_seqs_val = self.tokenizer(
                queries,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=(
                    False if "roberta" in self.args.model_checkpoint else True
                ),
            ).to("cuda")
            q_embedding = self.q_encoder(**q_seqs_val)
            q_embedding.squeeze_()
            self.q_embedding = q_embedding.cpu().detach().numpy()

        print("--- Sim Result ---")
        result = torch.matmul(
            torch.tensor(self.q_embedding), torch.tensor(self.p_embedding.T)
        )
        print(result.shape)

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
