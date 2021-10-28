from .dpr_train_utils import DPRTrainer
from typing import List, Tuple, NoReturn, Any, Optional, Union
from datasets import Dataset
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
from tqdm.auto import tqdm
from datasets import Sequence, Value, Features, DatasetDict, Dataset
from fuzzywuzzy import fuzz
import torch
import pickle


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DensePassageRetrieval(DPRTrainer):
    def __init__(
        self, args, tokenizer, wiki_dataset, train_dataset, eval_dataset, q_encoder
    ) -> None:
        super().__init__(args, tokenizer, wiki_dataset, train_dataset, eval_dataset)
        self.q_encoder = q_encoder

    def retrieve(self, query_or_dataset, topk=1):
        total = []
        alpha = 2  # 중복 방지
        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            query_or_dataset["question"], topk=topk
        )
        doc_indices = np.array(doc_indices)
        doc_indices = doc_indices.tolist()
        # doc_scores, doc_indices = self.get_relevant_doc_bulk(
        #     query_or_dataset["question"], topk=max(40 + topk, alpha * topk)
        # )
        for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieval: ")):

            #     doc_scores_topk = [doc_scores[idx][0]]
            #     doc_indices_topk = [doc_indices[idx][0]]

            #     pointer = 1

            #     while len(doc_indices_topk) != topk:
            #         is_non_duplicate = True
            #         new_text_idx = doc_indices[idx][pointer]
            #         new_text = self.contexts[new_text_idx]

            #         for d_id in doc_indices_topk:
            #             if fuzz.ratio(self.contexts[d_id], new_text) > 65:
            #                 is_non_duplicate = False
            #                 break

            #         if is_non_duplicate:
            #             doc_scores_topk.append(doc_scores[idx][pointer])
            #             doc_indices_topk.append(new_text_idx)

            #         pointer += 1
            #         if pointer == max(40 + topk, alpha * topk):
            #             break

            #     assert len(doc_indices_topk) == topk, "중복 없는 topk 추출을 위해 alpha 값을 증가시켜 주세요."

            #     for doc_id in range(topk):
            #         doc_idx = doc_indices_topk[doc_id]
            #         tmp = {
            #             "question": example["question"],
            #             "id": example["id"],
            #             "context_id": self.context_ids[doc_idx],  # retrieved id
            #             "context": self.contexts[doc_idx],  # retrieved passage
            #         }
            #         if "context" in example.keys() and "answers" in example.keys():
            #             tmp["original_context"] = example["context"]  # original passage
            #             tmp["answers"] = example["answers"]  # original answer
            #         total.append(tmp)

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
        print(df[:10])
        df.to_csv("./predict_df.csv")
        if self.args.predict is True:
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                    # "context_id": Value(dtype="int32", id=None),
                }
            )
        else:
            f = Features(
                {
                    "answers": Sequence(
                        feature={
                            "text": Value(dtype="string", id=None),
                            "answer_start": Value(dtype="int32", id=None),
                        },
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

    def get_relevant_doc_bulk(self, queries, topk=1):
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
        # # question embedding save
        # with open("question_embedding.bin", "wb") as file:
        #     pickle.dump(self.q_embedding, file)
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
