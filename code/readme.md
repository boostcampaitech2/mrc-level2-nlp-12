# Team AI-it code for ODQA(KLUE-MRC) task

## Dependencies
```bash
# 필요한 파이썬 패키지 설치. 
bash ./install/install_requirements.sh
```

## Data Tree
```
data
  ├── train_dataset
  │    ├── train
  │    │   ├── dataset.arrow
  │    │   ├── dataset_info.json
  │    │   ├── indices.arrow
  │    │   └── state.json
  │    ├── validation
  │    │   ├── dataset.arrow
  │    │   ├── dataset_info.json
  │    │   ├── indices.arrow
  │    │   └── state.json
  │    └── dataset_dict.json
  ├── test_dataset
  │    ├── validation
  │    │   ├── dataset.arrow
  │    │   ├── dataset_info.json
  │    │   ├── indices.arrow
  │    │   └── state.json
  │    └── dataset_dict.json
  └── wikipedia_documents.json
```

# train, eval, inference

## Train & Eval

만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요. 

roberta 모델을 사용할 경우 tokenizer 사용시 아래 함수의 옵션을 수정해야합니다.
베이스라인은 klue/bert-base로 진행되니 이 부분의 주석을 해제하여 사용해주세요 ! 
tokenizer는 train, validation (train.py), test(inference.py) 전처리를 위해 호출되어 사용됩니다.
(tokenizer의 return_token_type_ids=False로 설정해주어야 함)

```
# train.py
def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            #return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )
```

```
# 학습 예시 (train_dataset 사용)
python train.py --output_dir ./models/train_dataset --do_train
```

```
# 학습 & 평가 예시 (train_dataset 사용)
python train.py --output_dir ./models/train_dataset --do_train --do_eval
```

## Inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

* 학습한 모델의  test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다.

* 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(--do_eval)를 진행하면 됩니다.

```
# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict
```

## Things to know

1. `train.py` 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. **만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요!** 안그러면 존재하는 파일이 load 됩니다.
2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다. 
3. ./outputs/ 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.
