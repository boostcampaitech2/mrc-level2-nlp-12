# Team AI-it Repo for MRC tasks

## Contributor

- 김재현/T2050

- 박진영/T2096
- 안성민/T2127
- 양재욱/T2130
- 이연걸/T2163
- 조범준/T2211
- 진혜원/T2217

## Experiment Log

- [Experiment Log](https://jet-rook-fae.notion.site/MRC-d897b0cab1854936ba1d0e027e39c71c)

## Hardware

- GPU : V100

- Language : Python
- Develop tools : Jupyter Notebook, VSCode, Pycharm, Google Colab

## File List

```
code
  ├── README.md
  ├── install
  │   └── install_requirements.sh
  ├── reader
  │   ├── __init__.py
  │   └── conv
  │       ├── cnn.py
  │       └── custom_model.py
  ├── retrieval
  │   ├── __init__.py
  │	  ├── sparse
  │	  │	  ├── __init__.py
  │	  │   ├── tfidf.py
  │	  │   ├── es_dfr.py
  │	  │   └── es_bm25.py
  │   └── dense
  │	  	  ├── __init__.py
  │	  	  ├── st.py
  │	  	  └── DPR.py
  ├── utils
  │    ├── __init__.py
  │    ├── train_qa.py
  │    ├── utils_qa.py
  │    └── preprocess.py
  ├── arguments.py
  ├── inference.py
  └── train.py
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

## Getting Started

### Dependencies

- datasets==1.5.0
- transformers==4.5.0
- tqdm==4.41.1
- pandas==1.1.4
- scikit-learn=0.24.1
- konlpy==0.5.2

아래 스크립트를 실행하여 requirements 설치를 진행할 수 있습니다.

```
bash ./code/install/install_requirements.sh
```

### Training

```
python train.py --[args] [value]
python train.py --model klue-roberta-large
```

### Inference

```
python inference.py --[args] [value]
python inference.py --top_k_retrieval 30
```

## Apply

### Dataset

- Train: Given MRC dataset

### Model

- klue/roberta-large

- klue/roberta-base

### Optimizer & Loss

- Optimizer : AdamW
- Loss : Cross Entropy

### Wandb for Tracking



### Model Architecture

![모델 구조](https://user-images.githubusercontent.com/48538655/140604596-952cb7dd-bb30-4604-8bed-1b1981447293.PNG)

