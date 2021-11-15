# Team AI-it Repo for ODQA(KLUE-MRC) task

## Contributors

- 김재현/T2050
- 박진영/T2096
- 안성민/T2127
- 양재욱/T2130
- 이연걸/T2163
- 조범준/T2211
- 진혜원/T2217

## Experiment Log

- [Experiment Log](https://jet-rook-fae.notion.site/MRC-d897b0cab1854936ba1d0e027e39c71c)

## Wrap-up Report

- [Wrap-up Report](https://jet-rook-fae.notion.site/Wrap-up-Report-ODQA-Team-AI-it-2e1c2f63d8384a278db593abde80697b)

## Project Tree

```
code
  ├── install
  │   └── install_requirements.sh
  ├── reader
  │   ├── __init__.py
  │   └── conv
  │       ├── cnn.py
  │       └── custom_model.py
  ├── retrieval
  │   ├── __init__.py
  │   ├── hybrid.py
  │   ├── sparse
  │   │   ├── __init__.py
  │   │   ├── bm25.py
  │   │   ├── es_bm25.py
  │   │   ├── es_dfr.py
  │   │   └── tfidf.py
  │   └── dense
  │       ├── __init__.py
  │       ├── st.py
  │       └── dpr
  │           ├── __init__.py
  │           ├── dpr_dataset.py
  │           ├── dpr_model.py
  │           ├── dpr_retrieve.py
  │           └── dpr_train_utils.py
  ├── utils
  │    ├── __init__.py
  │    ├── pypreprocess.py
  │    ├── train_qa.py
  │    ├── utils_qa.py
  │    └── wiki_split.py
  ├── arguments.py
  ├── dpr_trainer.py
  ├── ensemble.py
  ├── inference.py
  ├── readme.md
  ├── train.py
  └── train_mlm.py
```

