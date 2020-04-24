# SetRank

> Tenforflow implementation of [SIGIR 2020] SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval.

[![Python 3.6](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## Usage

### Quick Start
> If you have downloaded **Istella Dataset** please comment `sh download_data.sh`.
```bash
sh run.sh
```

### Data Preparation
```bash
cd data
sh download_data.sh
python norm_split_dataset.py
```

### Model Training/Testing
```bash
sh ./scripts/train_lambdamart_istella.sh
sh ./scripts/prepare_data_lambda_istella.sh
sh ./scripts/train_transformer_istella.sh
```

## Citation

If you use SetRank in your research, please use the following BibTex entry.

```
@misc{pang2019setrank,
    title={SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval},
    author={Liang Pang and Jun Xu and Qingyao Ai and Yanyan Lan and Xueqi Cheng and Jirong Wen},
    year={2019},
    eprint={1912.05891},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
```

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

Copyright (c) 2019-present, Liang Pang (pl8787)
