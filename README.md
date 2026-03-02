# Towards Graph Foundation Model: Node Feature Transfer Invariant Modeling on General Graphs

This is the repository for the paper "**Towards Graph Foundation Model: Node Feature Transfer Invariant Modeling on General Graphs**".

![Overview of TIG](TIG.png)

# 1. Environment Configurations
```plaintext
python==3.9.21
scikit-learn==1.6.1
scipy==1.13.1
networkx==3.2.1
numpy==2.0.1
torch==2.4.0
```
More details can be found in `env_description.txt`.

# 2. How to use TIG
To evaluate performance on cross-domain node classification, follow these steps:

1. Navigate to the project directory:

```bash
cd TIG-node-classification\gcn
```

2. Run the main script:

```bash
python main.py
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------

To evaluate performance on cross-domain few-shot scenarios, follow these steps:
1. Navigate to the project directory:

```bash
cd TIG-few-shot-node-classification\gcn
```

2. Run the main script:

```bash
python main.py
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------

To evaluate cross-domain graph classification, follow these steps:
1. Navigate to the project directory:

```bash
cd TIG-graph-classification\gcn
```

2. Run the main script:

```bash
python run-graph.py
```

# Cite


If you like our paper, please cite:

```bibtex
@inproceedings{TIG,
  title={Towards Graph Foundation Model: Node Feature Transfer Invariant Modeling on General Graphs},
  author={Jitao Zhao and Yi Wang and Yawen Li and Dongxiao He and Di Jin and Zhiyong Feng and Weixiong Zhang},
  booktitle={Proceedings of the Web Conference 2026},
  year={2026}
}
