# HHGN
Paper: Hierarchical Heterogeneous Graph Network based Multimodal Emotion Recognition in Conversation

## Requirements
```
python==3.11.3
torch==2.0.1
torch-geometric==2.3.1
sentence-transformers==2.2.2
comet-ml==3.33.4
```
## data

If you want to download the `MELD` database, you can access the link: [https://affective-meld.github.io](https://affective-meld.github.io/)

If you want to download the `IEMOCAP` database, you can access the link:https://sail.usc.edu/iemocap/release_form.php

### Installation

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Comet.ml](https://www.comet.ml/docs/python-sdk/advanced/)
### Preparing datasets
```
 python preprocess.py --dataset="iemocap"
```
### Training
```
python train.py --dataset="iemocap" --modalities="atv" --from_begin --epochs=32 --learning_rate=0.00008 --optimizer="adam" --drop_rate=0.3 --batch_size=32 --rnn="lstm" 
```
### Evaluate
```
python eval.py --dataset="iemocap" --modalities="atv"
```
