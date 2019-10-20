## A Latent-Constrained Variational Neural Dialogue Model for Information-Rich Responses

This repository provides a Tensorflow implementation of Latent-Constrained Variational Neural Dialogue Model (LC-VNDM) proposed in [**A Latent-Constrained Variational Neural Dialogue Model for Information-Rich Responses**](https://github.com/zheng-yanan/latent-constrained-dialogue/edit/master/README.md), published as a long paper in CIKM2019.
Please refer to the paper for more details.

## Introduction

The variational neural models have achieved significant progress in dialogue generation. They are of encoder-decoder architecture, with stochastic latent variables learned at the utterance level. However, latent variables are usually approximated by factorized-form distributions, the value space of which is too large relative to latent features to be encoded, leading to the sparsity problem. As a result, little useful information is carried in latent representations, and generated responses tend to be non-committal and meaningless. To address it, we initially propose the Latent-Constrained Variational Neural Dialogue Model (LC-VNDM). It follows variational neural dialogue framework, with an utterance encoder, a context encoder and a response decoder hierarchically organized. Particularly, LC-VNDM uses a hierarchically-structured variational distribution form, which considers inter-dependencies between latent variables. Thus it defines a constrained latent value space, and prevents latent global features from being diluted. Therefore, latent representations sampled from it would carry richer global information to facilitate the decoding, generating meaningful responses. We conduct extensive experiments on three datasets using automatic evaluation and human evaluation. Experiments prove that LC-VNDM significantly outperforms the state-of-the-arts and can generate information-richer responses by learning a better-quality latent space.

<p align=center>
	<img "./LC-VNDM-arch.png" width=50% height=50%>
	<img "./infer-arch.png" width=50% height=50%>
</p>

## Usage:  
	python main.py --dataset <dataset_name> --word2vec True
will train the model with default settings and will save to path "./checkpoints/run_<dataset_name>_xxxxxxxx".  

	python main.py --dataset <dataset_name> --word2vec True --forward True --test_path run_<dataset_name>_xxxxxxxx  
will test the final-version model on valid & test sets, and will generate responses for test cases.

## Prerequisites
 - TensorFlow 1.4.0
 - Python 2.7    
 
## Data Preparation
- Please download the [GloVe pretrained word2vec embedding](https://nlp.stanford.edu/projects/glove/) into "./data/glove.twitter.27B.200d.txt"
- This repository provides both [DailyDialog](http://yanran.li/dailydialog) and [Switchboard](https://web.stanford.edu/~jurafsky/swb1_dialogact_annot.tar.gz) datasets. 
