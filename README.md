## A Latent-Constrained Variational Neural Dialogue Model for Information-Rich Responses

This is a Tensorflow implementation of Latent-Constrained Variational Neural Dialogue Model (LC-VNDM) proposed in [**A Latent-Constrained Variational Neural Dialogue Model for Information-Rich Responses**](https://github.com/zheng-yanan/latent-constrained-dialogue/edit/master/README.md), published as a long paper in CIKM2019.
Please refer to the paper for more details.

## Usage:  
	python main.py --dataset <dataset_name> --word2vec True
will train the model with default settings and will save to path "./checkpoints/run_<dataset_name>_xxxxxxxx".  

	python main.py --dataset <dataset_name> --word2vec True --forward True --test_path run_<dataset_name>_xxxxxxxx  
will test the final-version model on valid & test sets, and will generate responses for test cases.

## Prerequisites
 - TensorFlow 1.4.0
 - Python 2.7  
 - Please download the [GloVe pretrained word2vec embedding](https://nlp.stanford.edu/projects/glove/) into "./data/glove.twitter.27B.200d.txt"  
 
