# Tagger

A Joint Chinese segmentation and POS tagger based on bidirectional GRU-CRF

## News

The code is updated to TensorFlow 1.2.0 (2017.7.14)

Dyniamic bidirectional rnn is employed, now it requires drastically less memory both for training and tagging (2017.7.14)

Now the tagger supports bucket model to very efficiently tag very large files. 

## Requirements

Python 2.7

TensorFlow 1.2.0

Pygame (Convert Chinese characters into pictures)

## Reference

Yan Shao, Christian Hardmeier, Jörg Tiedemann and Joakim Nivre. "Character-based Joint Segmentation and POS Tagging for Chinese using Bidirectional RNN-CRF" arXiv preprint arXiv: 1704.01314 (2017).

https://arxiv.org/pdf/1704.01314.pdf

To reproduce the results reported in the paper:

## Single

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1 -emb Embeddings/glove.txt

python tagger.py test -p ud1 -e test.txt -m gru_full_ud1 -emb Embeddings/glove.txt

## Ensemble

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1_1 -emb Embeddings/glove.txt

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1_2 -emb Embeddings/glove.txt

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1_3 -emb Embeddings/glove.txt

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1_4 -emb Embeddings/glove.txt

python tagger.py test -ens -p ud1 -e test.txt -m model_ud1 -emb Embeddings/glove.txt
