# Tagger

A Joint Chinese segmentation and POS tagger based on bidirectional GRU-CRF

## News

Add instructions on how to tag raw sentences with trained models. (2017.12.9)

Intergrated the feedforward neural network model introduced in Zheng et al. (2013) (2017.11.25)

Updated HiddenLayer for efficiency. TimeDistributed is not applied for output inference anymore. (2017.11.25)

The code is updated to TensorFlow 1.2.0 (2017.7.14)

Dyniamic bidirectional rnn is employed, now it requires drastically less memory both for training and tagging (2017.7.14)

Now the tagger supports bucket model to very efficiently tag very large files. 

## Requirements

Python 2.7

TensorFlow 1.2.0 or newer

Pygame (Convert Chinese characters into pictures)

## Reference

Yan Shao, Christian Hardmeier, Jörg Tiedemann and Joakim Nivre. "Character-based Joint Segmentation and POS Tagging for Chinese using Bidirectional RNN-CRF", Proceedings of the The 8th International Joint Conference on Natural Language Processing, pages 173–183, Taipei, Taiwan, 2017

http://aclweb.org/anthology/I/I17/I17-1018.pdf

# To reproduce the results reported in the paper:

## Single

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1 -emb Embeddings/glove.txt

python tagger.py test -p ud1 -e test.txt -m model_ud1 -emb Embeddings/glove.txt

## Ensemble

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1_1 -emb Embeddings/glove.txt

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1_2 -emb Embeddings/glove.txt

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1_3 -emb Embeddings/glove.txt

python tagger.py train -p ud1 -t train.txt -d dev.txt -wv -cp -rd -gru -m model_ud1_4 -emb Embeddings/glove.txt

python tagger.py test -ens -p ud1 -e test.txt -m model_ud1 -emb Embeddings/glove.txt

# To tag raw sentences:

## Use simple model:

(simple)

python tagger.py tag -p ud1 -r raw.txt -m model_ud1 -emb Embeddings/glove.txt  -opth tagged_file.txt 

(ensemble)

python tagger.py tag -ens -p ud1 -r raw.txt -m model_ud1 -emb Embeddings/glove.txt  -opth tagged_file.txt 

## Use bucket model (recommended for tagging very large corpora):

(simple)

python tagger.py tag -p ud1 -r raw.txt -m model_ud1 -emb Embeddings/glove.txt  -opth tagged_file.txt -tl

(ensemble)

python tagger.py tag -ens -p ud1 -r raw.txt -m model_ud1 -emb Embeddings/glove.txt  -opth tagged_file.txt -tl


