# -*- coding: utf-8 -*-
import codecs
import sys

sys = reload(sys)
sys.setdefaultencoding('utf-8')

f1 = sys.argv[1]
f_gold = sys.argv[2]
f_prediction = sys.argv[3]

sent1 = []
sent2 = []

words_t = set()
w_pos_t = set()

total_w = 0

for line in codecs.open(f1, 'rb', encoding='utf-8'):
    line = line.strip()
    segs = line.split(' ')
    total_w += len(segs)
    for seg in segs:
        w_pos_t.add(seg)
        sp = seg.split('_')
        if len(sp) == 2:
            words_t.add(sp[0])

print 'Total numbers of words in the training set: %d.' % total_w
print 'Total numbers of unique words in the training set: %d (plain), %d (POS).' % (len(words_t), len(w_pos_t))
print ''

words_g = set()
w_pos_g = set()

total_word = 0
oov_w = 0
oov_pos = 0

o_words_g = set()
o_pos_g = set()

oov_w_dics = []
oov_pos_dics = []

for line in codecs.open(f_gold, 'rb', encoding='utf-8'):
    line = line.strip()
    segs = line.split(' ')
    total_word += len(segs)
    idx = 0
    dic_w = {}
    dic_pos = {}
    for seg in segs:
        w_pos_g.add(seg)
        if seg not in w_pos_t:
            o_pos_g.add(seg)
            oov_pos += 1
            dic_pos[idx] = seg
        sp = seg.split('_')
        if len(sp) == 2:
            words_g.add(sp[0])
            if sp[0] not in words_t:
                o_words_g.add(sp[0])
                oov_w += 1
                dic_w[idx] = sp
            idx += len(sp[0])
    oov_w_dics.append(dic_w)
    oov_pos_dics.append(dic_pos)


print 'Total numbers of words in golden test set: %d.' % total_word
print 'Total numbers of unique words in golden test set: %d (plain), %d (POS).' % (len(words_g), len(w_pos_g))
print ''
print 'Total numbers of OOV words in golden test set: %d (plain), %d (POS).' % (oov_w, oov_pos)
print 'Unique OOV words in golden test set: %d (plain), %d (POS).' % (len(o_words_g), len(o_pos_g))
print 'Percentages of OOV words in golden test set: %f (plain), %f (POS).' % (float(oov_w)/total_word, float(oov_pos)/total_word)
print ''

idx = 0
correct_w = 0
correct_pos = 0

incorrect_w = []
incorrect_pos = []

for line in codecs.open(f_prediction, 'rb', encoding='utf-8'):
    line = line.strip()
    segs = line.split(' ')
    idy = 0
    dic_w = {}
    dic_pos = {}
    for seg in segs:
        sp = seg.split('_')
        dic_pos[idy] = seg
        dic_w[idy] = sp[0]
        idy += len(sp[0])
    '''
    for k, v in oov_pos_dics[idx].items():
        if k in dic_pos:
            if v == dic_pos[k]:
                correct_pos += 1
            else:
                incorrect_pos.append(v)
    '''
    for k, v in oov_w_dics[idx].items():
        if k in dic_w:
            assert k in dic_pos
            if v[0] == dic_w[k]:
                correct_w += 1
                if v[0] + '_' + v[1] == dic_pos[k]:
                    correct_pos += 1
                else:
                    incorrect_pos.append(v[0] + '_' + v[1])
            else:
                incorrect_w.append(v)
                incorrect_pos.append(v[0] + '_' + v[1])
    idx += 1

print 'Correct predicted OOV words: %d (plain), %d (POS).' % (correct_w, correct_pos)
print 'Percentages of correct predicted OOV words: %f (plain), %f (POS).' % (float(correct_w)/oov_w, float(correct_pos)/oov_w)
print ''
'''
print 'Incorrect segmentations: '
for v in incorrect_w:
    print v.encode('utf-8')
print ''
print 'Incorrect segmentations & POS tags: '
for v in incorrect_pos:
    print v.encode('utf-8')
print ''
'''