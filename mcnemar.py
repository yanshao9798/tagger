# -*- coding: utf-8 -*-

import sys
import codecs
import copy
from scipy.stats import binom

gold = sys.argv[1]
ref1 = sys.argv[2]
ref2 = sys.argv[3]


def reader(path):
    sents = []
    for line in codecs.open(path, 'r', encoding='utf-8'):
        line = line.strip()
        sent = []
        segs = line.split(' ')
        for seg in segs:
            sent.append(seg.split('_'))
        sents.append(sent)
    return sents


def evaluate(gold, ref):
    idx_g = 0
    idx_r = 0
    seg = []
    pos = []
    while gold and ref:
        if gold[0][0] == ref[0][0]:
            seg.append(1)
            if gold[0][1] == ref[0][1]:
                pos.append(1)
            else:
                pos.append(0)
            gold.pop(0)
            ref.pop(0)
        else:
            if idx_g == idx_r:
                idx_g += len(gold[0][0])
                idx_r += len(ref[0][0])
                seg.append(0)
                pos.append(0)
                gold.pop(0)
                ref.pop(0)
            elif idx_g < idx_r:
                idx_g += len(gold[0][0])
                seg.append(0)
                pos.append(0)
                gold.pop(0)
            else:
                idx_r += len(ref[0][0])
                ref.pop(0)
    while gold:
        seg.append(0)
        pos.append(0)
        gold.pop(0)
    assert len(seg) == len(pos)
    return seg, pos


def compare(s1, s2):
    a, b, c, d = 0, 0, 0, 0
    assert len(s1) == len(s2)
    for sa, sb in zip(s1, s2):
        if sa != sb:
            if sa == 1:
                b += 1
            else:
                c += 1
        else:
            if sa == 1:
                a += 1
            else:
                d += 1
    return a, b, c, d


def mcnemar_midp(b, c):
    """
    Compute McNemar's test using the "mid-p" variant suggested by:

    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for
    binary matched-pairs data: Mid-p and asymptotic are better than exact
    conditional. BMC Medical Research Methodology 13: 91.

    `b` is the number of observations correctly labeled by the first---but
    not the second---system; `c` is the number of observations correctly
    labeled by the second---but not the first---system.
    """
    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    return midp


gold = reader(gold)
ref1 = reader(ref1)
ref2 = reader(ref2)

assert len(gold) == len(ref1)
assert len(gold) == len(ref2)

v_a1 = 0
v_d1 = 0

v_b1 = 0
v_c1 = 0

v_a2 = 0
v_d2 = 0

v_b2 = 0
v_c2 = 0

for g, r1, r2 in zip(gold, ref1, ref2):
    g1 = copy.copy(g)
    seg1, pos1 = evaluate(g1, r1)
    seg2, pos2 = evaluate(g, r2)
    stats1 = compare(seg1, seg2)
    stats2 = compare(pos1, pos2)
    v_b1 += stats1[1]
    v_c1 += stats1[2]

    v_a1 += stats1[0]
    v_d1 += stats1[3]

    v_b2 += stats2[1]
    v_c2 += stats2[2]

    v_a2 += stats2[0]
    v_d2 += stats2[3]


print 'Segmentation:'

print '\t\t' + 't2 positve' + '\t' + 't2 negative' + '\t' + 'row total'
print 't1 positive' + '\t' + str(v_a1) + '\t\t' + str(v_b1) + '\t\t' + str(v_a1 + v_b1)
print 't1 negative' + '\t' + str(v_c1) + '\t\t' + str(v_d1) + '\t\t' + str(v_c1 + v_d1)
print 'column total' + '\t' + str(v_c1 + v_a1) + '\t\t' + str(v_d1 + v_b1) + '\n'

print 'mid-p value: ', mcnemar_midp(v_b1, v_c1)

print 'POS tagging:'

print '\t\t' + 't2 positve' + '\t' + 't2 negative' + '\t' + 'row total'
print 't1 positive' + '\t' + str(v_a2) + '\t\t' + str(v_b2) + '\t\t' + str(v_a2 + v_b2)
print 't1 negative' + '\t' + str(v_c2) + '\t\t' + str(v_d2) + '\t\t' + str(v_c2 + v_d2)
print 'column total' + '\t' + str(v_c2 + v_a2) + '\t\t' + str(v_d2 + v_b2) + '\n'

print 'mid-p value: ', mcnemar_midp(v_b2, v_c2)