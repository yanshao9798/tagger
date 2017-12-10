# -*- coding: utf-8 -*-
import codecs
import os
import numpy as np
import sys
import tensorflow as tf
import math
import re
import pygame
import copy


from evaluation import score, score_boundaries

sys.stdout = codecs.getwriter('utf8')(sys.stdout)


def get_ngrams(raw, gram):
    gram_set = set()
    li = gram/2
    ri = gram - li - 1
    p = '<PAD>'
    for line in raw:
        for i in range(len(line)):
            if i - li < 0:
                lp = p * (li - i) + line[:i]
            else:
                lp = line[i - li:i]
            if i + ri + 1 > len(line):
                rp = line[i:] + p*(i + ri + 1 - len(line))
            else:
                rp = line[i:i+ri+1]
            ch = lp + rp
            gram_set.add(ch)
    return gram_set


def get_vocab_tag(path, filelist, ngram=1):
    out_char = codecs.open(path + '/chars.txt', 'w', encoding='utf-8')
    out_tag = codecs.open(path + '/tags.txt', 'w', encoding='utf-8')
    char_set = set()
    tag_set = {}
    raw = []
    for file_name in filelist:
        for line in codecs.open(path + '/' + file_name, 'rb', encoding='utf-8'):
            line = line.strip()
            raw_l = ''
            sets = line.split(' ')
            if len(sets) > 0:
                for seg in sets:
                    spos = seg.split('_')
                    if len(spos) == 2:
                        for ch in spos[0]:
                            char_set.add(ch)
                            raw_l += ch
                        if spos[1] in tag_set:
                            if tag_set[spos[1]] < len(spos[0]):
                                tag_set[spos[1]] = len(spos[0])
                        else:
                            tag_set[spos[1]] = len(spos[0])
                raw.append(raw_l)
            elif len(line) == 0:
                continue
            else:
                print line
                raise Exception('Check your text file.')
    char_set = list(char_set)
    #tag_set = list(tag_set)
    if ngram > 1:
        for i in range(2, ngram + 1):
            out_gram = codecs.open(path + '/' + str(i) + 'gram.txt', 'w', encoding='utf-8')
            grams = get_ngrams(raw, i)
            for g in grams:
                out_gram.write(g + '\n')
            out_gram.close()
    for item in char_set:
        out_char.write(item + '\n')
    out_char.close()
    for k, v in tag_set.items():
        out_tag.write(k + ' ' + str(v) + '\n')
    out_tag.close()


def read_vocab_tag(path, ngrams=1):
    char_set = set()
    tag_set = {}
    ngram_set = None
    for line in codecs.open(path + '/chars.txt', 'rb', encoding='utf-8'):
        char_set.add(line.strip())
    for line in codecs.open(path + '/tags.txt', 'rb', encoding='utf-8'):
        line = line.strip()
        sp = line.split(' ')
        tag_set[sp[0]] = int(sp[1])
    char_set = list(char_set)
    if ngrams > 1:
        ngram_set = []
        for i in range(2, ngrams + 1):
            ng_set = set()
            for line in codecs.open(path + '/' + str(i) + 'gram.txt', 'rb', encoding='utf-8'):
                line = line.strip()
                ng_set.add(line)
            ngram_set.append(ng_set)
    return char_set, tag_set, ngram_set


def get_sample_embedding(path, emb, chars, default='unk'):
    short_emb = emb[emb.index('/') + 1: emb.index('.')]
    emb_dic = {}
    for line in codecs.open(emb, 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        emb_dic[sets[0]] = np.asarray(sets[1:], dtype='float32')
    emb_dim = len(emb_dic.values()[0])
    fout = codecs.open(path + '/' + short_emb + '_sub.txt', 'w', encoding='utf-8')
    p_line = '<P>'
    if '<P>' in emb_dic:
        for emb in emb_dic['<P>']:
            p_line += ' ' + unicode(emb)
    else:
        rand_emb = np.random.uniform(-math.sqrt(float(3)/emb_dim), math.sqrt(float(3)/ emb_dim), emb_dim)
        for emb in rand_emb:
            p_line += ' ' + unicode(emb)
    fout.write(p_line + '\n')
    p_line = '<UNK>'
    if '<UNK>' in emb_dic:
        for emb in emb_dic['<UNK>']:
            p_line += ' ' + unicode(emb)
    else:
        rand_emb = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
        emb_dic['<UNK>'] = rand_emb
        for emb in rand_emb:
            p_line += ' ' + unicode(emb)
    fout.write(p_line + '\n')

    p_line = '<NUM>'
    if '<NUM>' in emb_dic:
        for emb in emb_dic['<NUM>']:
            p_line += ' ' + unicode(emb)
    else:
        rand_emb = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
        for emb in rand_emb:
            p_line += ' ' + unicode(emb)
    fout.write(p_line + '\n')

    p_line = '<FW>'
    if '<FW>' in emb_dic:
        for emb in emb_dic['<FW>']:
            p_line += ' ' + unicode(emb)
    else:
        rand_emb = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
        for emb in rand_emb:
            p_line += ' ' + unicode(emb)
    fout.write(p_line + '\n')

    for ch in chars:
        p_line = ch
        if ch in emb_dic:
            for emb in emb_dic[ch]:
                p_line += ' ' + unicode(emb)
        else:
            if default == 'unk':
                for emb in emb_dic['<UNK>']:
                    p_line += ' ' + unicode(emb)
            else:
                rand_emb = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
                for emb in rand_emb:
                    p_line += ' ' + unicode(emb)
        fout.write(p_line + '\n')
    fout.close()


def get_ngram_embedding(path, emb, ngrams, default='unk'):
    for gram in ngrams:
        n = len(list(gram)[0])
        real_emb = emb + '_' + str(n) + 'gram.txt'
        get_sample_embedding(path, real_emb, gram, default=default)


def read_sample_embedding(path, short_emb):
    emb_values = []
    for line in codecs.open(path + '/' + short_emb + '_sub.txt', 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        emb_values.append(np.asarray(sets[1:], dtype='float32'))
    emb_dim = len(emb_values[0])
    return emb_dim, emb_values


def read_ngram_embedding(path, short_emb, n):
    embs = []
    for i in range(2, n + 1):
        _, emb_v = read_sample_embedding(path, short_emb + '_' + str(i) + 'gram')
        embs.append(emb_v)
    return embs


def get_chars_pixels(path, chars, font, pt_size, utf8=True):
    pix_dic = {}
    pygame.init()
    ft = pygame.font.Font('fonts/' + font, pt_size)
    font_name = font[:font.index('.')]
    fout = codecs.open(path + '/' + font_name + str(pt_size) + '_pixels.txt', 'w', encoding='utf-8')
    zeros = np.zeros((pt_size, pt_size), dtype='float32').flatten()
    pix_dic['<P>'] = zeros
    p_line = '<P>'
    for n in zeros:
        p_line += ' ' + unicode(n)
    fout.write(p_line + '\n')
    p_line = '<UNK>'
    pix_dic['<UNK>'] = zeros
    for n in zeros:
        p_line += ' ' + unicode(n)
    fout.write(p_line + '\n')
    p_line = '<NUM>'
    ch = '0'
    if utf8:
        u_ch = ch
    else:
        u_ch = ch.decode('utf-8')
    rtext = ft.render(u_ch, True, (0, 0, 0), (255, 255, 255))
    ch_ary = pygame.surfarray.array2d(rtext)
    ch_ary = ch_ary[:, :pt_size]
    ch_ary = ch_ary[: pt_size]
    if ch_ary.shape[0] < pt_size:
        ch_ary = np.pad(ch_ary, ((pt_size - ch_ary.shape[0], 0), (0, 0)), 'constant', constant_values=0)
    ch_ary = ch_ary.flatten()
    for n in ch_ary:
        p_line += ' ' + unicode(float(n)/255)
    fout.write(p_line + '\n')

    p_line = '<FW>'
    ch = 'a'
    if utf8:
        u_ch = ch
    else:
        u_ch = ch.decode('utf-8')
    rtext = ft.render(u_ch, True, (0, 0, 0), (255, 255, 255))
    ch_ary = pygame.surfarray.array2d(rtext)
    ch_ary = ch_ary[:, :pt_size]
    ch_ary = ch_ary[: pt_size]
    if ch_ary.shape[0] < pt_size:
        ch_ary = np.pad(ch_ary, ((pt_size - ch_ary.shape[0], 0), (0, 0)), 'constant', constant_values=0)
    ch_ary = ch_ary.flatten()
    for n in ch_ary:
        p_line += ' ' + unicode(float(n) / 255)
    fout.write(p_line + '\n')

    for ch in chars:
        p_line = ch
        if utf8:
            u_ch = ch
        else:
            u_ch = ch.decode('utf-8')
        rtext = ft.render(u_ch, True, (0, 0, 0), (255, 255, 255))
        ch_ary = pygame.surfarray.array2d(rtext)
        ch_ary = ch_ary[:, :pt_size]
        ch_ary = ch_ary[: pt_size]
        if ch_ary.shape[0] < pt_size:
            ch_ary = np.pad(ch_ary, ((pt_size - ch_ary.shape[0], 0), (0, 0)), 'constant', constant_values=0)
        ch_ary = ch_ary.flatten()
        pix_dic[ch] = ch_ary.astype('float32')
        for n in ch_ary:
            p_line += ' ' + unicode(float(n)/255)
        fout.write(p_line + '\n')
    fout.close()


def read_chars_pixels(path, font_name, pt_size):
    pix_dic = []
    for line in codecs.open(path + '/' + font_name + str(pt_size) +  '_pixels.txt', 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        ch_ary = np.array(sets[1:], dtype='float32')
        pix_dic.append(ch_ary)
    return pix_dic


def update_char_dict(char2idx, new_chars, valid_chars=None):
    dim = len(char2idx)
    unk_dim = dim
    o_dim = dim
    unk_char2idx = copy.copy(char2idx)
    unk_idx = char2idx['<UNK>']
    for char in new_chars:
        if char not in char2idx:
            char2idx[char] = dim
            if valid_chars is not None and char in valid_chars and unk_dim - o_dim < 500:
                unk_char2idx[char] = unk_dim
            else:
                unk_char2idx[char] = unk_idx
            dim += 1
    idx2char = {k: v for v, k in char2idx.items()}
    return char2idx, idx2char, unk_char2idx


def update_gram_dicts(gram2idx, new_grams):
    assert len(gram2idx) == len(new_grams)
    new_gram2idx = []
    for dic, n_gram in zip(gram2idx, new_grams):
        assert len(dic.keys()[0]) == len(n_gram[0])
        new_dic, _, _ = update_char_dict(dic, n_gram)
        new_gram2idx.append(new_dic)
    return new_gram2idx


def get_radical_dic(path='radical.txt'):
    rad_dic= {}
    for line in codecs.open(path, 'r', encoding='utf-8'):
        line = line.strip()
        rad_dic[ord(line)] = line
    return rad_dic


def get_radical_idx(ch, rad_dic, keys=None):
    if keys is None:
        keys = rad_dic.keys()
        keys = sorted(keys)
    idx = ord(ch)
    if idx < keys[0] or idx > keys[-1] + 6:
        return '<NULL>'
    else:
        pre = 0
        for k in keys:
            if k > idx:
                break
            pre = k
        return pre


def get_new_chars(path, char2idx, type='ctb'):
    new_chars = set()
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        if type == 'ctb':
            segs = line.split(' ')
            for seg in segs:
                items = seg.split('_')
                assert len(items) == 2
                for ch in items[0]:
                    if ch not in char2idx:
                        new_chars.add(ch)
        else:
            line = re.sub('[\s+]', '', line)
            for ch in line:
                if ch not in char2idx:
                    new_chars.add(ch)
    return new_chars


def get_new_chars_raw(lines, char2idx, type='ctb'):
    new_chars = set()
    for line in lines:
        line = line.strip()
        if type == 'ctb':
            segs = line.split(' ')
            for seg in segs:
                items = seg.split('_')
                assert len(items) == 2
                for ch in items[0]:
                    if ch not in char2idx:
                        new_chars.add(ch)
        else:
            line = re.sub('[\s+]', '', line)
            for ch in line:
                if ch not in char2idx:
                    new_chars.add(ch)
    return new_chars


def get_new_grams(path, gram2idx, type='ctb'):
    raw = []
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        raw_l = ''
        if type == 'ctb':
            segs = line.split(' ')
            for seg in segs:
                items = seg.split('_')
                assert len(items) == 2
                raw_l += items[0]
        else:
            line = re.sub('[\s+]', '', line)
            raw_l = line
        raw.append(raw_l)
    new_grams = []
    for g_dic in gram2idx:
        new_g = []
        n = len(g_dic.keys()[0])
        grams = get_ngrams(raw, n)
        for g in grams:
            if g not in g_dic:
                new_g.append(g)
        new_grams.append(new_g)
    return new_grams


def get_new_embeddings(new_chars, emb_dim, emb_path=None):
    if emb_path is None:
        return tf.random_uniform([len(new_chars), emb_dim], -math.sqrt(3 / emb_dim), math.sqrt(3 / emb_dim))
    else:
        assert os.path.isfile(emb_path)
        emb = {}
        new_emb = []
        for line in codecs.open(emb_path, 'rb', encoding='utf-8'):
            line = line.strip()
            sets = line.split(' ')
            emb[sets[0]] = np.asarray(sets[1:], dtype='float32')
        if '<UNK>' not in emb:
            unk = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
            emb['<UNK>'] = np.asarray(unk, dtype='float32')
        for ch in new_chars:
            if ch in emb:
                new_emb.append(emb[ch])
            else:
                new_emb.append(emb['<UNK>'])
        return tf.stack(new_emb)


def get_new_ng_embeddings(new_grams, emb_dim, emb_path=None):
    new_embs = []
    for i in range(len(new_grams)):
        n = len(new_grams[i][0])
        if emb_path is not None:
            real_path = emb_path + '_' + str(n) + 'gram.txt'
        else:
            real_path = None
        n_emb = get_new_embeddings(new_grams[i], emb_dim, real_path)
        new_embs.append(n_emb)
    return new_embs


def get_valid_chars(chars, emb_path):
    valid_chars = []
    total = []
    for line in codecs.open(emb_path, 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        total.append(sets[0])
    for ch in chars:
        if ch in total:
            valid_chars.append(ch)
    return valid_chars


def get_valid_grams(ngram, emb_path):
    valid_grams = []
    for gram in ngram:
        valid = []
        n = len(gram[0])
        real_path = emb_path + '_' + str(n) + 'gram.txt'
        total = []
        for line in codecs.open(real_path, 'rb', encoding='utf-8'):
            line = line.strip()
            sets = line.split(' ')
            total.append(sets[0])
        for g in gram:
            if g in total:
                valid.append(g)
        valid_grams.append(valid)
    return valid_grams


def get_new_pixels(new_chars, font, pt_size):
    new_pixels = []
    pygame.init()
    ft = pygame.font.Font('fonts/' + font, pt_size)
    for ch in new_chars:
        rtext = ft.render(ch.decode('utf-8'), True, (0, 0, 0), (255, 255, 255))
        ch_ary = pygame.surfarray.array2d(rtext)
        ch_ary = ch_ary[:, :pt_size]
        if ch_ary.shape[0] < pt_size:
            ch_ary = np.pad(ch_ary, ((pt_size - ch_ary.shape[0], 0), (0, 0)), 'constant', constant_values=0)
        ch_ary = ch_ary.flatten()
        new_pixels.append(np.asarray(ch_ary, dtype='float32')/255)
    return new_pixels


def down_pool(pixel_dim, pooling_size):
    if pixel_dim % pooling_size == 0:
        p_size = pixel_dim / pooling_size
    else:
        p_size = pixel_dim / pooling_size + 1

    return p_size


def next_batch(x, y, start_idx, batch_size):
    last_idx = start_idx + batch_size
    batch_x = x[start_idx:last_idx]
    batch_y = y[start_idx:last_idx]
    return batch_x, batch_y


def viterbi(max_scores, max_scores_pre, length, batch_size):
    best_paths = []
    for m in range(batch_size):
        path = []
        last_max_node = np.argmax(max_scores[m][length[m] - 1])
        path.append(last_max_node)
        for t in range(1, length[m])[::-1]:
            last_max_node = max_scores_pre[m][t][last_max_node]
            path.append(last_max_node)
        path = path[::-1]
        best_paths.append(path)
    return best_paths


def get_comb_tags(tags, tag_type):
    tag2index = {}
    tag2index['<P>'] = 0
    idx = 1
    for k, v in tags.items():
        real_tag_type = tag_type
        if v == 1:
            if tag_type == 'BIES':
                real_tag_type = tag_type[-1:]
            else:
                real_tag_type = tag_type[0]
        elif v == 2:
            if tag_type == 'BIES' or tag_type == 'BIE':
                real_tag_type = tag_type[: 1] + tag_type[-2:]
        for t_type in real_tag_type:
            tag2index[str(t_type + '-' + k)] = idx
            idx += 1
    return tag2index


def get_dic(chars, tags):
    char2index = {}
    char2index['<P>'] = 0
    char2index['<UNK>'] = 1
    char2index['<NUM>'] = 2
    char2index['<FW>'] = 3
    idx = 4
    for ch in chars:
        char2index[ch] = idx
        idx += 1
    index2char = {v: k for k, v in char2index.items()}

    #0.seg BIES  1. BI; 2. BIE; 3. BIES
    seg_tags2index = {'<P>':0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
    tag2index = {'seg': seg_tags2index, 'BI': get_comb_tags(tags, 'BI'), 'BIE': get_comb_tags(tags, 'BIE'), 'BIES':
        get_comb_tags(tags, 'BIES')}
    index2tag = {}
    for dic_keys in tag2index:
        index2tag[dic_keys] = {v: k for k, v in tag2index[dic_keys].items()}
    return char2index, index2char, tag2index, index2tag


def get_ngram_dic(ngrams):
    gram_dics = []
    for i, gram in enumerate(ngrams):
        g_dic = {}
        g_dic['<P>'] = 0
        g_dic['<UNK>'] = 1
        idx = 2
        for g in gram:
            g_dic[g] = idx
            idx += 1
        gram_dics.append(g_dic)
    return gram_dics


def sub_num(x, char2index):
    num_k = 0
    fw_k = 0
    num_set = set()
    fw_set = set()
    for k in char2index.keys():
        if k == '<NUM>':
            num_k = char2index[k]
        elif k == '<FW>':
            fw_k = char2index[k]
        elif ('0' <= k <= '9') | ('０' <= k <= '９'):
            num_set.add(char2index[k])
        elif ('Ａ' <= k <= 'Ｚ') | ('ａ' <= k <= 'ｚ') | ('A' <= k <= 'Z') | ('a' <= k <= 'z'):
            fw_set.add(char2index[k])

    if num_k == 0:
        raise Exception('<NUM> key is not contained in the dictionary')

    if fw_k == 0:
        raise Exception('<FW> key is not contained in the dictionary')

    for l in x:
        for idx, ch in enumerate(l):
            if ch in num_set:
                l[idx] = num_k
            elif ch in fw_set:
                l[idx] = fw_k
    return x


def get_input_vec(path, fname, char2index, tag2index, rad_dic=None, tag_scheme='BIES'):
    max_sent_len_c = 0
    max_sent_len_w = 0
    max_word_len = 0
    t_len = 0
    key_map = {}
    keys = []
    if rad_dic is None:
        x_m = [[]]
    else:
        x_m = [[], []]
        keys = sorted(rad_dic.keys())
        key_map['<NULL>'] = 0
        idx = 1
        for k in keys:
            key_map[k] = idx
            idx += 1

    y_m = [[]]

    for line in codecs.open(path + '/' + fname, 'r', encoding='utf-8'):
        charIndices = []
        raw_l = ''
        if rad_dic is not None:
            radIndices = []

        tagIndices = {}

        for k in tag2index.keys():
            tagIndices[k] = []
        line = line.strip()
        segs = line.split(' ')

        if len(segs) > max_sent_len_w:
            max_sent_len_w = len(segs)
        if len(segs) > 0 and len(line) > 0:
            for seg in segs:
                splits = seg.split('_')
                assert len(splits) == 2

                w_len = len(splits[0])
                raw_l += splits[0]
                if w_len > max_word_len:
                    max_word_len = w_len

                t_len += w_len

                if w_len == 1:
                    charIndices.append(char2index[splits[0]])
                    if rad_dic is not None:
                        radIndices.append(key_map[get_radical_idx(splits[0], rad_dic, keys)])

                    tagIndices['seg'].append(tag2index['seg']['S'])
                    tagIndices['BI'].append(tag2index['BI']['B-' + splits[1]])
                    tagIndices['BIE'].append(tag2index['BIE']['B-' + splits[1]])
                    tagIndices['BIES'].append(tag2index['BIES']['S-' + splits[1]])

                else:

                    for x in range(w_len):
                        c_ch = splits[0][x]
                        charIndices.append(char2index[c_ch])

                        if rad_dic is not None:
                            radIndices.append(key_map[get_radical_idx(c_ch, rad_dic, keys)])

                        if x == 0:

                            tagIndices['seg'].append(tag2index['seg']['B'])

                            tagIndices['BI'].append(tag2index['BI']['B-' + splits[1]])
                            tagIndices['BIE'].append(tag2index['BIE']['B-' + splits[1]])
                            tagIndices['BIES'].append(tag2index['BIES']['B-' + splits[1]])

                        elif x == len(splits[0]) - 1:

                            tagIndices['seg'].append(tag2index['seg']['E'])

                            tagIndices['BI'].append(tag2index['BI']['I-' + splits[1]])
                            tagIndices['BIE'].append(tag2index['BIE']['E-' + splits[1]])
                            tagIndices['BIES'].append(tag2index['BIES']['E-' + splits[1]])

                        else:

                            tagIndices['seg'].append(tag2index['seg']['I'])

                            tagIndices['BI'].append(tag2index['BI']['I-' + splits[1]])
                            tagIndices['BIE'].append(tag2index['BIE']['I-' + splits[1]])
                            tagIndices['BIES'].append(tag2index['BIES']['I-' + splits[1]])

            if t_len > max_sent_len_c:
                max_sent_len_c = t_len
            t_len = 0
            x_m[0].append(charIndices)
            if rad_dic is not None:
                x_m[1].append(radIndices)

            y_m[0].append(tagIndices[tag_scheme])
    return x_m, y_m, max_sent_len_c, max_sent_len_w, max_word_len


def gram_vec(raw, dic):
    out = []
    ngram = len(dic.keys()[0])
    li = ngram/2
    ri = ngram - li - 1
    p = '<PAD>'
    for line in raw:
        indices = []
        for i in range(len(line)):
            if i - li < 0:
                lp = p * (li - i) + line[:i]
            else:
                lp = line[i - li:i]
            if i + ri + 1 > len(line):
                rp = line[i:] + p*(i + ri + 1 - len(line))
            else:
                rp = line[i:i+ri+1]
            ch = lp + rp
            if ch in dic:
                indices.append(dic[ch])
            else:
                indices.append(dic['<UNK>'])
        out.append(indices)
    return out


def get_gram_vec(path, fname, gram2index, is_raw=False):
    raw = []

    if path is None:
        real_path = fname
    else:
        real_path = path + '/' + fname

    if is_raw:
        for line in codecs.open(real_path, 'r', encoding='utf-8'):
            line = line.strip()
            raw.append(line)
    else:
        for line in codecs.open(real_path, 'r', encoding='utf-8'):
            line = line.strip()
            segs = line.split(' ')
            if len(segs) > 0 and len(line) > 0:
                raw_l = ''
                for seg in segs:
                    sp = seg.split('_')
                    if len(sp) == 2:
                        raw_l += sp[0]
                raw.append(raw_l)
    out = []
    for g_dic in gram2index:
        out.append(gram_vec(raw, g_dic))
    return out


def get_gram_vec_raw(raw, gram2index):
    out = []
    for g_dic in gram2index:
        out.append(gram_vec(raw, g_dic))
    return out


def get_input_vec_raw(path, fname, char2index, rad_dic=None):
    max_len = 0
    key_map = {}
    keys = []
    if rad_dic is None:
        x_m = [[]]
    else:
        x_m = [[], []]
        keys = sorted(rad_dic.keys())
        key_map['<NULL>'] = 0
        idx = 1
        for k in keys:
            key_map[k] = idx
            idx += 1

    if path is None:
        real_path = fname
    else:
        real_path = path + '/' + fname

    for line in codecs.open(real_path, 'r', encoding='utf-8'):
        charIndices = []
        radIndices = []
        line = re.sub('[\s+]', '', line.strip())
        if len(line) > max_len:
            max_len = len(line)
        for ch in line:
            charIndices.append(char2index[ch])
            if rad_dic is not None:
                radIndices.append(key_map[get_radical_idx(ch, rad_dic, keys)])
        x_m[0].append(charIndices)
        if rad_dic is not None:
            x_m[1].append(radIndices)
    return x_m, max_len



def get_input_vec_line(lines, char2index, rad_dic=None):
    max_len = 0
    key_map = {}
    keys = []
    if rad_dic is None:
        x_m = [[]]
    else:
        x_m = [[], []]
        keys = sorted(rad_dic.keys())
        key_map['<NULL>'] = 0
        idx = 1
        for k in keys:
            key_map[k] = idx
            idx += 1

    for line in lines:
        charIndices = []
        radIndices = []
        line = re.sub('[\s+]', '', line)
        if len(line) > max_len:
            max_len = len(line)
        for ch in line:
            charIndices.append(char2index[ch])
            if rad_dic is not None:
                radIndices.append(key_map[get_radical_idx(ch, rad_dic, keys)])
        x_m[0].append(charIndices)
        if rad_dic is not None:
            x_m[1].append(radIndices)
    return x_m, max_len


def pad_zeros(l, max_len):
    if type(l) is list:
        return [np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0) for item in l]
    elif type(l) is dict:
        padded = {}
        for k, v in l.iteritems():
            padded[k] = [np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0) for item in v]
        return padded


def unpad_zeros(l):
    out = []
    for tags in l:
        out.append([np.trim_zeros(line) for line in tags])
    return out


def decode_tags(idx, index2tags, tag_scheme):
    out = []

    dic = index2tags[tag_scheme]
    for id in idx:
        sents = []
        for line in id:
            sent = []
            for item in line:
                tag = dic[item]
                if '-' in tag:
                    tag = tag.replace('E-', 'I-')
                    tag = tag.replace('S-', 'B-')
                else:
                    tag = tag.replace('E', 'I')
                    tag = tag.replace('S', 'B')
                sent.append(tag)
            sents.append(sent)
        out.append(sents)
    return out


def decode_chars(idx, idx2chars):
    out = []
    for line in idx:
        line = np.trim_zeros(line)
        out.append([idx2chars[item] for item in line])
    return out


def trim_output(out, length):
    assert len(out) == len(length)
    trimmed_out = []
    for item, l in zip(out, length):
        trimmed_out.append(item[:l])
    return trimmed_out


def get_nums_tags(tag2idx, tag_scheme):
    nums_tags = [len(tag2idx[tag_scheme])]
    return nums_tags


def generate_output(chars, tags, tag_scheme):
    out = []
    for i, tag in enumerate(tags):
        assert len(chars) == len(tag)
        sub_out = []
        for chs, tgs in zip(chars, tag):
            #print len(chs), len(tgs)
            assert len(chs) == len(tgs)
            c_word = ''
            c_tag = ''
            p_line = ''
            for ch, tg in zip(chs, tgs):
                if tag_scheme == 'seg':
                    if tg == 'I':
                        c_word += ch
                    else:
                        p_line += ' ' + c_word + '_' + '<UNK>'
                        c_word = ch
                else:
                    tg_sets = tg.split('-')
                    if tg_sets[0] == 'I' and tg_sets[1] == c_tag:
                        c_word += ch
                    else:
                        p_line += ' ' + c_word + '_' + c_tag
                        c_word = ch
                        if len(tg_sets) < 2:
                            c_tag = '<UNK>'
                        else:
                            c_tag = tg_sets[1]
            if len(c_word) > 0:
                if tag_scheme == 'seg':
                    p_line += ' ' + c_word + '_' + '<UNK>'
                elif len(c_tag) > 0:
                    p_line += ' ' + c_word + '_' + c_tag
            if tag_scheme == 'seg':
                sub_out.append(p_line[8:])
            else:
                sub_out.append(p_line[3:])
        out.append(sub_out)
    return out


def evaluator(prediction, gold, metric='F1-score', tag_num=1, verbose=False):
    assert len(prediction) == len(gold)
    scores = (0, 0, 0, 0, 0, 0)
    scores_b = (0, 0, 0, 0, 0, 0)
    if metric in ['F1-score', 'Precision', 'Recall', 'All']:
        scores = score(gold[0], prediction[0], tag_num,  verbose)
    if metric in ['Boundary-F1-score', 'All']:
        scores_b = score_boundaries(gold[0], prediction[0],  verbose)
    return scores + scores_b



def printer(predictions, out_path):
    fout = codecs.open(out_path, 'w', encoding='utf-8')
    for line in predictions:
        fout.write(line + '\n')
    fout.close()


def buckets(x, y, size=10):
    assert len(x[0]) == len(y[0])
    num_inputs = len(x)
    samples = x + y
    num_items = len(samples)
    xy = zip(*samples)
    xy.sort(key=lambda i: len(i[0]))
    t_len = size
    idx = 0
    bucks = [[[]] for _ in range(num_items)]
    for item in xy:
        if len(item[0]) > t_len:
            if len(bucks[0][idx]) > 0:
                for buck in bucks:
                    buck.append([])
                idx += 1
            while len(item[0]) > t_len:
                t_len += size
        for i in range(num_items):
            bucks[i][idx].append(item[i])

    return bucks[:num_inputs], bucks[num_inputs:]


def pad_bucket(x, y, bucket_len_c=None):
    assert len(x[0]) == len(y[0])
    num_inputs = len(x)
    num_tags = len(y)
    padded = [[] for _ in range(num_tags + num_inputs)]
    bucket_counts = []
    samples = x + y
    xy = zip(*samples)
    if bucket_len_c is None:
        bucket_len_c = []
        for item in xy:
            max_len = len(item[0][-1])
            bucket_len_c.append(max_len)
            bucket_counts.append(len(item[0]))
            for idx in range(num_tags + num_inputs):
                padded[idx].append(pad_zeros(item[idx], max_len))
        print 'Number of buckets: ', len(bucket_len_c)
    else:
        idy = 0
        for item in xy:
            max_len = len(item[0][-1])
            while idy < len(bucket_len_c) and max_len > bucket_len_c[idy]:
                idy += 1
            bucket_counts.append(len(item[0]))
            if idy >= len(bucket_len_c):
                for idx in range(num_tags + num_inputs):
                    padded[idx].append(pad_zeros(item[idx], max_len))
                bucket_len_c.append(max_len)
            else:
                for idx in range(num_tags + num_inputs):
                    padded[idx].append(pad_zeros(item[idx], bucket_len_c[idy]))

    return padded[:num_inputs], padded[num_inputs:], bucket_len_c, bucket_counts


def merge_bucket(x):
    out = []
    for item in x:
        m = []
        for i in item:
            m += i
        out.append(m)
    return out


def get_real_batch(counts, b_size):
    real_batch_sizes = []
    for c in counts:
        if c < b_size:
            real_batch_sizes.append(c)
        else:
            real_batch_sizes.append(b_size)
    return real_batch_sizes


def get_batch_pixels(ids, pixels):
    out = []
    for id in ids:
        l_out = [pixels[idx] for idx in id]
        out.append(l_out)
    return np.asarray(out)


def get_maxstep(raw_file, bt_size):
    max_bt = 300 / bt_size + 1
    wt = [None]
    wt[0] = codecs.open(raw_file + '_' + str(0), 'w', encoding='utf-8')
    maxstep = 0
    wt_long_idx = codecs.open(raw_file + '_lidx', 'w', encoding='utf-8')
    for line in codecs.open(raw_file, 'r', encoding='utf-8'):
        line = line.strip()
        l_len = len(line)
        if len(line) > maxstep:
            maxstep = l_len
            if maxstep > 300:
                if len(wt) < max_bt:
                    for i in range(len(wt), max_bt):
                        wt.append(codecs.open(raw_file + '_' + str(i), 'w', encoding='utf-8'))
                wt[max_bt - 1].write(line + '\n')
                wt_long_idx.write(str(max_bt - 1) + '\n')
            else:
                bt_idx = (l_len - 1) / bt_size
                if bt_idx > len(wt) - 1:
                    for i in range(len(wt), bt_idx + 1):
                        wt.append(codecs.open(raw_file + '_' + str(i), 'w', encoding='utf-8'))
                wt[bt_idx].write(line + '\n')
                wt_long_idx.write(str(bt_idx) + '\n')
        elif l_len > 0:
            if l_len > 300:
                wt[max_bt - 1].write(line + '\n')
                wt_long_idx.write(str(max_bt - 1) + '\n')
            else:
                bt_idx = (l_len - 1) / bt_size
                wt[bt_idx].write(line + '\n')
                wt_long_idx.write(str(bt_idx) + '\n')
    for i in range(len(wt)):
        wt[i].close()
    wt_long_idx.close()
    return maxstep


def merge_files(out_path, raw_file, bt_num):
    wt = codecs.open(out_path, 'w', encoding='utf-8')
    rd = []
    for i in range(bt_num):
        rd.append(codecs.open(out_path + '_' + str(i), 'r', encoding='utf-8'))
    for line in codecs.open(raw_file + '_lidx', 'r', encoding='utf-8'):
        line = line.strip()
        r_idx = int(line)
        line = rd[r_idx].readline()
        wt.write(line)
    wt.close()

    for i in range(bt_num):
        rd[i].close()
        os.remove(raw_file + '_' + str(i))
        os.remove(out_path + '_' + str(i))

    os.remove(raw_file + '_lidx')


