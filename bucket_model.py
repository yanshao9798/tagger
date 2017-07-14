# -*- coding: utf-8 -*-
import tensorflow as tf
from layers import EmbeddingLayer, BiLSTM, HiddenLayer, TimeDistributed, DropoutLayer, Convolution, Maxpooling, Forward
from time import time
import losses
import toolbox
import batch as Batch
import numpy as np
import random
import cPickle as pickle
import math


class Model(object):

    def __init__(self, nums_chars, nums_tags, buckets_char, counts=None, pic_size=None, font=None, batch_size=10, tag_scheme='BIES', word_vec=True, radical=False, graphic=False, crf=1, ngram=None, metric='F1-score'):
        self.nums_chars = nums_chars
        self.nums_tags = nums_tags
        self.buckets_char = buckets_char
        self.counts = counts
        self.tag_scheme = tag_scheme
        self.graphic = graphic
        self.word_vec = word_vec
        self.radical = radical
        self.crf = crf
        self.ngram = ngram
        self.emb_layer = None
        self.radical_layer = None
        self.gram_layers = []
        self.font = font
        self.pic_size = pic_size
        self.batch_size = batch_size
        self.l_rate = None
        self.decay = None
        self.train_step = None
        self.saver = None
        self.decode_holders = None
        self.scores = None
        self.params = None
        self.pixels = None
        self.metric = metric
        self.updates = []
        self.bucket_dit = {}
        self.input_v = []
        self.input_w = []
        self.input_p = None
        self.output = []
        self.output_ = []
        self.output_p = []
        self.output_w = []
        self.output_w_ = []
        if self.crf > 0:
            self.transition_char = []
            for i in range(len(self.nums_tags)):
                self.transition_char.append(tf.get_variable('transitions_char' + str(i), [self.nums_tags[i] + 1, self.nums_tags[i] + 1]))

        self.all_metrics = None

        self.all_metrics = ['Precision', 'Recall', 'F1-score', 'True-Negative-Rate', 'Boundary-F1-score']

        while len(self.buckets_char) > len(self.counts):
            self.counts.append(1)

        self.real_batches = toolbox.get_real_batch(self.counts, self.batch_size)

    def main_graph(self, trained_model, scope, emb_dim, gru, rnn_dim, rnn_num, drop_out=0.5, rad_dim=30, emb=None, ng_embs=None, pixels=None, con_width=None, filters=None, pooling_size=None):
        if trained_model is not None:
            param_dic = {}
            param_dic['nums_chars'] = self.nums_chars
            param_dic['nums_tags'] = self.nums_tags
            param_dic['tag_scheme'] = self.tag_scheme
            param_dic['graphic'] = self.graphic
            param_dic['pic_size'] = self.pic_size
            param_dic['word_vec'] = self.word_vec
            param_dic['radical'] = self.radical
            param_dic['crf'] = self.crf
            param_dic['emb_dim'] = emb_dim
            param_dic['gru'] = gru
            param_dic['rnn_dim'] = rnn_dim
            param_dic['rnn_num'] = rnn_num
            param_dic['drop_out'] = drop_out
            param_dic['filter_size'] = con_width
            param_dic['filters'] = filters
            param_dic['pooling_size'] = pooling_size
            param_dic['font'] = self.font
            param_dic['buckets_char'] = self.buckets_char
            param_dic['ngram'] = self.ngram
            #print param_dic
            if self.metric == 'All':
                pindex = trained_model.rindex('/') + 1
                for m in self.all_metrics:
                    f_model = open(trained_model[:pindex] + m + '_' + trained_model[pindex:], 'w')
                    pickle.dump(param_dic, f_model)
                    f_model.close()
            else:
                f_model = open(trained_model, 'w')
                pickle.dump(param_dic, f_model)
                f_model.close()

        # define shared weights and variables

        dr = tf.placeholder(tf.float32, [], name='drop_out_holder')
        self.drop_out = dr
        self.drop_out_v = drop_out

        if self.word_vec:
            self.emb_layer = EmbeddingLayer(self.nums_chars + 500, emb_dim, weights=emb, name='emb_layer')

        if self.radical:
            self.radical_layer = EmbeddingLayer(216, rad_dim, name='radical_layer')

        if self.ngram is not None:
            if ng_embs is not None:
                assert len(ng_embs) == len(self.ngram)
            else:
                ng_embs = [None for _ in range(len(self.ngram))]
            for i, n_gram in enumerate(self.ngram):
                self.gram_layers.append(EmbeddingLayer(n_gram + 1000 * (i + 2), emb_dim, weights=ng_embs[i], name= str(i + 2) + 'gram_layer'))

        wrapper_conv_1, wrapper_mp_1, wrapper_conv_2, wrapper_mp_2, wrapper_dense, wrapper_dr = None, None, None, None, None, None

        if self.graphic:
            self.input_p = []
            assert pixels is not None and filters is not None and pooling_size is not None and con_width is not None

            self.pixels = pixels
            pixel_dim = int(math.sqrt(len(pixels[0])))

            wrapper_conv_1 = TimeDistributed(Convolution(con_width, 1, filters, name='conv_1'), name='wrapper_c1')
            wrapper_mp_1 = TimeDistributed(Maxpooling(pooling_size, pooling_size, name='pooling_1'), name='wrapper_p1')

            p_size_1 = toolbox.down_pool(pixel_dim, pooling_size)

            wrapper_conv_2 = TimeDistributed(Convolution(con_width, filters, filters, name='conv_2'), name='wrapper_c2')
            wrapper_mp_2 = TimeDistributed(Maxpooling(pooling_size, pooling_size, name='pooling_2'), name='wrapper_p2')

            p_size_2 = toolbox.down_pool(p_size_1, pooling_size)

            wrapper_dense = TimeDistributed(HiddenLayer(p_size_2 * p_size_2 * filters, 100, activation='tanh', name='conv_dense'), name='wrapper_3')
            wrapper_dr = TimeDistributed(DropoutLayer(self.drop_out), name='wrapper_dr')

        with tf.variable_scope('BiRNN'):

            if gru:
                fw_rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
                bw_rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
            else:
                fw_rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_dim, state_is_tuple=True)
                bw_rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_dim, state_is_tuple=True)

            if rnn_num > 1:
                fw_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([fw_rnn_cell]*rnn_num, state_is_tuple=True)
                bw_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([bw_rnn_cell]*rnn_num, state_is_tuple=True)

        output_wrapper = TimeDistributed(HiddenLayer(rnn_dim * 2, self.nums_tags[0], activation='linear', name='hidden'), name='wrapper')

        #define model for each bucket
        for idx, bucket in enumerate(self.buckets_char):
            if idx == 1:
                scope.reuse_variables()
            t1 = time()

            input_v = tf.placeholder(tf.int32, [None, bucket], name='input_' + str(bucket))

            self.input_v.append([input_v])

            emb_set = []

            if self.word_vec:
                word_out = self.emb_layer(input_v)
                emb_set.append(word_out)

            if self.radical:
                input_r = tf.placeholder(tf.int32, [None, bucket], name='input_r' + str(bucket))

                self.input_v[-1].append(input_r)
                radical_out = self.radical_layer(input_r)
                emb_set.append(radical_out)

            if self.ngram is not None:
                for i in range(len(self.ngram)):
                    input_g = tf.placeholder(tf.int32, [None, bucket], name='input_g' + str(i) + str(bucket))
                    self.input_v[-1].append(input_g)
                    gram_out = self.gram_layers[i](input_g)
                    emb_set.append(gram_out)

            if self.graphic:
                input_p = tf.placeholder(tf.float32, [None, bucket, pixel_dim*pixel_dim])
                self.input_p.append(input_p)

                pix_out = tf.reshape(input_p, [-1, bucket, pixel_dim, pixel_dim, 1])

                conv_out_1 = wrapper_conv_1(pix_out)
                pooling_out_1 = wrapper_mp_1(conv_out_1)

                conv_out_2 = wrapper_conv_2(pooling_out_1)
                pooling_out_2 = wrapper_mp_2(conv_out_2)

                assert p_size_2 == pooling_out_2[0].get_shape().as_list()[1]
                pooling_out = tf.reshape(pooling_out_2, [-1, bucket, p_size_2 * p_size_2 * filters])
                pooling_out = tf.unstack(pooling_out, axis=1)

                graphic_out = wrapper_dense(pooling_out)
                graphic_out = wrapper_dr(graphic_out)

                emb_set.append(graphic_out)


            if len(emb_set) > 1:
                emb_out = tf.concat(axis=2, values=emb_set)

            else:
                emb_out = emb_set[0]

            rnn_out = BiLSTM(rnn_dim, fw_cell=fw_rnn_cell, bw_cell=bw_rnn_cell, p=dr, name='BiLSTM' + str(bucket), scope='BiRNN')(emb_out, input_v)

            output = output_wrapper(rnn_out)

            self.output.append([output])

            self.output_.append([tf.placeholder(tf.int32, [None, bucket], name='tags' + str(bucket))])

            self.bucket_dit[bucket] = idx

            print 'Bucket %d, %f seconds' % (idx + 1, time() - t1)

        assert len(self.input_v) == len(self.output) and len(self.output) == len(self.output_) and len(self.output) == len(self.counts)

        self.params = tf.trainable_variables()

        self.saver = tf.train.Saver()

    def config(self, optimizer, decay, lr_v=None, momentum=None, clipping=False, max_gradient_norm=5.0):

        self.decay = decay
        print 'Training preparation...'

        print 'Defining loss...'
        loss = []
        if self.crf > 0:
            loss_function = losses.crf_loss
            for i in range(len(self.input_v)):
                bucket_loss = losses.loss_wrapper(self.output[i], self.output_[i], loss_function, transitions=self.transition_char, nums_tags=self.nums_tags, batch_size=self.real_batches[i])
                loss.append(bucket_loss)
        else:
            loss_function = losses.sparse_cross_entropy
            for output, output_ in zip(self.output, self.output_):
                bucket_loss = losses.loss_wrapper(output, output_, loss_function)
                loss.append(bucket_loss)

        l_rate = tf.placeholder(tf.float32, [], name='learning_rate_holder')
        self.l_rate = l_rate

        if optimizer == 'sgd':
            if momentum is None:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate=l_rate, momentum=momentum)
        elif optimizer == 'adagrad':
            assert lr_v is not None
            optimizer = tf.train.AdagradOptimizer(learning_rate=l_rate)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        else:
            raise Exception('optimiser error')

        self.train_step = []

        print 'Computing gradients...'

        for idx, l in enumerate(loss):
            t2 = time()
            if clipping:
                gradients = tf.gradients(l, self.params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                train_step = optimizer.apply_gradients(zip(clipped_gradients, self.params))
            else:
                train_step = optimizer.minimize(l)
            print 'Bucket %d, %f seconds' % (idx + 1, time() - t2)
            self.train_step.append(train_step)

    def decode_graph(self):
        self.decode_holders = []
        self.scores = []
        for bucket in self.buckets_char:
            decode_holders = []
            scores = []
            for nt in self.nums_tags:
                ob = tf.placeholder(tf.float32, [None, bucket, nt])
                trans = tf.placeholder(tf.float32, [nt + 1, nt + 1])
                nums_steps = ob.get_shape().as_list()[1]
                length = tf.placeholder(tf.int32, [None])
                b_size = tf.placeholder(tf.int32, [])
                small = -1000
                class_pad = tf.stack(small * tf.ones([b_size, nums_steps, 1]))
                observations = tf.concat(axis=2, values=[ob, class_pad])
                b_vec = tf.tile(([small] * nt + [0]), [b_size])
                b_vec = tf.cast(b_vec, tf.float32)
                b_vec = tf.reshape(b_vec, [b_size, 1, -1])
                e_vec = tf.tile(([0] + [small] * nt), [b_size])
                e_vec = tf.cast(e_vec, tf.float32)
                e_vec = tf.reshape(e_vec, [b_size, 1, -1])
                observations = tf.concat(axis=1, values=[b_vec, observations, e_vec])
                transitions = tf.reshape(tf.tile(trans, [b_size, 1]), [b_size, nt + 1, nt + 1])
                observations = tf.reshape(observations, [-1, nums_steps + 2, nt + 1, 1])
                observations = tf.transpose(observations, [1, 0, 2, 3])
                previous = observations[0, :, :, :]
                max_scores = []
                max_scores_pre = []
                alphas = [previous]
                for t in xrange(1, nums_steps + 2):
                    previous = tf.reshape(previous, [-1, nt + 1, 1])
                    current = tf.reshape(observations[t, :, :, :], [-1, 1, nt + 1])
                    alpha_t = previous + current + transitions
                    max_scores.append(tf.reduce_max(alpha_t, axis=1))
                    max_scores_pre.append(tf.argmax(alpha_t, axis=1))
                    alpha_t = tf.reshape(Forward.log_sum_exp(alpha_t, axis=1), [-1, nt + 1, 1])
                    alphas.append(alpha_t)
                    previous = alpha_t
                max_scores = tf.stack(max_scores, axis=1)
                max_scores_pre = tf.stack(max_scores_pre, axis=1)
                decode_holders.append([ob, trans, length, b_size])
                scores.append((max_scores, max_scores_pre))
            self.decode_holders.append(decode_holders)
            self.scores.append(scores)

    def train(self, t_x, t_y, v_x, v_y, idx2tag, idx2char, sess, epochs, trained_model, lr=0.05, decay=0.05, decay_step=1, tag_num=1):
        lr_r = lr

        best_epoch, best_score, best_seg, best_pos, c_tag, c_seg, c_score = {}, {}, {}, {}, {}, {}, {}

        pindex = 0

        metric = self.metric

        for m in self.all_metrics:
            best_epoch[m] = 0
            best_score[m] = 0

            best_seg[m] = 0
            best_pos[m] = 0

            c_tag[m] = 0
            c_seg[m] = 0
            c_score[m] = 0

        v_y = toolbox.merge_bucket(v_y)
        v_y = toolbox.unpad_zeros(v_y)

        gold = toolbox.decode_tags(v_y, idx2tag, self.tag_scheme)
        input_chars = toolbox.merge_bucket([v_x[0]])

        chars = toolbox.decode_chars(input_chars[0], idx2char)

        gold_out = toolbox.generate_output(chars, gold, self.tag_scheme)

        for epoch in range(epochs):
            print 'epoch: %d' % (epoch + 1)
            t = time()
            if epoch % decay_step == 0 and decay > 0:
                lr_r = lr/(1 + decay*(epoch/decay_step))

            data_list = t_x + t_y

            samples = zip(*data_list)

            random.shuffle(samples)

            for sample in samples:
                c_len = len(sample[0][0])
                idx = self.bucket_dit[c_len]
                real_batch_size = self.real_batches[idx]
                model = self.input_v[idx] + self.output_[idx]
                pt_holder = None
                if self.graphic:
                    pt_holder = self.input_p[idx]
                Batch.train(sess=sess[0], model=model, batch_size=real_batch_size, config=self.train_step[idx], lr=self.l_rate, lrv=lr_r, dr=self.drop_out, drv=self.drop_out_v, data=list(sample), pt_h=pt_holder, pixels=self.pixels, verbose=False)

            predictions = []

            for v_b_x in zip(*v_x):
                c_len = len(v_b_x[0][0])
                idx = self.bucket_dit[c_len]
                pt_holder = None
                if self.graphic:
                    pt_holder = self.input_p[idx]
                b_prediction = self.predict(data=v_b_x, sess=sess, model=self.input_v[idx] + self.output[idx], index=idx, pt_h=pt_holder, pt=self.pixels, batch_size=100)
                b_prediction = toolbox.decode_tags(b_prediction, idx2tag, self.tag_scheme)
                predictions.append(b_prediction)

            predictions = zip(*predictions)
            predictions = toolbox.merge_bucket(predictions)

            prediction_out = toolbox.generate_output(chars, predictions, self.tag_scheme)

            scores = toolbox.evaluator(prediction_out, gold_out, metric=metric, verbose=True, tag_num=tag_num)
            scores = np.asarray(scores)

            #Score_seg * Score_seg&tag
            c_seg['Precision'] = scores[0]
            c_seg['Recall'] = scores[1]
            c_seg['F1-score'] = scores[2]
            c_seg['True-Negative-Rate'] = scores[6]
            c_seg['Boundary-F1-score'] = scores[10]
            if self.tag_scheme != 'seg':
                c_tag['Precision'] = scores[3]
                c_tag['Recall'] = scores[4]
                c_tag['F1-score'] = scores[5]
                c_tag['True-Negative-Rate'] = scores[7]
                c_tag['Boundary-F1-score'] = scores[13]
            else:
                c_tag['Precision'] = 1
                c_tag['Recall'] = 1
                c_tag['F1-score'] = 1
                c_tag['True-Negative-Rate'] = 1
                c_tag['Boundary-F1-score'] = 1

            if metric == 'All':
                for m in self.all_metrics:
                    print 'Segmentation ' + m + ': %f' % c_seg[m]
                    print 'POS Tagging ' + m + ': %f\n' % c_tag[m]
                pindex = trained_model.rindex('/') + 1
            else:
                print 'Segmentation ' + metric + ': %f' % c_seg[metric]
                if self.tag_scheme != 'seg':
                    print 'POS Tagging ' + metric + ': %f\n' % c_tag[metric]

            for m in self.all_metrics:
                c_score[m] = c_seg[m] * c_tag[m]

            if metric == 'All':
                for m in self.all_metrics:
                    if c_score[m] > best_score[m] and epoch > 4:
                        best_epoch[m] = epoch + 1
                        best_score[m] = c_score[m]
                        best_seg[m] = c_seg[m]
                        best_pos[m] = c_tag[m]
                        self.saver.save(sess[0],  trained_model[:pindex] + m + '_' + trained_model[pindex:], write_meta_graph=False)

            elif c_score[metric] > best_score[metric] and epoch > 4:
                best_epoch[metric] = epoch + 1
                best_score[metric] = c_score[metric]
                best_seg[metric] = c_seg[metric]
                best_pos[metric] = c_tag[metric]
                self.saver.save(sess[0], trained_model, write_meta_graph=False)
            print 'Time consumed: %d seconds' % int(time() - t)
        print 'Training is finished!'

        if metric == 'All':
            for m in self.all_metrics:
                print 'Best segmentation ' + m + ': %f' % best_seg[m]
                print 'Best POS Tagging ' + m + ': %f' % best_pos[m]
                print 'Best epoch: %d\n' % best_epoch[m]
        else:
            print 'Best segmentation ' + metric + ': %f' % best_seg[metric]
            print 'Best POS Tagging ' + metric + ': %f' % best_pos[metric]
            print 'Best epoch: %d\n' % best_epoch[metric]

    def define_updates(self, new_chars, emb_path, char2idx, new_grams=None, ng_emb_path=None, gram2idx=None):

        self.nums_chars += len(new_chars)

        if self.word_vec and emb_path is not None:

            old_emb_weights = self.emb_layer.embeddings
            emb_dim = old_emb_weights.get_shape().as_list()[1]
            emb_len = old_emb_weights.get_shape().as_list()[0]
            new_emb = toolbox.get_new_embeddings(new_chars, emb_dim, emb_path)
            n_emb_sh = new_emb.get_shape().as_list()
            if len(n_emb_sh) > 1:
                new_emb_weights = tf.concat(axis=0, values=[old_emb_weights[:len(char2idx) - len(new_chars)], new_emb, old_emb_weights[len(char2idx):]])
                if new_emb_weights.get_shape().as_list()[0] > emb_len:
                    new_emb_weights = new_emb_weights[:emb_len]
                assign_op = old_emb_weights.assign(new_emb_weights)
                self.updates.append(assign_op)

        if self.ngram is not None and ng_emb_path is not None:
            old_gram_weights = [ng_layer.embeddings for ng_layer in self.gram_layers]
            ng_emb_dim = old_gram_weights[0].get_shape().as_list()[1]
            new_ng_emb = toolbox.get_new_ng_embeddings(new_grams, ng_emb_dim, ng_emb_path)
            for i in range(len(old_gram_weights)):
                new_ng_weight = tf.concat(axis=0, values=[old_gram_weights[i][:len(gram2idx[i]) - len(new_grams[i])], new_ng_emb[i], old_gram_weights[i][len(gram2idx[i]):]])
                assign_op = old_gram_weights[i].assign(new_ng_weight)
                self.updates.append(assign_op)

    def run_updates(self, sess, weight_path):
        self.saver.restore(sess, weight_path)
        for op in self.updates:
            sess.run(op)

        print 'Loaded.'

    def test(self, sess, t_x, t_y, idx2tag, idx2char, outpath=None, ensemble=None, batch_size=200, tag_num=1):

        t_y = toolbox.unpad_zeros(t_y)
        gold = toolbox.decode_tags(t_y, idx2tag, self.tag_scheme)
        chars = toolbox.decode_chars(t_x[0], idx2char)
        gold_out = toolbox.generate_output(chars, gold, self.tag_scheme)

        pt_holder = None
        if self.graphic:
            pt_holder = self.input_p[0]

        prediction = self.predict(data=t_x, sess=sess, model=self.input_v[0] + self.output[0], index=0, pt_h=pt_holder, pt=self.pixels, ensemble=ensemble, batch_size=batch_size)
        prediction = toolbox.decode_tags(prediction, idx2tag, self.tag_scheme)
        prediction_out = toolbox.generate_output(chars, prediction, self.tag_scheme)

        scores = toolbox.evaluator(prediction_out, gold_out, metric='All', verbose=True, tag_num=tag_num)

        print 'Best scores: '

        print 'Segmentation F1-score: %f' % scores[2]
        print 'Segmentation Precision: %f' % scores[0]
        print 'Segmentation Recall: %f' % scores[1]
        print 'Segmentation True Negative Rate: %f' % scores[6]
        print 'Segmentation Boundary-F1-score: %f\n' % scores[10]

        print 'Joint POS tagging F-score: %f' % scores[5]
        print 'Joint POS tagging Precision: %f' % scores[3]
        print 'Joint POS tagging Recall: %f' % scores[4]
        print 'Joint POS True Negative Rate: %f' % scores[7]
        print 'Joint POS tagging Boundary-F1-score: %f\n' % scores[13]

        if outpath is not None:
            final_out = prediction_out[0]
            toolbox.printer(final_out, outpath)

    def tag(self, sess, r_x, idx2tag, idx2char, char2idx, outpath='out.txt', ensemble=None, batch_size=200, large_file=False):

        chars = toolbox.decode_chars(r_x[0], idx2char)

        char_num = len(set(char2idx.values()))

        r_x = np.asarray(r_x)

        r_x[0][r_x[0] > char_num - 1] = char2idx['<UNK>']

        pt_holder = None
        if self.graphic:
            pt_holder = self.input_p[0]

        c_len = len(r_x[0][0])
        idx = self.bucket_dit[c_len]

        real_batch = batch_size * 300 / c_len

        prediction = self.predict(data=r_x, sess=sess, model=self.input_v[idx] + self.output[idx], index=idx, pt_h=pt_holder, pt=self.pixels, ensemble=ensemble, batch_size=real_batch)
        prediction = toolbox.decode_tags(prediction, idx2tag, self.tag_scheme)
        prediction_out = toolbox.generate_output(chars, prediction, self.tag_scheme)

        final_out = prediction_out[0]
        if large_file:
            return final_out
        else:
            toolbox.printer(final_out, outpath)

    def predict(self, data, sess, model, index=None, argmax=True, batch_size=100, pt_h=None, pt=None, ensemble=None, verbose=False):
        if self.crf:
            assert index is not None
            predictions = Batch.predict(sess=sess[0], decode_sess=sess[1], model=model, transitions=self.transition_char, crf=self.crf, scores=self.scores[index], decode_holders=self.decode_holders[index], argmax=argmax, batch_size=batch_size, data=data, dr=self.drop_out, pixels=pt, pt_h=pt_h, ensemble=ensemble, verbose=verbose)
        else:
            predictions = Batch.predict(sess=sess[0], model=model, crf=self.crf, argmax=argmax, batch_size=batch_size, data=data, dr=self.drop_out, ensemble=ensemble, verbose=verbose)
        return predictions





