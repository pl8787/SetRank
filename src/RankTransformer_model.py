from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import numpy as np

from six.moves import xrange# pylint: disable=redefined-builtin
import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import xrange# pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import copy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from modules import *

DEBUG = False

# TODO(ebrevdo): Remove once _linear is fully deprecated.
#linear = rnn_cell_impl._linear  # pylint: disable=protected-access
linear = core_rnn_cell._linear

class RankTransformer(object):
    def __init__(self, rank_list_size, embed_size, expand_embed_size, batch_size,
                 hparam_str, forward_only=False, feed_previous = False):
        """Create the model.

        """
        self.hparams = tf.contrib.training.HParams(
            learning_rate=0.001,                 # Learning rate.
            learning_rate_decay_factor=1.0, # Learning rate decays by this much.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            reverse_input=True,                # Set to True for reverse input sequences.
            loss_func='attrank',            # Select Loss function
            l2_loss=0.0,                    # Set strength for L2 regularization.
            use_residua=False,                # Set to True for using the initial scores to compute residua.
            softRank_theta=0.1,                # Set Gaussian distribution theta for softRank.
            num_blocks=6,
            dropout_rate=0.1,
            hidden_units=256,
            num_heads=8,
            empty_bound=-10.0,
            multi_abstract=False,
            activation='relu',
            use_mask=False,
            gaussian_kernel=0,
            spline=0,
            num_induced=0,
            weight_temper=1.0,
            pos_embed=False,
            pos_embed_multi=1,
        )
        self.hparams.parse(hparam_str)

        print(self.hparams)

        print_op_list_show = []
        print_op_list = []
        def desc_tensor(t, p=print_op_list):
            if t.dtype == tf.bool:
                p.append(
                    tf.print('#'*20,
                        '\n\tName:', t.name,
                        '\n\tShape:', tf.shape(t),
                    '\n', '#'*20, '\n')
                )
            else:
                p.append(
                    tf.print('#'*20,
                        '\n\tName:', t.name, str(t.dtype),
                        '\n\tShape:', tf.shape(t),
                        '\n\tMean:', tf.reduce_mean(tf.abs(t)),
                        '\n\tMin/Max:', tf.reduce_min(t), tf.reduce_max(t),
                    '\n', '#'*20, '\n')
                )
            return

        hp = self.hparams
        is_training = ~forward_only

        self.start_index = 0
        self.count = 1
        self.rank_list_size = rank_list_size
        self.embed_size = embed_size
        self.expand_embed_size = expand_embed_size if expand_embed_size > 0 else 0
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * self.hparams.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        
        if hp.activation == 'relu':
            hp.activation = tf.nn.relu
        elif hp.activation == 'elu':
            hp.activation = tf.nn.elu
        else:
            hp.activation = tf.nn.relu

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        
        # Feeds for inputs.
        self.encoder_inputs = []
        self.embeddings = tf.placeholder(tf.float32, shape=[None, embed_size], name="embeddings")
        self.target_labels = []
        self.target_weights = []
        self.target_initial_score = []
        for i in xrange(self.rank_list_size):
            self.encoder_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                            name="encoder{0}".format(i)))
        for i in xrange(self.rank_list_size):
            self.target_labels.append(tf.placeholder(tf.int64, shape=[None],
                                        name="targets{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                            name="weight{0}".format(i)))
            self.target_initial_score.append(tf.placeholder(tf.float32, shape=[None],
                                            name="initial_score{0}".format(i)))

        if hp.pos_embed:

            self.pos_embeddings = tf.Variable(
                    np.random.normal(size=[self.rank_list_size, hp.hidden_units],
                    loc=0,
                    scale=0.1),
                dtype=np.float32,
                name='pos_embeddings')
            pos_r = tf.contrib.framework.argsort(tf.stack(self.target_initial_score, axis=1))
            pos = tf.map_fn(tf.math.invert_permutation, pos_r)
            print_op_list.append(tf.print('INIT_SCORE', self.target_initial_score, summarize=-1))
            print_op_list.append(tf.print('POS', pos[0], summarize=-1))
            self.pos_embed = embedding_ops.embedding_lookup(self.pos_embeddings, pos)

            #m = 10
            #pos_smoother_np = np.float32(np.tile(np.arange(1.1, 0.1, -1.0 / self.rank_list_size), (m, 1))).T
            #for i in range(m-1):
            #    pos_smoother_np[i+1] *= pos_smoother_np[i]
            #self.pos_smoother = tf.constant(pos_smoother_np)
            #self.pos_embeddings_var = tf.Variable(
            #        np.random.normal(size=[m, hp.hidden_units],
            #        loc=0,
            #        scale=0.1),
            #    dtype=np.float32,
            #    name='pos_embeddings')
            #self.pos_embeddings = tf.matmul(self.pos_smoother, self.pos_embeddings_var)
            #pos_r = tf.contrib.framework.argsort(tf.stack(self.target_initial_score, axis=1))
            #pos = tf.map_fn(tf.math.invert_permutation, pos_r)
            #print_op_list.append(tf.print('INIT_SCORE', self.target_initial_score, summarize=-1))
            #print_op_list.append(tf.print('POS', pos[0], summarize=-1))
            #self.pos_embed = embedding_ops.embedding_lookup(self.pos_embeddings, pos)

            #pos = tf.stack(self.target_initial_score, axis=1)
            #self.pos_embed = positional_encoding(pos, hp.hidden_units, zero_pad=False)[:,::-1,:]

            #max_rank_list_size = 1000
            #self.pos_embeddings = tf.Variable(
            #        np.random.normal(size=[max_rank_list_size, hp.hidden_units],
            #        loc=0,
            #        scale=0.1),
            #    dtype=np.float32,
            #    name='pos_embeddings')
            #pos_r = tf.contrib.framework.argsort(tf.stack(self.target_initial_score, axis=1))
            #pos = tf.map_fn(tf.math.invert_permutation, pos_r)
            #start_pos = tf.random.uniform(
            #    shape = [tf.shape(pos)[0], 1],
            #    minval = 0,
            #    maxval = max_rank_list_size-self.rank_list_size,
            #    dtype = tf.int32)
            #pos = start_pos + pos
            #print_op_list.append(tf.print('INIT_SCORE', self.target_initial_score, summarize=-1))
            #print_op_list.append(tf.print('POS', pos[0], summarize=-1))
            #self.pos_embed = embedding_ops.embedding_lookup(self.pos_embeddings, pos)

            desc_tensor(self.pos_embed)

        self.encoder_inputs_tensor = tf.stack(self.encoder_inputs, axis=1)
        desc_tensor(self.encoder_inputs_tensor)

        self.batch_index_bias = tf.placeholder(tf.int64, shape=[None])
        self.batch_expansion_mat = tf.placeholder(tf.float32, shape=[None,1])
        self.batch_diag = tf.placeholder(tf.float32, shape=[None,self.rank_list_size,self.rank_list_size])
        self.GO_embed = tf.get_variable("GO_embed", [1,self.embed_size + expand_embed_size],dtype=tf.float32)
        self.PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)

        # The ID of GO Embedding, tf.shape(self.embeddings)[-1]+2
        #self.decoder_inputs = tf.concat((tf.ones_like(self.target_labels[:, :1])*(tf.shape(self.embeddings)[-1]+2), self.target_labels[:, :-1]), axis=-1)

        # Create the Transformer
        self.inputs_mask = tf.equal(self.encoder_inputs_tensor, tf.cast(tf.shape(self.embeddings)[0], tf.int64))
        desc_tensor(self.inputs_mask)
        self.enc_mask = 1.0 - tf.cast(self.inputs_mask, tf.float32)
        desc_tensor(self.enc_mask)

        print_op_list_show.append(tf.print('input', self.encoder_inputs_tensor, summarize=-1))
        embeddings_pad = tf.concat(axis=0, values=[self.embeddings, self.PAD_embed])
        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding_ops.embedding_lookup(embeddings_pad, self.encoder_inputs_tensor)
            desc_tensor(self.enc)
            if hp.gaussian_kernel != 0:
                self.mu = tf.Variable(
                    np.random.uniform(size=[1, 1, self.embed_size, hp.gaussian_kernel],
                        low=-1.0,
                        #low=0.0,
                        high=1.0),
                    dtype=tf.float32,
                    name='gaussian_mu')
                self.sigma = tf.Variable(
                    np.random.uniform(size=[1, 1, self.embed_size, hp.gaussian_kernel],
                        low=0.01,
                        high=1.0),
                    dtype=tf.float32,
                    constraint=lambda x: tf.clip_by_value(x, 0.01, 1.0),
                    name='gaussian_sigma')
                self.enc = tf.expand_dims(self.enc, axis=-1)
                self.enc = tf.exp(-(self.enc-self.mu)**2/0.01)
                desc_tensor(self.enc)
                
                batch_size = tf.shape(self.enc)[0]
                self.enc = tf.reshape(self.enc, [batch_size, self.rank_list_size, self.embed_size * hp.gaussian_kernel])
            elif hp.spline != 0:
                self.spbias = tf.Variable(
                    tf.random_uniform(shape=[1, 1, self.embed_size, hp.spline],
                        #minval=-1.0,
                        minval=0.0,
                        maxval=1.0,
                        dtype=tf.float32),
                    name='spline')
                self.enc = tf.expand_dims(self.enc, axis=-1)
                self.enc = tf.nn.relu(self.enc+self.spbias)
                desc_tensor(self.enc)
                
                batch_size = tf.shape(self.enc)[0]
                self.enc = tf.reshape(self.enc, [batch_size, self.rank_list_size, self.embed_size * hp.spline])

            if hp.multi_abstract:
                self.enc = feedforward_multi(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units], activation=hp.activation)
                desc_tensor(self.enc)
            else:
                self.enc = tf.layers.dense(self.enc, hp.hidden_units, use_bias=True, activation=hp.activation)
                desc_tensor(self.enc)

            ## Dropout
            self.enc = tf.layers.dropout(self.enc, 
                                        rate=hp.dropout_rate, 
                                        training=tf.convert_to_tensor(is_training))
            
            desc_tensor(self.enc)
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    print_op_list.append(tf.print('S1', self.enc[0][0]))
                    
                    if hp.pos_embed and i % hp.pos_embed_multi == 0:
                        print('[Info] Add positional embedding.', i)
                        with tf.variable_scope("pos_embed"):
                            self.enc += self.pos_embed
                            desc_tensor(self.enc)

                    if hp.num_induced == 0 and hp.use_mask:
                        print('[Info] Use Masked Transformer.', i)
                        self.enc = multihead_attention_mask(queries=self.enc, 
                                                    keys=self.enc, 
                                                    query_masks=self.enc_mask,
                                                    key_masks=self.enc_mask,
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False,
                                                    activation=hp.activation)
                    elif hp.num_induced == 0 and not hp.use_mask:
                        print('[Info] Use Transformer.')
                        self.enc, print_ops = multihead_attention(queries=self.enc, 
                                                    keys=self.enc, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False,
                                                    activation=hp.activation)
                        #print_op_list += print_ops
                    elif hp.num_induced > 0 and hp.use_mask:
                        print('[Info] Use Induced Masked Transformer.')
                        self.enc = induced_multihead_attention_mask(queries=self.enc, 
                                                    keys=self.enc, 
                                                    query_masks=self.enc_mask,
                                                    key_masks=self.enc_mask,
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    num_induced=hp.num_induced,
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training)
                    elif hp.num_induced > 0 and not hp.use_mask:
                        print('[Info] Use Induced Transformer.')
                        self.enc, print_ops = induced_multihead_attention(queries=self.enc, 
                                                    keys=self.enc, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    num_induced=hp.num_induced,
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training)
                        print_op_list_show += print_ops
                    else:
                        print('[ERROR] hp.num_induced is negative.')
                        exit()
                    print_op_list.append(tf.print('S2', self.enc[0][0]))
                    
                    desc_tensor(self.enc)
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units], activation=hp.activation)
                    desc_tensor(self.enc)
                    print_op_list.append(tf.print('S3', self.enc[0][0]))

        #self.enc = tf.layers.dense(self.enc, 128, use_bias=True, activation=tf.nn.relu)
        #desc_tensor(self.enc)
        self.enc = tf.layers.dense(self.enc, 1, use_bias=True)
        desc_tensor(self.enc)
        
        self.outputs = tf.squeeze(self.enc, axis=-1)
        desc_tensor(self.outputs)

        print_op_list.append(tf.print('Data', self.encoder_inputs_tensor[0]))
        print_op_list.append(tf.print('Mask', self.inputs_mask[0]))
        self.outputs = tf.where(self.inputs_mask, tf.tile(tf.constant([[np.float32(hp.empty_bound)]]), tf.shape(self.inputs_mask)), self.outputs)
        desc_tensor(self.outputs)

        self.outputs = [self.outputs]

        #residua learning
        if self.hparams.use_residua:
            self.outputs[0] = self.outputs[0] + tf.stack(self.target_initial_score, axis=1)

        # Training outputs and losses.
        if not DEBUG:
            print_op_list = []
            print_op_list_show = []
        with tf.control_dependencies(print_op_list + print_op_list_show):
            print('Loss Function is ' + self.hparams.loss_func)
            self.loss = None
            if self.hparams.loss_func == 'attrank':
                self.loss = self.attrank_loss(self.outputs[0], self.target_labels, self.target_weights)
            elif self.hparams.loss_func == 'listMLE':
                self.loss = self.listMLE(self.outputs[0], self.target_labels, self.target_weights)
            elif self.hparams.loss_func == 'softRank':
                self.loss = self.softRank(self.outputs[0], self.target_labels, self.target_weights)


        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
        if not forward_only:
            #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.gradients = tf.gradients(self.loss, params)
            self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                     self.hparams.max_gradient_norm)
            self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                             global_step=self.global_step)
            tf.summary.scalar('train_loss', tf.reduce_mean(self.loss))
            tf.summary.scalar('learning_rate', self.learning_rate)
        else:
            tf.summary.scalar('valid_loss', tf.reduce_mean(self.loss))

        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())



    def step(self, session, encoder_inputs, embeddings, target_labels,
             target_weights, target_initial_score, forward_only):
        """Run a step of the model feeding the given inputs.

        """
        # Check if the sizes match.
        if len(encoder_inputs) != self.rank_list_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), self.rank_list_size))
        if len(target_labels) != self.rank_list_size:
            raise ValueError("Labels length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_labels), self.rank_list_size))
        if len(target_weights) != self.rank_list_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), self.rank_list_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.batch_index_bias.name] = np.array([i * self.rank_list_size for i in xrange(self.batch_size)])
        input_feed[self.batch_expansion_mat.name] = np.ones((self.batch_size,1))
        input_feed[self.batch_diag.name] = np.array([np.diag([0.5 for x in xrange(self.rank_list_size)]) for _ in xrange(self.batch_size)])
        input_feed[self.embeddings.name] = np.array(embeddings)
        for l in xrange(self.rank_list_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(self.rank_list_size):
            input_feed[self.target_weights[l].name] = target_weights[l]
            input_feed[self.target_labels[l].name] = target_labels[l]
            input_feed[self.target_initial_score[l].name] = target_initial_score[l]


        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates,    # Update Op that does SGD.
                            self.norm,    # Gradient norm.
                            self.loss,    # Loss for this batch.
                            self.outputs,
                            self.summary # Summarize statistics.
                            ]    
        else:
            output_feed = [self.loss, # Loss for this batch.
                        self.summary # Summarize statistics.
            ]    
            output_feed += self.outputs


        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            print('loss=', outputs[2])
        if DEBUG:
            input()
        if not forward_only:
            return outputs[1], outputs[2], outputs[3], outputs[-1]    # Gradient norm, loss, no outputs, summary.
        else:
            return None, outputs[0], outputs[2:], outputs[1]    # No gradient norm, loss, outputs, summary.

    def prepare_data_with_index(self, input_seq, output_seq, output_weights, output_initial_score,
                                features, index, encoder_inputs, decoder_targets,
                                embeddings, decoder_weights, decoder_initial_scores):
        alpha = 1.0
        i = index
        base = len(embeddings)
        for x in input_seq[i]:
            if x >= 0:
                embeddings.append(features[x])
        decoder_targets.append([x if output_seq[i][x] < 0 else output_seq[i][x] for x in xrange(len(output_seq[i]))])
        #decoder_weights.append([output_weights[i][x] for x in xrange(len(input_seq[i]))])
        if self.hparams.reverse_input:
            encoder_inputs.append(list(reversed([-1 if input_seq[i][x] < 0 else base+x for x in xrange(len(input_seq[i]))])))
            decoder_weights.append(list(reversed([0 if input_seq[i][x] < 0 else output_weights[i][x] for x in xrange(len(input_seq[i]))])))
            #decoder_initial_scores.append(list(reversed([0 if input_seq[i][x] < 0 else output_initial_score[i][x] for x in xrange(len(input_seq[i]))])))
            decoder_initial_scores.append(list(reversed([-10 if input_seq[i][x] < 0 else output_initial_score[i][x] for x in xrange(len(input_seq[i]))])))
            if self.hparams.loss_func == 'attrank':
                weight_sum = 0
                for w in xrange(len(decoder_weights[-1])):
                    decoder_weights[-1][w] = math.exp(decoder_weights[-1][w]*self.hparams.weight_temper) if decoder_weights[-1][w] > 0 else 0
                    weight_sum += decoder_weights[-1][w]
                if weight_sum > 0:
                    for w in xrange(len(decoder_weights[-1])):
                        decoder_weights[-1][w] /= weight_sum
            for j in xrange(len(decoder_targets[-1])):
                decoder_targets[-1][j] = self.rank_list_size - 1 - decoder_targets[-1][j]
        else:
            encoder_input = []
            decoder_weight = []
            decoder_initial_score = []
            tmp = 0
            for x in xrange(len(input_seq[i])):
                if input_seq[i][x] < 0:
                    encoder_input.append(-1)
                    decoder_weight.append(-1)
                    tmp += 1
                else:
                    encoder_input.append(base+x-tmp)
                    decoder_weight.append(output_weights[i][x-tmp])
                    decoder_initial_score.append(output_initial_score[i][x-tmp])
            #encoder_inputs.append([-1 if input_seq[i][x] < 0 else base+x for x in xrange(len(input_seq[i]))])
            encoder_inputs.append(encoder_input)
            #decoder_weights.append([-1 if input_seq[i][x] < 0 else output_weights[i][x] for x in xrange(len(input_seq[i]))])
            decoder_weights.append(decoder_weight)
            decoder_initial_scores.append(decoder_initial_score)
            count = 0
            for x in encoder_inputs[-1]:
                if x < 0:
                    count += 1
            for j in xrange(len(decoder_targets[-1])):
                index = count + decoder_targets[-1][j]
                if index < self.rank_list_size:
                    decoder_targets[-1][j] = index
                else:
                    decoder_targets[-1][j] = index - self.rank_list_size
        for x in xrange(len(decoder_weights[-1])):
            decoder_weights[-1][x] *= alpha

    def get_batch(self, input_seq, output_seq, output_weights, output_initial_score, features):
        """Get a random batch of data from the specified bucket, prepare for step.

        """

        if len(input_seq[0]) != self.rank_list_size:
            raise ValueError("Input ranklist length must be equal to the one in bucket,"
                             " %d != %d." % (len(input_seq[0]), self.rank_list_size))
        length = len(input_seq)
        encoder_inputs, decoder_targets, embeddings, decoder_weights, decoder_initial_scores = [], [], [], [], []
        cache = None
        #if self.start_index + self.batch_size > len(input_seq):
        #    self.count += 1
        #    self.start_index = self.count
        for _ in xrange(self.batch_size):
            #i = self.start_index + _
            i = int(random.random() * length)
            self.prepare_data_with_index(input_seq, output_seq, output_weights, output_initial_score, features, i,
                                encoder_inputs, decoder_targets, embeddings, decoder_weights, decoder_initial_scores)

            if cache == None:
                cache = [input_seq[i], decoder_weights[-1]]

        #self.start_index += self.batch_size

        embedings_length = len(embeddings)
        for i in xrange(self.batch_size):
            for j in xrange(self.rank_list_size):
                if encoder_inputs[i][j] < 0:
                    encoder_inputs[i][j] = embedings_length


        batch_encoder_inputs = []
        batch_weights = []
        batch_targets = []
        batch_initial_scores = []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(self.rank_list_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))
        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(self.rank_list_size):
            batch_targets.append(
                np.array([decoder_targets[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            batch_weights.append(
                np.array([decoder_weights[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.float32))
            batch_initial_scores.append(
                np.array([decoder_initial_scores[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.float32))

        return batch_encoder_inputs, embeddings, batch_targets, batch_weights, batch_initial_scores, cache

    def get_next_batch(self, index, input_seq, output_seq, output_weights, output_initial_score, features):
        """Get the next batch of data from the specified bucket, prepare for step.

        """
        alpha = 1.0

        if len(input_seq[0]) != self.rank_list_size:
            raise ValueError("Input ranklist length must be equal to the one in bucket,"
                             " %d != %d." % (len(input_seq[0]), self.rank_list_size))
        length = len(input_seq)
        encoder_inputs, decoder_targets, embeddings, decoder_weights, decoder_initial_scores = [], [], [], [], []
        cache = None
        #if self.start_index + self.batch_size > len(input_seq):
        #    self.count += 1
        #    self.start_index = self.count
        for offset in xrange(self.batch_size):
            #randomly select a positive instance
            i = index + offset
            self.prepare_data_with_index(input_seq, output_seq, output_weights, output_initial_score, features, i,
                                encoder_inputs, decoder_targets, embeddings, decoder_weights, decoder_initial_scores)

            if cache == None:
                cache = [input_seq[i], decoder_weights[-1], decoder_initial_scores[-1]]

            #if self.hparams.loss_func == 'softRank':
            #    for x in xrange(len(decoder_weights[-1])):
            #        decoder_weights[-1][x] = 0 if decoder_weights[-1][x] < 0 else decoder_weights[-1][x]

        #self.start_index += self.batch_size

        embedings_length = len(embeddings)
        for i in xrange(self.batch_size):
            for j in xrange(self.rank_list_size):
                if encoder_inputs[i][j] < 0:
                    encoder_inputs[i][j] = embedings_length


        batch_encoder_inputs = []
        batch_weights = []
        batch_initial_scores = []
        batch_targets = []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(self.rank_list_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))
        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(self.rank_list_size):
            batch_targets.append(
                np.array([decoder_targets[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            batch_weights.append(
                np.array([decoder_weights[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.float32))
            batch_initial_scores.append(
                np.array([decoder_initial_scores[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.float32))


        return batch_encoder_inputs, embeddings, batch_targets, batch_weights, batch_initial_scores, cache


    def get_data_by_index(self, input_seq, output_seq, output_weights, output_initial_score, features, index): #not fixed
        """Get one data from the specified index, prepare for step.

        """
        if len(input_seq[0]) != self.rank_list_size:
            raise ValueError("Input ranklist length must be equal to the one in bucket,"
                             " %d != %d." % (len(input_seq[0]), self.rank_list_size))

        self.batch_size = 1
        length = len(input_seq)
        encoder_inputs, decoder_targets, embeddings, decoder_weights, decoder_initial_scores = [], [], [], [], []
        cache = None
        i = index
        self.prepare_data_with_index(input_seq, output_seq, output_weights, output_initial_score, features, i,
                                encoder_inputs, decoder_targets, embeddings, decoder_weights, decoder_initial_scores)

        embedings_length = len(embeddings)
        for i in xrange(self.batch_size):
            for j in xrange(self.rank_list_size):
                if encoder_inputs[i][j] < 0:
                    encoder_inputs[i][j] = embedings_length

        batch_encoder_inputs = []
        batch_weights = []
        batch_initial_scores = []
        batch_targets = []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(self.rank_list_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.float32))
        for length_idx in xrange(self.rank_list_size):
            batch_targets.append(
                np.array([decoder_targets[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            batch_weights.append(
                np.array([decoder_weights[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.float32))
            batch_initial_scores.append(
                np.array([decoder_initial_scores[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.float32))

        return batch_encoder_inputs, embeddings, batch_targets, batch_weights, batch_initial_scores

    def attrank_loss(self, output, target_indexs, target_rels, name=None):
        loss = 600
        with ops.name_scope(name, "attrank_loss",[output] + target_indexs + target_rels):
            target = tf.transpose(ops.convert_to_tensor(target_rels))
            #target = tf.nn.softmax(target)
            #target = target / tf.reduce_sum(target,1,keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
        batch_size = tf.shape(target_rels[0])[0]
        return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)

    def pairwise_loss(self, output, target_indexs, target_rels, name=None):
        loss = 0
        batch_size = tf.shape(target_rels[0])[0]
        with ops.name_scope(name, "pairwise_loss",[output] + target_indexs + target_rels):
            for i in xrange(batch_size):
                for j1 in xrange(self.rank_list_size):
                    for j2 in xrange(self.rank_list_size):
                        if output[i][j1] > output[i][j2] and target_rels[i][j1] < target_rels[i][j2]:
                            loss += target_rels[i][j2] - target_rels[i][j1]
        return loss

        
        return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)

    def listMLE(self, output, target_indexs, target_rels, name=None):
        loss = None
        with ops.name_scope(name, "listMLE",[output] + target_indexs + target_rels):
            output = tf.nn.l2_normalize(output, 1)
            loss = -1.0 * math_ops.reduce_sum(output,1)
            print(loss.get_shape())
            exp_output = tf.exp(output)
            exp_output_table = tf.reshape(exp_output,[-1])
            print(exp_output.get_shape())
            print(exp_output_table.get_shape())
            sum_exp_output = math_ops.reduce_sum(exp_output,1)
            loss = tf.add(loss, tf.log(sum_exp_output))
            #compute MLE
            for i in xrange(self.rank_list_size-1):
                idx = target_indexs[i] + tf.to_int64(self.batch_index_bias)
                y_i = embedding_ops.embedding_lookup(exp_output_table, idx)
                #y_i = tf.gather_nd(exp_output, idx)
                sum_exp_output = tf.subtract(sum_exp_output, y_i)
                loss = tf.add(loss, tf.log(sum_exp_output))
        batch_size = tf.shape(target_rels[0])[0]
        return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)

    def softRank(self, output, target_indexs, target_rels, name=None):
        loss = None
        batch_size = tf.shape(target_rels[0])[0]
        theta = 0.1
        with ops.name_scope(name, "softRank",[output] + target_indexs + target_rels):
            output = tf.nn.l2_normalize(output, 1)
            #compute pi_i_j
            tmp = tf.concat(axis=1, values=[self.batch_expansion_mat for _ in xrange(self.rank_list_size)])
            tmp_expand = tf.expand_dims(tmp, -2)
            output_expand = tf.expand_dims(output, -2)
            dif = tf.subtract(tf.matmul(tf.matrix_transpose(output_expand), tmp_expand),
                            tf.matmul(tf.matrix_transpose(tmp_expand), output_expand))
            #unpacked_pi = self.integral_Guaussian(dif, theta)
            unpacked_pi = tf.add(self.integral_Guaussian(dif, self.hparams.softRank_theta), self.batch_diag) #make diag equal to 1.0
            #may need to unpack pi: pi_i_j is the probability that i is bigger than j
            pi = tf.unstack(unpacked_pi, None, 1)
            for i in xrange(self.rank_list_size):
                pi[i] = tf.unstack(pi[i], None, 1)
            #compute rank distribution p_j_r
            one_zeros = tf.matmul(self.batch_expansion_mat, 
                        tf.constant([1.0]+[0.0 for r in xrange(self.rank_list_size-1)], tf.float32, [1,self.rank_list_size]))
            #initial_value = tf.unpack(one_zeros, None, 1)
            pr = [one_zeros for _ in xrange(self.rank_list_size)] #[i][r][None]
            #debug_pr_1 = [one_zeros for _ in xrange(self.rank_list_size)] #[i][r][None]
            for i in xrange(self.rank_list_size):
                for j in xrange(self.rank_list_size):
                    #if i != j: #insert doc j
                    pr_1 = tf.pad(tf.stack(tf.unstack(pr[i], None, 1)[:-1],1), [[0,0],[1,0]], mode='CONSTANT')
                    #debug_pr_1[i] = pr_1
                        #pr_1 = tf.concat(1, [self.batch_expansion_mat*0.0, tf.unpack(pr[i], None, 1)[:-1]])
                    factor = tf.tile(tf.expand_dims(pi[i][j], -1),[1,self.rank_list_size])
                        #print(factor.get_shape())
                    pr[i] = tf.add(tf.multiply(pr[i], factor),
                                    tf.multiply(pr_1, 1.0 - factor))
                        #for r in reversed(xrange(self.rank_list_size)):
                            #if r < 1:
                            #    pr[i][r] = tf.mul(pr[i][r], pi[i][j])
                            #else:
                            #    pr[i][r] = tf.add(tf.mul(pr[i][r], pi[i][j]),
                            #            tf.mul(pr[i][r-1], 1.0 - pi[i][j]))

            #compute expected NDCG
            #compute Gmax
            Dr = tf.matmul(self.batch_expansion_mat, 
                    tf.constant([1.0/math.log(2.0+r) for r in xrange(self.rank_list_size)], tf.float32, [1,self.rank_list_size]))
            gmaxs = []
            for i in xrange(self.rank_list_size):
                idx = target_indexs[i] + tf.to_int64(self.batch_index_bias)
                g = embedding_ops.embedding_lookup(target_rels, idx)
                gmaxs.append(g)
            _gmax = tf.exp(tf.stack(gmaxs, 1)) * (1.0 / math.log(2))
            Gmax = tf.reduce_sum(tf.multiply(Dr, _gmax), 1)
            #compute E(Dr)
            Edrs = []
            for i in xrange(self.rank_list_size):
                edr = tf.multiply(Dr, pr[i])
                Edrs.append(tf.reduce_sum(edr,1))
            #compute g(j)
            g = tf.exp(tf.stack(target_rels, 1)) * (1.0 / math.log(2))
            dcg = tf.multiply(g, tf.stack(Edrs, 1))
            Edcg = tf.reduce_sum(dcg, 1)
            Ndcg = tf.div(Edcg, Gmax)
            #compute loss
            loss = (Ndcg * -1.0 + 1) * 10
        return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)#, pi, pr, Ndcg]

    def integral_Guaussian(self, mu, theta):
        a = -4.0/math.sqrt(2.0*math.pi)/theta
        exp_mu = tf.exp(a * mu)
        ig = tf.div(exp_mu, exp_mu + 1) * -1.0 + 1
        return ig

    def clip_by_each_value(self, t_list, clip_max_value = None, clip_min_value = None, name=None):
        if (not isinstance(t_list, collections.Sequence)
            or isinstance(t_list, six.string_types)):
            raise TypeError("t_list should be a sequence")
        t_list = list(t_list)

        with ops.name_scope(name, "clip_by_each_value",t_list + [clip_norm]) as name:
            values = [
                    ops.convert_to_tensor(
                            t.values if isinstance(t, ops.IndexedSlices) else t,
                            name="t_%d" % i)
                    if t is not None else t
                    for i, t in enumerate(t_list)]

            values_clipped = []
            for i, v in enumerate(values):
                if v is None:
                    values_clipped.append(None)
                else:
                    t = None
                    if clip_value_max != None:
                        t = math_ops.minimum(v, clip_value_max)
                    if clip_value_min != None:
                        t = math_ops.maximum(t, clip_value_min, name=name)
                    with ops.colocate_with(t):
                        values_clipped.append(
                                tf.identity(t, name="%s_%d" % (name, i)))

            list_clipped = [
                    ops.IndexedSlices(c_v, t.indices, t.dense_shape)
                    if isinstance(t, ops.IndexedSlices)
                    else c_v
                    for (c_v, t) in zip(values_clipped, t_list)]

        return list_clipped


