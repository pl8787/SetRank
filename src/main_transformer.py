from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
from tqdm import tqdm

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils

from RankTransformer_model import RankTransformer

tf.logging.set_verbosity(tf.logging.ERROR)


#rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "/tmp/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./tmp/", "Training directory.")
tf.app.flags.DEFINE_string("test_dir", "./tmp/", "Directory for output test results.")
tf.app.flags.DEFINE_string("hparams", "", "Hyper-parameters for models.")
tf.app.flags.DEFINE_integer("ckpt_step", -1, "Checkpoint of the global steps.")

tf.app.flags.DEFINE_integer("batch_size", 256,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("embed_size", 1024,
                            "Size of each model layer (hidden layer input size).")
tf.app.flags.DEFINE_integer("max_train_iteration", 0,
                            "Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("boost_max_num", 50,
                            "The max number of new data for boosting one training instance.")
tf.app.flags.DEFINE_integer("boost_swap_num", 10,
                            "How many time to swap when boosting one training instance.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for decoding data.")
tf.app.flags.DEFINE_boolean("decode_train", False,
                            "Set to True for decoding training data.")
tf.app.flags.DEFINE_boolean("feed_previous", False,
                            "Set to True for feed previous internal output for training.")
tf.app.flags.DEFINE_boolean("boost_training_data", False,
                            "Boost training data througn swapping docs with same relevance scores.")
tf.app.flags.DEFINE_integer("perm_pair_cnt", 0,
                            "Permutation Count for adding noise.")



FLAGS = tf.app.flags.FLAGS


def create_model(session, data_set, forward_only, ckpt_step=-1):
    """Create translation model and initialize or load parameters in session."""
    expand_embed_size = max(FLAGS.embed_size - data_set.embed_size, 0)
    model = RankTransformer(data_set.rank_list_size, data_set.embed_size,
                     expand_embed_size, FLAGS.batch_size, FLAGS.hparams,
                     forward_only, FLAGS.feed_previous)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        # model/train_transformer/RankTransformer.ckpt-600
        model_path = ckpt.model_checkpoint_path
        if ckpt_step != -1:
            model_path_seg = ckpt.model_checkpoint_path.split('/')
            model_prefix = '/'.join(model_path_seg[:-1]) + '/' + '-'.join(model_path_seg[-1].split('-')[:-1])
            model_path = '{}-{}'.format(model_prefix, ckpt_step)
        print("Reading model parameters from %s" % model_path)
        model.saver.restore(session, model_path)
        #print(model.mu)
        #print(session.run(model.mu))
        #exit()
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    # Prepare data.
    print("Reading data in %s" % FLAGS.data_dir)
    
    train_set = data_utils.read_data(FLAGS.data_dir, 'train', with_dummy=True)
    if FLAGS.boost_training_data:
        print('Boosting training data')
        train_set.boost_training_data(FLAGS.boost_max_num, FLAGS.boost_swap_num)
    valid_set = data_utils.read_data(FLAGS.data_dir, 'valid', with_dummy=True)
    print("Rank list size %d" % train_set.rank_list_size)

    print("Train dataset number:", len(train_set.initial_list))
    print("Valid dataset number:", len(valid_set.initial_list))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating model...")
        model = create_model(sess, train_set, False)

        # Create tensorboard summarizations.
        train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train_log',
                                        sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.train_dir + '/valid_log')

        #pad data
        train_set.pad(train_set.rank_list_size, model.hparams.reverse_input)
        valid_set.pad(valid_set.rank_list_size, model.hparams.reverse_input)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        best_perplexity = None
        while True:
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, embeddings, decoder_targets, target_weights, target_initial_scores, _ = model.get_batch(train_set.initial_list,
                                                                                train_set.gold_list, train_set.gold_weights,
                                                                                train_set.initial_scores, train_set.features)
            #print(np.array(embeddings).shape, np.mean(np.array(embeddings)))
            #input()
            #print(target_initial_scores)
            #input()
            _, step_loss, results, summary = model.step(sess, encoder_inputs, embeddings, decoder_targets,
                                        target_weights, target_initial_scores, False)
            train_writer.add_summary(summary, current_step)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            #output_label = results[0][0]
            #print('OUTPUT')
            #print(output_label)

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:

                print(model.start_index)
                train_writer.add_summary(summary, current_step)
                
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                             "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                                 step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                
                if perplexity == float('inf'):
                    break

                # Validate model
                it = 0
                count_batch = 0.0
                valid_loss = 0
                while it < len(valid_set.initial_list) - model.batch_size:
                    encoder_inputs, embeddings, decoder_targets, target_weights, target_initial_scores, cache = model.get_next_batch(it, valid_set.initial_list,
                                                                                valid_set.gold_list, valid_set.gold_weights,
                                                                                valid_set.initial_scores, valid_set.features)
                    _, v_loss, results, summary = model.step(sess, encoder_inputs, embeddings, decoder_targets,
                                                    target_weights, target_initial_scores, True)
                    it += model.batch_size
                    valid_loss += v_loss
                    #valid_writer.add_summary(summary, current_step+count_batch)
                    count_batch += 1.0
                valid_loss /= count_batch
                valid_loss_summary = tf.Summary()
                valid_loss_summary.value.add(tag='train_loss', simple_value=valid_loss)
                valid_writer.add_summary(valid_loss_summary, current_step)
                eval_ppx = math.exp(valid_loss) if valid_loss < 300 else float('inf')
                print("  eval: perplexity %.2f" % (eval_ppx))
                print("[Iter {}] valid loss = {}".format(current_step, valid_loss))

                # Save checkpoint and zero timer and loss.
                #if True: #best_perplexity == None or best_perplexity >= eval_ppx:
                if best_perplexity == None or best_perplexity >= eval_ppx:
                    best_perplexity = eval_ppx
                    checkpoint_path = os.path.join(FLAGS.train_dir, "RankTransformer.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                
                output_label = results[0][0]
                
                original_input = [encoder_inputs[l][0] for l in range(model.rank_list_size)]
                print('ENCODE INPUTS')
                print(original_input)
                print('ENCODE CONTENT')
                print(cache[0])
                print('DECODER WEIGHT')
                print(cache[1])
                print('DECODER INIT SCORE')
                print(cache[2])
                print('DECODER TARGET')
                gold_output = [decoder_targets[l][0] for l in range(model.rank_list_size)]
                print(gold_output)
                print('OUTPUT')
                print(output_label)

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

                if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                    break



def decode():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load test data.
        print("Reading data in %s" % FLAGS.data_dir)
        test_set = None
        if FLAGS.decode_train:
            test_set = data_utils.read_data(FLAGS.data_dir,'train', with_dummy=True)
        else:
            test_set = data_utils.read_data(FLAGS.data_dir,'test', with_dummy=True)

        # Create model and load parameters.
        model = create_model(sess, test_set, True, FLAGS.ckpt_step)
        model.batch_size = 1    # We decode one sentence at a time.

        test_set.pad(test_set.rank_list_size, model.hparams.reverse_input)

        rerank_scores = []

        perm_pair_cnt = FLAGS.perm_pair_cnt

        # Decode from test data.
        for i in tqdm(range(len(test_set.initial_list))):
            encoder_inputs, embeddings, decoder_targets, target_weights, target_initial_scores = model.get_data_by_index(test_set.initial_list,
                                                                                            test_set.gold_list,
                                                                                            test_set.gold_weights,
                                                                                            test_set.initial_scores,
                                                                                            test_set.features, i)
            #target_initial_scores_fake = [np.ones_like(x) for x in target_initial_scores]
            
            # Permutation noise
            enc_len = np.sum(np.int32((np.array(encoder_inputs) >= 0)))
            enc_num = range(enc_len)
            rule = []
            for _ in range(perm_pair_cnt):
                rule.append(random.sample(enc_num, 2))

            for r1, r2 in rule:
                # Add position noise
                # encoder_inputs[r1], encoder_inputs[r2] = encoder_inputs[r2], encoder_inputs[r1]
                # Add score noise
                target_initial_scores[r1], target_initial_scores[r2] = target_initial_scores[r2], target_initial_scores[r1]

            _, test_loss, output_logits, summary = model.step(sess, encoder_inputs, embeddings, decoder_targets,
                                            target_weights, target_initial_scores, True)

            output_logit = output_logits[0][0]

            # for r1, r2 in rule[::-1]:
                # Recover order
                # output_logit[r1], output_logit[r2] = output_logit[r2], output_logit[r1]

            _, test_loss, output_logits, summary = model.step(sess, encoder_inputs, embeddings, decoder_targets,
                                            target_weights, target_initial_scores, True)

            '''
            #print(test_loss)
            print(output_logits)
            #reorder = sorted(range(len(target_initial_scores)), key=lambda x: target_initial_scores[x], reverse=True)
            reorder = list(range(len(target_initial_scores)))#[::-1]
            reorder_sh = reorder[1:]
            #reorder[2], reorder[3] = reorder[3], reorder[2]
            #random.shuffle(reorder_sh)
            reorder_sh = reorder_sh[::-1]
            reorder[1:] = reorder_sh
            reorder[-1], reorder[0] = reorder[0], reorder[-1]
            embeddings = [embeddings[int(encoder_inputs[x])] for x in reorder[:-1]]
            target_initial_scores = [target_initial_scores[x] for x in reorder[::-1]]
            output_logits_reorder = [[[output_logits[0][0][x] for x in reorder[::-1]]]]
            _, test_loss, output_logits, summary = model.step(sess, encoder_inputs, embeddings, decoder_targets,
                                            target_weights, target_initial_scores, True)
            print(output_logits_reorder)
            print(output_logits)
            print(np.array(output_logits) - np.array(output_logits_reorder))
            #exit()
            #The output is a list of rerank index for decoder_inputs (which represents the gold rank list)
            from IPython import embed
            embed()
            if i == 0:
                exit()
            '''
            rerank_scores.append(output_logit)
            #if i % FLAGS.steps_per_checkpoint == 0:
            #    print("Decoding %.2f \r" % (float(i)/len(test_set.initial_list))),

        #get rerank indexes with new scores
        print('shuffle before sort')
        rerank_lists = []
        for i in range(len(rerank_scores)):
            scores = np.array(rerank_scores[i])
            #random.shuffle(scores)
            scores += np.random.uniform(low=-1e-5, high=1e-5, size=scores.shape)
            rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))

        if FLAGS.decode_train:
            data_utils.output_ranklist(test_set, rerank_lists, FLAGS.test_dir, model.hparams.reverse_input, 'train')
        else:
            data_utils.output_ranklist(test_set, rerank_lists, FLAGS.test_dir, model.hparams.reverse_input, 'test')

    return


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
