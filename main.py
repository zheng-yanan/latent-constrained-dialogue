import os
import time
import numpy as np
import tensorflow as tf
from beeprint import pp
from data_apis.corpus import SWDADialogCorpus, DailyDialogCorpus
from data_apis.data_utils import SWDADataLoader, DailyDataLoader
from models.new_model import Model as Model

# constants
tf.app.flags.DEFINE_bool("word2vec", True, "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("dataset", "dailydialog", "[dailydialog, swda]")
tf.app.flags.DEFINE_string("work_dir", "checkpoints", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "", "the dir to load checkpoint for forward only")
FLAGS = tf.app.flags.FLAGS


class Config(object):

    update_limit = 3000 
    sent_type = "rnn" # rnn or bi_rnn
    latent_size = 200  
    full_kl_step = 10000  
    dec_keep_prob = 1.0  # word dropout rate
    cell_type = "gru"  # gru or lstm
    embed_size = 200 
    topic_embed_size = 30 
    cxt_cell_size = 512  
    sent_cell_size = 256  
    dec_cell_size = 256 
    backward_size = 10
    step_size = 1
    max_utt_len = 40
    num_layer = 1
    op = "adam"
    grad_clip = 5.0 
    init_w = 0.08
    batch_size = 30
    init_lr = 0.001 
    lr_hold = 1 
    lr_decay = 0.6 
    keep_prob = 1.0 
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True
    max_epoch = 60



def main():

    # config for training
    config = Config()

    # config for validation
    valid_config = Config()
    valid_config.keep_prob = 1.0
    valid_config.dec_keep_prob = 1.0
    valid_config.batch_size = 60

    # configuration for testing
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = 1

    pp(config)


    if FLAGS.dataset == "dailydialog":
    	api = DailyDialogCorpus(word2vec=FLAGS.word2vec, word2vec_dim=config.embed_size)
    elif FLAGS.dataset == "swda":
   		api = SWDADialogCorpus(word2vec=FLAGS.word2vec, word2vec_dim=config.embed_size)

    dial_corpus = api.get_dialog_corpus()
    meta_corpus = api.get_meta_corpus()
    train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, valid_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")

    if FLAGS.dataset == "dailydialog":
	    train_feed = DailyDataLoader("Train", train_dial, train_meta, config)
	    valid_feed = DailyDataLoader("Valid", valid_dial, valid_meta, config)
	    test_feed = DailyDataLoader("Test", test_dial, test_meta, config)
	    config.update_limit = int(1978/2)

    elif FLAGS.dataset == "swda":    
	    train_feed = SWDADataLoader("Train", train_dial, train_meta, config)
	    valid_feed = SWDADataLoader("Valid", valid_dial, valid_meta, config)
	    test_feed = SWDADataLoader("Test", test_dial, test_meta, config)

    if FLAGS.forward_only or FLAGS.resume:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run_"+FLAGS.dataset+"_"+str(int(time.time())))

    # begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
        scope = "model"
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            model = Model(sess, config, api, log_dir=None if FLAGS.forward_only else log_dir, forward=False, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            valid_model = Model(sess, valid_config, api, log_dir=None, forward=False, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            test_model = Model(sess, test_config, api, log_dir=None, forward=True, scope=scope)

        print("Created computation graphs")
        if api.word2vec is not None and not FLAGS.forward_only:
            print("Loaded word2vec")
            sess.run(model.embedding.assign(np.array(api.word2vec)))

        # write config to a file for logging
        if not FLAGS.forward_only:
            with open(os.path.join(log_dir, "run.log"), "wb") as f:
                f.write(pp(config, output=False))

        # create a folder by force
        ckp_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        print("Created models with fresh parameters.")
        sess.run(tf.global_variables_initializer())


        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        if ckpt:
            print("Reading dm models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        model.symbol2index.init.run()
        valid_model.symbol2index.init.run()
        test_model.symbol2index.init.run()
        
        if not FLAGS.forward_only:
            dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ ".ckpt")
            global_t = 1
            patience = 10  # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            for epoch in range(config.max_epoch):
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

                # begin training
                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, config.backward_size,
                                          config.step_size, shuffle=True)
                global_t, train_loss = model.train(global_t, sess, train_feed, update_limit=config.update_limit)


                test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                     test_config.step_size, shuffle=True, intra_shuffle=False)
                test_model.test(sess, test_feed, num_batch=5)


                # begin validation
                valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                      valid_config.step_size, shuffle=False, intra_shuffle=False)
                valid_loss = valid_model.valid("ELBO_VALID", sess, valid_feed)


                done_epoch = epoch + 1
                # only save a models if the dev loss is smaller
                # Decrease learning rate if no improvement was seen over last 3 times.
                if config.op == "sgd" and done_epoch > config.lr_hold:
                    sess.run(model.learning_rate_decay_op)

                if valid_loss < best_dev_loss:
                    if valid_loss <= dev_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch * config.patient_increase)
                        dev_loss_threshold = valid_loss

                    # still save the best train model
                    if FLAGS.save_model:
                        print("Save model!!")
                        model.saver.save(sess, dm_checkpoint_path, global_step=epoch)
                    best_dev_loss = valid_loss

                if config.early_stop and patience <= done_epoch:
                    print("!!Early stop due to run out of patience!!")
                    break
            print("Best validation loss %f" % best_dev_loss)
            print("Done training")
        else:
            # begin validation
            # begin validation
            valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            valid_model.valid("ELBO_VALID", sess, valid_feed)

            test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            valid_model.valid("ELBO_TEST", sess, test_feed)

            dest_f = open(os.path.join(log_dir, "test.txt"), "wb")
            test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                 test_config.step_size, shuffle=False, intra_shuffle=False)
            test_model.test(sess, test_feed, num_batch=None, repeat=10, dest=dest_f)
            dest_f.close()

if __name__ == "__main__":
    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main()













