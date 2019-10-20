import os
import re
import sys
import time

import numpy as np
import tensorflow as tf

from models.model_utils.output_projection import output_projection_layer
from models.model_utils import decoder_fn

import model_utils.utils as utils
from models.model_utils.seq2seq import dynamic_rnn_decoder
from model_utils.utils import gaussian_kld
from model_utils.utils import get_bi_rnn_encode
from model_utils.utils import get_bow
from model_utils.utils import get_rnn_encode
from model_utils.utils import norm_log_liklihood
from model_utils.utils import sample_gaussian

from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer

from base_model import BaseModel


class Model(BaseModel):

	def __init__(self, sess, config, api, log_dir, forward, scope=None):

		self.vocab = api.vocab
		self.rev_vocab = api.rev_vocab
		self.vocab_size = len(self.vocab)

		self.topic_vocab = api.topic_vocab
		self.topic_vocab_size = len(self.topic_vocab)

		self.da_vocab = api.dialog_act_vocab
		self.da_vocab_size = len(self.da_vocab)

		self.sess = sess
		self.scope = scope

		self.pad_id = self.rev_vocab["<pad>"]
		self.sos_id = self.rev_vocab["<s>"]
		self.eos_id = self.rev_vocab["</s>"]
		self.unk_id = self.rev_vocab["<unk>"]

		self.context_cell_size = config.cxt_cell_size
		self.sent_cell_size = config.sent_cell_size
		self.dec_cell_size = config.dec_cell_size
		self.latent_size = config.latent_size


		with tf.name_scope("io"):
			
			self.input_contexts = tf.placeholder(dtype=tf.string, shape=(None, None, None), name="dialog_context")
			self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="context_lens")
			self.topics = tf.placeholder(dtype=tf.int32, shape=(None,), name="topics")

			self.output_tokens = tf.placeholder(dtype=tf.string, shape=(None, None, None), name="output_token")
			self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_lens")

			self.learning_rate = tf.Variable(float(config.init_lr), trainable=False, name="learning_rate")
			self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, config.lr_decay))
			self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
			self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")


		batch_size = tf.shape(self.input_contexts)[0]
		max_dialog_len = tf.shape(self.input_contexts)[1]
		max_out_len = tf.shape(self.output_tokens)[2]


		with tf.variable_scope("tokenization"):
			self.symbols = tf.Variable(self.vocab, trainable=False, name="symbols")
			self.symbol2index = HashTable(KeyValueTensorInitializer(self.symbols, tf.Variable(np.array([i for i in range(self.vocab_size)], dtype=np.int32), False)), default_value=self.unk_id, name="symbol2index")
			
			self.contexts = self.symbol2index.lookup(self.input_contexts)
			self.responses_target = self.symbol2index.lookup(self.output_tokens)


		with tf.variable_scope("topic_embedding"):
			t_embedding = tf.get_variable("embedding", [self.topic_vocab_size, config.topic_embed_size], dtype=tf.float32)
			topic_embedding = tf.nn.embedding_lookup(t_embedding, self.topics)
			# [batch_size, topic_embed_size]


		with tf.variable_scope("word_embedding"):
			self.embedding = tf.get_variable("embedding", [self.vocab_size, config.embed_size], dtype=tf.float32)
			embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32, shape=[self.vocab_size, 1])
			embedding = self.embedding * embedding_mask

			input_embedding = tf.nn.embedding_lookup(embedding, tf.reshape(self.contexts, [-1]))
			input_embedding = tf.reshape(input_embedding, [batch_size*max_dialog_len, -1, config.embed_size])
			output_embedding = tf.nn.embedding_lookup(embedding, tf.reshape(self.responses_target, [-1]))
			output_embedding = tf.reshape(output_embedding, [batch_size*max_dialog_len, -1, config.embed_size])


		with tf.variable_scope("uttrance_encoder"):

			if config.sent_type == "rnn":
				sent_cell = self.create_rnn_cell(self.sent_cell_size)
				input_embedding, sent_size = get_rnn_encode(input_embedding, sent_cell, scope="sent_rnn")
				output_embedding, _ = get_rnn_encode(output_embedding, sent_cell, tf.reshape(self.output_lens, [-1]), scope="sent_rnn", reuse=True)

			elif config.sent_type == "bi_rnn":
				fwd_sent_cell = self.create_rnn_cell(self.sent_cell_size)
				bwd_sent_cell = self.create_rnn_cell(self.sent_cell_size)
				input_embedding, sent_size = get_bi_rnn_encode(input_embedding, fwd_sent_cell, bwd_sent_cell, scope="sent_bi_rnn")
				output_embedding, _ = get_bi_rnn_encode(output_embedding, fwd_sent_cell, bwd_sent_cell, tf.reshape(self.output_lens, [-1]), scope="sent_bi_rnn", reuse=True)
			else:
				raise ValueError("Unknown sent_type. Must be one of [rnn, bi_rnn]")


			input_embedding = tf.reshape(input_embedding, [batch_size, max_dialog_len, sent_size])
			if config.keep_prob < 1.0:
				input_embedding = tf.nn.dropout(input_embedding, config.keep_prob)

			output_embedding = tf.reshape(output_embedding, [batch_size, max_dialog_len, sent_size])


		with tf.variable_scope("context_encoder"):

			enc_cell = self.create_rnn_cell(self.context_cell_size)

			cxt_outputs,_ = tf.nn.dynamic_rnn(
				enc_cell,
				input_embedding,
				dtype=tf.float32,
				sequence_length=self.context_lens)
			# [batch_size, max_dialog_len, context_cell_size]



		tile_topic_embedding=tf.reshape(tf.tile(topic_embedding,[1, max_dialog_len]), [batch_size, max_dialog_len, config.topic_embed_size])
		cond_embedding = tf.concat([tile_topic_embedding, cxt_outputs], -1)
		# [batch_size, max_dialog_len, context_cell_size + topic_embed_size]


		with tf.variable_scope("posterior_network"):
			recog_input = tf.concat([cond_embedding, output_embedding], -1)
			post_sample, recog_mu_1, recog_logvar_1, recog_mu_2, recog_logvar_2 = self.hierarchical_inference_net(recog_input)

		with tf.variable_scope("prior_network"):
			prior_input = cond_embedding
			prior_sample, prior_mu_1, prior_logvar_1, prior_mu_2, prior_logvar_2 = self.hierarchical_inference_net(prior_input)

		latent_sample = tf.cond(self.use_prior,
			lambda: prior_sample,
			lambda: post_sample)		

		with tf.variable_scope("decoder"):

			dec_inputs = tf.concat([cond_embedding, latent_sample], -1)
			dec_inputs_dim = config.latent_size + config.topic_embed_size + self.context_cell_size
			dec_inputs = tf.reshape(dec_inputs, [batch_size*max_dialog_len, dec_inputs_dim])

			dec_init_state = tf.contrib.layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None, scope="init_state")
			dec_cell = self.create_rnn_cell(self.dec_cell_size)

			output_fn, sampled_sequence_loss = output_projection_layer(self.dec_cell_size, self.vocab_size)
			decoder_fn_train = decoder_fn.simple_decoder_fn_train(dec_init_state, dec_inputs)
			decoder_fn_inference = decoder_fn.simple_decoder_fn_inference(
				output_fn, 
				dec_init_state, 
				dec_inputs, 
				embedding,
				self.sos_id, self.eos_id, max_out_len*2, self.vocab_size)

			if forward:
				dec_outs, _, final_context_state = dynamic_rnn_decoder(
					dec_cell, 
					decoder_fn_inference,
					scope="decoder")
			else:
				dec_input_embedding = tf.nn.embedding_lookup(embedding, tf.reshape(self.responses_target, [-1]))
				dec_input_embedding = tf.reshape(dec_input_embedding, [batch_size*max_dialog_len, -1, config.embed_size])
				dec_input_embedding = dec_input_embedding[:, 0:-1, :]
				dec_seq_lens = tf.reshape(self.output_lens, [-1]) - 1

				if config.dec_keep_prob < 1.0:
					keep_mask = tf.less_equal(
						tf.random_uniform((batch_size*max_dialog_len, max_out_len-1), minval=0.0, maxval=1.0), config.dec_keep_prob)
					keep_mask = tf.expand_dims(tf.to_float(keep_mask), 2)
					dec_input_embedding = dec_input_embedding * keep_mask
					dec_input_embedding = tf.reshape(dec_input_embedding, [-1, max_out_len-1, config.embed_size])

				dec_outs, _, final_context_state = dynamic_rnn_decoder(
					dec_cell, 
					decoder_fn_train,
					dec_input_embedding, 
					dec_seq_lens, 
					scope="decoder")


				reshape_target = tf.reshape(self.responses_target, [batch_size*max_dialog_len, -1])
				labels = reshape_target[:, 1:]
				label_mask = tf.to_float(tf.sign(labels))
				local_loss = sampled_sequence_loss(
					dec_outs,
					labels, 
					label_mask)


			if final_context_state is not None:
				final_context_state = final_context_state[:, 0:tf.shape(dec_outs)[1]]
				mask = tf.to_int32(tf.sign(tf.reduce_max(dec_outs, axis=2)))
				dec_out_words = tf.multiply(tf.reverse(final_context_state, axis=[1]), mask)
			else:
				dec_out_words = tf.argmax(dec_outs, 2)

			self.dec_out_words = tf.reshape(dec_out_words, [batch_size, max_dialog_len, -1])[:,-1,:]


		if not forward:
			with tf.variable_scope("loss"):

				self.avg_rc_loss = tf.reduce_mean(local_loss)
				self.rc_ppl = tf.reduce_sum(local_loss)
				self.total_word = tf.reduce_sum(label_mask)


				new_recog_mu_2 = tf.reshape(recog_mu_2, [-1, config.latent_size])
				new_recog_logvar_2 = tf.reshape(recog_logvar_2, [-1, config.latent_size])
				new_prior_mu_1 = tf.reshape(prior_mu_1, [-1, config.latent_size])
				new_prior_logvar_1 = tf.reshape(prior_logvar_1, [-1, config.latent_size])
				new_recog_mu_1 = tf.reshape(recog_mu_1, [-1, config.latent_size])
				new_recog_logvar_1 = tf.reshape(recog_logvar_1, [-1, config.latent_size])
				new_prior_mu_2 = tf.reshape(prior_mu_2, [-1, config.latent_size])
				new_prior_logvar_2 = tf.reshape(prior_logvar_2, [-1, config.latent_size])

				kld_1 = gaussian_kld(new_recog_mu_2, new_recog_logvar_2, new_prior_mu_1, new_prior_logvar_1)
				kld_2 = gaussian_kld(new_recog_mu_1, new_recog_logvar_1, new_prior_mu_2, new_prior_logvar_2)
				kld = kld_1 + kld_2

				self.avg_kld = tf.reduce_mean(kld)
				if log_dir is not None:
					self.kl_w = tf.minimum(tf.to_float(self.global_t)/config.full_kl_step, 1.0)
				else:
					self.kl_w = tf.constant(1.0)


				aug_elbo = self.elbo = self.avg_rc_loss + self.kl_w * self.avg_kld


				tf.summary.scalar("rc_loss", self.avg_rc_loss)
				tf.summary.scalar("elbo", self.elbo)
				tf.summary.scalar("kld", self.avg_kld)
				self.summary_op = tf.summary.merge_all()

				"""
				self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
				self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
				self.est_marginal = tf.reduce_mean(- self.log_p_z + self.log_q_z_xy)
				"""

			self.optimize(sess, config, aug_elbo, log_dir)

		self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)


	def batch_2_feed(self, batch, global_t, use_prior, repeat=1):

		context, context_lens, floors, topics, my_profiles, ot_profiles, outputs, output_lens, output_das = batch
		
		feed_dict = {self.input_contexts: context, 
					 self.context_lens:context_lens,
					 self.topics:topics,
					 self.output_tokens: outputs,
					 self.output_lens: output_lens,
					 self.use_prior: use_prior}

		if repeat > 1:
			tiled_feed_dict = {}
			for key, val in feed_dict.items():
				if key is self.use_prior:
					tiled_feed_dict[key] = val
					continue
				multipliers = [1]*len(val.shape)
				multipliers[0] = repeat
				tiled_feed_dict[key] = np.tile(val, multipliers)
			feed_dict = tiled_feed_dict

		if global_t is not None:
			feed_dict[self.global_t] = global_t

		return feed_dict


	def train(self, global_t, sess, train_feed, update_limit=5000):
		elbo_losses = []
		rc_losses = []

		rc_ppls = []
		total_words = []

		kl_losses = []
		local_t = 0
		start_time = time.time()
		loss_names =  ["elbo_loss", "rc_loss", "kl_loss"]
		while True:
			batch = train_feed.next_new_batch()
			if batch is None:
				break
			if update_limit is not None and local_t >= update_limit:
				break
			feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)
			_, sum_op, elbo_loss, rc_loss, rc_ppl, kl_loss, total_word = sess.run([self.train_ops, self.summary_op,
																		 self.elbo, 
																		 self.avg_rc_loss, self.rc_ppl, self.avg_kld, self.total_word],
																		 feed_dict)
			self.train_summary_writer.add_summary(sum_op, global_t)

			total_words.append(total_word)
			elbo_losses.append(elbo_loss)
			rc_ppls.append(rc_ppl)
			rc_losses.append(rc_loss)
			kl_losses.append(kl_loss)

			global_t += 1
			local_t += 1
			if local_t % (train_feed.num_batch / 20) == 0:
				kl_w = sess.run(self.kl_w, {self.global_t: global_t})
				self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
								loss_names, [elbo_losses, rc_losses, kl_losses], "kl_w %f, perplexity: %f" % 
								(kl_w, np.exp(np.sum(rc_ppls)/np.sum(total_words))))

		# finish epoch!
		epoch_time = time.time() - start_time
		avg_losses = self.print_loss("Epoch Done", loss_names,
									 [elbo_losses, rc_losses, kl_losses],
									 "step time %.4f, perplexity: %f" % 
									 (epoch_time / train_feed.num_batch, np.exp(np.sum(rc_ppls)/np.sum(total_words))))

		return global_t, avg_losses[0]




	def valid(self, name, sess, valid_feed):
		elbo_losses = []
		rc_losses = []
		rc_ppls = []
		kl_losses = []
		total_words = []

		while True:
			batch = valid_feed.next_new_batch()
			if batch is None:
				break
			feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)

			elbo_loss, rc_loss, rc_ppl, kl_loss, total_word = sess.run(
				[self.elbo, self.avg_rc_loss,
				 self.rc_ppl, self.avg_kld, self.total_word], feed_dict)
			
			total_words.append(total_word)
			elbo_losses.append(elbo_loss)
			rc_losses.append(rc_loss)
			rc_ppls.append(rc_ppl)
			kl_losses.append(kl_loss)

		avg_losses = self.print_loss(name, ["elbo_loss", "rc_loss", "kl_loss"],
									 [elbo_losses, rc_losses, kl_losses], "perplexity: %f" % np.exp(np.sum(rc_ppls)/np.sum(total_words)))
		return avg_losses[0]


	def test(self, sess, test_feed, num_batch=None, repeat=5, dest=sys.stdout):

		local_t = 0
		recall_bleus = []
		prec_bleus = []

		while True:
			batch = test_feed.next_new_batch()
			if batch is None or (num_batch is not None and local_t > num_batch):
				break
			feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=repeat)
			word_outs = sess.run(self.dec_out_words, feed_dict)
			
			sample_words = np.split(word_outs, repeat, axis=0)
			
			true_srcs = feed_dict[self.input_contexts]
			true_src_lens = feed_dict[self.context_lens]
			true_outs = feed_dict[self.output_tokens][:,-1,:]
			true_topics = feed_dict[self.topics]
			local_t += 1

			if dest != sys.stdout:
				if local_t % (test_feed.num_batch / 10) == 0:
					print("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch))),

			for b_id in range(test_feed.batch_size):
				dest.write("Batch %d index %d of topic %s\n" % (local_t, b_id, self.topic_vocab[true_topics[b_id]]))

				start = np.maximum(0, true_src_lens[b_id]-5)
				for t_id in range(start, true_srcs.shape[1], 1):
					src_str = " ".join([w for w in true_srcs[b_id, t_id].tolist() if w not in ["<pad>"]])
					dest.write("Src %d: %s\n" % (t_id, src_str))
				

				true_tokens = [w for w in true_outs[b_id].tolist() if w not in ["<pad>", "<s>", "</s>"]]
				true_str = " ".join(true_tokens).replace(" ' ", "'")
				dest.write("Target >> %s\n" % (true_str))


				local_tokens = []
				for r_id in range(repeat):
					pred_outs = sample_words[r_id]
					# pred_da = np.argmax(sample_das[r_id], axis=1)[0]
					pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e not in [self.eos_id, self.pad_id, self.sos_id]]
					pred_str = " ".join(pred_tokens).replace(" ' ", "'")
					dest.write("Sample %d >> %s\n" % (r_id, pred_str))
					local_tokens.append(pred_tokens)


				max_bleu, avg_bleu = utils.get_bleu_stats(true_tokens, local_tokens)
				recall_bleus.append(max_bleu)
				prec_bleus.append(avg_bleu)
				dest.write("\n")

		avg_recall_bleu = float(np.mean(recall_bleus))
		avg_prec_bleu = float(np.mean(prec_bleus))
		avg_f1 = 2*(avg_prec_bleu*avg_recall_bleu) / (avg_prec_bleu+avg_recall_bleu+10e-12)
		report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
				 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
		print report
		dest.write(report + "\n")
		print("Done testing")



	def hierarchical_inference_net(self, inputs):

		num_group = 2
		group_dim = int(self.latent_size/2)

		recog_mulogvar_1 = tf.contrib.layers.fully_connected(inputs, group_dim * 2, activation_fn=None, scope="muvar")
		recog_mu_1, recog_logvar_1 = tf.split(recog_mulogvar_1, 2, axis=-1)		
		z_post_1 = sample_gaussian(recog_mu_1, recog_logvar_1)


		cont_inputs = tf.concat([z_post_1, inputs], -1)
		recog_mulogvar_2 = tf.contrib.layers.fully_connected(cont_inputs, group_dim * 2, activation_fn=None, scope="muvar1")
		recog_mu_2, recog_logvar_2 = tf.split(recog_mulogvar_2, 2, axis=-1)
		z_post_2 = sample_gaussian(recog_mu_2, recog_logvar_2)

		z_post = tf.concat([z_post_1, z_post_2], -1)

		return z_post, recog_mu_1, recog_logvar_1, recog_mu_2, recog_logvar_2