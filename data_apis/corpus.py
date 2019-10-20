import pickle as pkl
from collections import Counter
import numpy as np
import nltk





class DailyDialogCorpus(object):
	def __init__(self, corpus_path="data/dailydialog/dailydialog_split.pkl", 
		max_vocab_cnt=30000, word2vec=True, word2vec_dim=None):

		self.word_vec = word2vec
		self.word2vec_dim = word2vec_dim
		self.word2vec = None

		data = pkl.load(open(corpus_path, "rb"))
		self.train_data = data["train"]
		self.valid_data = data["valid"]
		self.test_data = data["test"]

		print("DailyDialog Statistics: ")
		print("train data size: %d" % len(self.train_data))
		print("valid data size: %d" % len(self.valid_data))
		print("test data size: %d" % len(self.test_data))
		print("\n")

		# DailyDialog Statistics:
		# train data size: 10117
		# valid data size: 1500
		# test data size: 1500

		self.train_corpus = self.process(self.train_data)
		self.valid_corpus = self.process(self.valid_data)
		self.test_corpus = self.process(self.test_data)

		print(" [*] Building word vocabulary.")
		self.build_vocab(max_vocab_cnt)

		print(" [*] Loading word2vec.")
		self.load_word2vec()
		
	def process(self, data):

		new_meta = []
		new_dialog = []
		all_lenes = []
		new_utts = []

		for obj in data:
			
			topic = obj["topic"]
			dial = obj["utts"]

			lower_utts = [
				(
					item["floor"],
					# ["<s>"] + item["text"].lower().strip().split(" ") + ["</s>"],
					["<s>"] + nltk.WordPunctTokenizer().tokenize(item["text"].lower().strip()) + ["</s>"],
					(item["act"], item["emot"])
				) for item in dial]

			# first
			all_lenes.extend([len(u) for c, u, f in lower_utts])

			# second
			new_utts.extend([utt for floor, utt, feat in lower_utts])

			# third
			dialog = [(utt, floor, feat) for floor, utt, feat in lower_utts]
			new_dialog.append(dialog)

			# fourth
			meta = (topic,)
			new_meta.append(meta)

		
		print("max_utt_len %d, min_utt_len %d, mean_utt_len %.4f" % \
			(np.max(all_lenes),np.min(all_lenes), float(np.mean(all_lenes))))

		# Max utt len 298, Min utt len 3, Mean utt len 16.54
		# Max utt len 156, Min utt len 3, Mean utt len 16.83
		# Max utt len 232, Min utt len 3, Mean utt len 16.80

		return {"dialog": new_dialog, "meta": new_meta, "utts": new_utts}

	def build_vocab(self, max_vocab_cnt):

		all_words = []
		for tokens in self.train_corpus["utts"]:
			all_words.extend(tokens)

		vocab_count = Counter(all_words).most_common()
		raw_vocab_size = len(vocab_count)

		discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
		vocab_count = vocab_count[0:max_vocab_cnt]

		self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count] # list
		self.rev_vocab = self.word2idx = {word:idx for idx, word in enumerate(self.vocab)} # dict
		self.idx2word = {idx:word for idx, word in enumerate(self.vocab)} # dict
		
		self.pad_id = self.word2idx["<pad>"]
		self.unk_id = self.word2idx["<unk>"]
		self.sos_id = self.word2idx["<s>"]
		self.eos_id = self.word2idx["</s>"]
		self.vocab_size = len(self.vocab)
		
		print("raw_vocab_size %d, actual_vocab_size %d, at cut_off frequent %d OOV rate %f"
			  % (raw_vocab_size,  self.vocab_size, 
				vocab_count[-1][1], 
				float(discard_wc) / len(all_words)))

		print("<pad> index %d" % self.pad_id)
		print("<unk> index %d" % self.unk_id)
		print("<s> index %d" % self.sos_id)
		print("</s> index %d" % self.eos_id)
		print("\n")


		print("Building topic vocabulary...")
		all_topics = []
		for topic, in self.train_corpus["meta"]:
			all_topics.append(topic)
		self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
		self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
		print("number of topics: %d" % len(self.topic_vocab))
		print(self.topic_vocab)
		print("\n")


		
		all_dialog_acts = []
		all_emots = []
		for dialog in self.train_corpus["dialog"]:
			all_dialog_acts.extend([feat[0] for floor, utt, feat in dialog if feat is not None])
			all_emots.extend([feat[1] for floor, utt, feat in dialog if feat is not None])



		print("Building act vocabulary...")
		self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
		self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
		print("number of acts: %d" % len(self.dialog_act_vocab))
		print(self.dialog_act_vocab)
		print("\n")



		print("Building emotion vocabulary...")
		self.dialog_emot_vocab = [t for t, cnt in Counter(all_emots).most_common()]
		self.rev_dialog_emot_vocab = {t: idx for idx, t in enumerate(self.dialog_emot_vocab)}
		print("number of emots: %d" % len(self.dialog_emot_vocab))
		print(self.dialog_emot_vocab)
		print("\n")

	def load_word2vec(self):

		if self.word_vec is False:
			print(" [*] No word2vec provided.")
			return None

		with open("data/glove.twitter.27B.200d.txt", "r") as f:
			lines = f.readlines()

		raw_word2vec = {}
		for l in lines:
			w, vec = l.split(" ", 1)
			raw_word2vec[w] = vec
		
		self.word2vec = None
		oov_cnt = 0
		for word in self.vocab:
			str_vec = raw_word2vec.get(word, None)
			if str_vec is None:
				oov_cnt += 1
				vec = np.random.randn(self.word2vec_dim) * 0.1
			else:
				vec = np.fromstring(str_vec, sep=" ")
			vec = np.expand_dims(vec, axis=0)
			self.word2vec = np.concatenate((self.word2vec, vec),0) if self.word2vec is not None else vec

		print(" [*] word2vec shape: ")
		print(self.word2vec.shape)
		print(" [*] word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))
		
		return self.word2vec

	def get_dialog_corpus(self):
		def _to_id_corpus(data):
			results = []
			for dialog in data:
				temp = []
				for utt, floor, feat in dialog:
					if feat is not None:
						id_feat = list(feat)
						id_feat[0] = self.rev_dialog_act_vocab[feat[0]]
						id_feat[1] = self.rev_dialog_emot_vocab[feat[1]]
					else:
						id_feat = None
					# temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
					temp.append((utt, floor, id_feat))
				results.append(temp)
			return results

		id_train = _to_id_corpus(self.train_corpus["dialog"])
		id_valid = _to_id_corpus(self.valid_corpus["dialog"])
		id_test = _to_id_corpus(self.test_corpus["dialog"])
		return {'train': id_train, 'valid': id_valid, 'test': id_test}

	def get_meta_corpus(self):
		def _to_id_corpus(data):
			results = []
			for (topic,) in data:
				results.append((self.rev_topic_vocab[topic]))
			return results
		id_train = _to_id_corpus(self.train_corpus["meta"])
		id_valid = _to_id_corpus(self.valid_corpus["meta"])
		id_test = _to_id_corpus(self.test_corpus["meta"])
		return {'train': id_train, 'valid': id_valid, 'test': id_test}








class SWDADialogCorpus(object):
	dialog_act_id = 0
	sentiment_id = 1
	liwc_id = 2

	def __init__(self, corpus_path="data/swda/full_swda_clean_42da_sentiment_dialog_corpus.p", 
		max_vocab_cnt=30000, word2vec=True, word2vec_dim=None):

		self.word_vec = word2vec
		self.word2vec_dim = word2vec_dim
		self.word2vec = None

		# self.dialog_id = 0
		# self.meta_id = 1
		# self.utt_id = 2
		# self.sil_utt = ["<s>", "<sil>", "</s>"]

		data = pkl.load(open(corpus_path, "rb"))
		self.train_corpus = self.process(data["train"])
		self.valid_corpus = self.process(data["valid"])
		self.test_corpus = self.process(data["test"])

		print("SWDA Statistics: ")
		print("train data size: %d" % len(self.train_corpus))
		print("valid data size: %d" % len(self.valid_corpus))
		print("test data size: %d" % len(self.test_corpus))
		print("\n")

		self.build_vocab(max_vocab_cnt)
		self.load_word2vec()
		print("Done loading corpus")


	def process(self, data):

		new_dialog = []
		new_meta = []
		new_utts = []
		# bod_utt = ["<s>", "<d>", "</s>"]
		all_lenes = []
		for l in data:
			lower_utts = [(caller, ["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower().strip()) + ["</s>"], feat)
						  for caller, utt, feat in l["utts"]]
			all_lenes.extend([len(u) for c, u, f in lower_utts])
			a_age = float(l["A"]["age"])/100.0
			b_age = float(l["B"]["age"])/100.0
			a_edu = float(l["A"]["education"])/3.0
			b_edu = float(l["B"]["education"])/3.0
			vec_a_meta = [a_age, a_edu] + ([0, 1] if l["A"]["sex"] == "FEMALE" else [1, 0])
			vec_b_meta = [b_age, b_edu] + ([0, 1] if l["B"]["sex"] == "FEMALE" else [1, 0])

			# for joint model we mode two side of speakers together. if A then its 0 other wise 1
			meta = (vec_a_meta, vec_b_meta, l["topic"])
			dialog = [(utt, int(caller=="B"), feat) for caller, utt, feat in lower_utts]
			# dialog = [(bod_utt, 0, None)] + [(utt, int(caller=="B"), feat) for caller, utt, feat in lower_utts]

			# new_utts.extend([bod_utt] + [utt for caller, utt, feat in lower_utts])
			new_utts.extend([utt for caller, utt, feat in lower_utts])
			new_dialog.append(dialog)
			new_meta.append(meta)

		print("max_utt_len %d, mean_utt_len %.2f, min_utt_len %d" % (
			np.max(all_lenes), float(np.mean(all_lenes)), np.min(all_lenes)))

		return {"dialog": new_dialog, "meta": new_meta, "utts": new_utts}


	def build_vocab(self, max_vocab_cnt):

		all_words = []
		for tokens in self.train_corpus["utts"]:
			all_words.extend(tokens)

		vocab_count = Counter(all_words).most_common()
		raw_vocab_size = len(vocab_count)
		discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
		vocab_count = vocab_count[0:max_vocab_cnt]

		print("raw vocab size %d, vocab size %d, at cut_off %d OOV rate %f"
			  % (raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

		self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
		self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}

		self.pad_id = self.rev_vocab["<pad>"]
		self.unk_id = self.rev_vocab["<unk>"]
		self.sos_id = self.rev_vocab["<s>"]
		self.eos_id = self.rev_vocab["</s>"]
		self.vocab_size = len(self.vocab)

		print("<pad> index %d" % self.rev_vocab["<pad>"])
		print("<unk> index %d" % self.rev_vocab["<unk>"])
		print("<s> index %d" % self.rev_vocab["<s>"])
		print("</s> index %d" % self.rev_vocab["</s>"])

		
		all_topics = []
		for a, b, topic in self.train_corpus["meta"]:
			all_topics.append(topic)
		self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
		self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
		print("%d topics in train data" % len(self.topic_vocab))
		print(self.topic_vocab)


		all_dialog_acts = []
		for dialog in self.train_corpus["dialog"]:
			all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])
		self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
		self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
		print("%d dialog acts in train data" % len(self.dialog_act_vocab))
		print(self.dialog_act_vocab)



	def load_word2vec(self):

		if self.word_vec is False:
			print(" [*] No word2vec provided.")
			return None

		with open("data/glove.twitter.27B.200d.txt", "r") as f:
			lines = f.readlines()

		raw_word2vec = {}
		for l in lines:
			w, vec = l.split(" ", 1)
			raw_word2vec[w] = vec
		
		self.word2vec = None
		oov_cnt = 0
		for word in self.vocab:
			str_vec = raw_word2vec.get(word, None)
			if str_vec is None:
				oov_cnt += 1
				vec = np.random.randn(self.word2vec_dim) * 0.1
			else:
				vec = np.fromstring(str_vec, sep=" ")
			vec = np.expand_dims(vec, axis=0)
			self.word2vec = np.concatenate((self.word2vec, vec),0) if self.word2vec is not None else vec

		print(" [*] word2vec shape: ")
		print(self.word2vec.shape)
		print(" [*] word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))
		
		return self.word2vec


	"""
	def get_utt_corpus(self):
		def _to_id_corpus(data):
			results = []
			for line in data:
				results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
			return results
		# convert the corpus into ID
		id_train = _to_id_corpus(self.train_corpus[self.utt_id])
		id_valid = _to_id_corpus(self.valid_corpus[self.utt_id])
		id_test = _to_id_corpus(self.test_corpus[self.utt_id])
		return {'train': id_train, 'valid': id_valid, 'test': id_test}
	"""


	def get_dialog_corpus(self):
		def _to_id_corpus(data):
			results = []
			for dialog in data:
				temp = []
				
				for utt, floor, feat in dialog:
					if feat is not None:
						id_feat = list(feat)
						id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
					else:
						id_feat = None

					temp.append((utt, floor, id_feat))
					# temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
				results.append(temp)
			return results

		id_train = _to_id_corpus(self.train_corpus["dialog"])
		id_valid = _to_id_corpus(self.valid_corpus["dialog"])
		id_test = _to_id_corpus(self.test_corpus["dialog"])

		return {'train': id_train, 'valid': id_valid, 'test': id_test}


	def get_meta_corpus(self):
		def _to_id_corpus(data):
			results = []
			for m_meta, o_meta, topic in data:
				results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
			return results

		id_train = _to_id_corpus(self.train_corpus[self.meta_id])
		id_valid = _to_id_corpus(self.valid_corpus[self.meta_id])
		id_test = _to_id_corpus(self.test_corpus[self.meta_id])
		return {'train': id_train, 'valid': id_valid, 'test': id_test}