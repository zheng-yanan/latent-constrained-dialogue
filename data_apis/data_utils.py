import numpy as np


class LongDataLoader(object):
	"""A special efficient data loader for TBPTT"""
	batch_size = 0
	backward_size = 0
	step_size = 0
	ptr = 0
	num_batch = None
	batch_indexes = None
	grid_indexes = None
	indexes = None
	data_lens = None
	data_size = None
	prev_alive_size = 0
	name = None

	def _shuffle_batch_indexes(self):
		np.random.shuffle(self.batch_indexes)

	def _prepare_batch(self, cur_grid, prev_grid):
		raise NotImplementedError("Have to override prepare batch")

	def epoch_init(self, batch_size, backward_size, step_size, shuffle=True, intra_shuffle=True):
		assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

		self.ptr = 0
		self.batch_size = batch_size
		self.backward_size = backward_size
		self.step_size = step_size
		self.prev_alive_size = batch_size

		# create batch indexes
		temp_num_batch = self.data_size // batch_size
		self.batch_indexes = []
		for i in range(temp_num_batch):
			self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

		left_over = self.data_size-temp_num_batch*batch_size

		# shuffle batch indexes
		if shuffle:
			self._shuffle_batch_indexes()

		# create grid indexes
		self.grid_indexes = []
		for idx, b_ids in enumerate(self.batch_indexes):
			# assume the b_ids are sorted
			all_lens = [self.data_lens[i] for i in b_ids]
			max_len = self.data_lens[b_ids[-1]]
			min_len = self.data_lens[b_ids[0]]
			assert np.max(all_lens) == max_len
			assert np.min(all_lens) == min_len
			num_seg = (max_len-self.backward_size) // self.step_size
			if num_seg > 0:
				cut_start = range(0, num_seg*self.step_size, step_size)
				cut_end = range(self.backward_size, num_seg*self.step_size+self.backward_size, step_size)
				assert cut_end[-1] < max_len
				cut_start = [0] * (self.backward_size-2) +cut_start # since we give up on the seq training idea
				cut_end = range(2, self.backward_size) + cut_end
			else:
				cut_start = [0] * (max_len-2)
				cut_end = range(2, max_len)

			new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_len-1]
			if intra_shuffle and shuffle:
			   np.random.shuffle(new_grids)
			self.grid_indexes.extend(new_grids)

		self.num_batch = len(self.grid_indexes)
		print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))


	def next_batch(self):
		if self.ptr < self.num_batch:
			current_grid = self.grid_indexes[self.ptr]
			if self.ptr > 0:
				prev_grid = self.grid_indexes[self.ptr-1]
			else:
				prev_grid = None
			self.ptr += 1
			return self._prepare_batch(cur_grid=current_grid, prev_grid=prev_grid)
		else:
			return None


	def next_new_batch(self):
		if self.ptr < self.num_batch:
			current_grid = self.grid_indexes[self.ptr]
			if self.ptr > 0:
				prev_grid = self.grid_indexes[self.ptr-1]
			else:
				prev_grid = None
			self.ptr += 1
			return self._prepare_new_batch(cur_grid=current_grid, prev_grid=prev_grid)
		else:
			return None




class DailyDataLoader(LongDataLoader):

	def __init__(self, name, data, meta_data, config):

		assert len(data) == len(meta_data)
		self.name = name
		self.data = data
		self.meta_data = meta_data
		self.data_size = len(data)

		self.max_utt_len = config.max_utt_len

		self.data_lens = all_lens = [len(dial) for dial in self.data]

		print("max_dialog_len %d and min_dialog_len %d and mean_dialog_len %f" % 
			(np.max(all_lens), np.min(all_lens), float(np.mean(all_lens))))
		
		self.indexes = list(np.argsort(all_lens))


	def pad_to(self, tokens, size, do_pad=True):
		if len(tokens) >= size:
			return tokens[0:size-1] + [tokens[-1]]
		elif do_pad:
			return tokens + ["<pad>"] * (size-len(tokens))
		else:
			return tokens


	def _prepare_batch(self, cur_grid, prev_grid=None):
		
		b_id, s_id, e_id = cur_grid
		batch_ids = self.batch_indexes[b_id]
		rows = [self.data[idx] for idx in batch_ids]
		meta_rows = [self.meta_data[idx] for idx in batch_ids]
		topics = np.array([meta for meta in meta_rows])
		
		context_lens, context_utts, floors, out_utts, out_lens, out_floors, out_das = [], [], [], [], [], [], []

		# count length
		in_len_temp = []
		out_len_temp = []
		for row in rows:
			if s_id < len(row)-1:
				cur_row = row[s_id:e_id]
				in_row = cur_row[0:-1]
				out_row = cur_row[-1]

				for utt, floor, feat in in_row:
					in_len_temp.append(len(utt))

				utt, floor, feat = out_row
				out_len_temp.append(len(utt))

			else:
				print(row)
				raise ValueError("S_ID %d larger than row" % s_id)

		max_in_len = min(max(in_len_temp), self.max_utt_len)
		max_out_len = min(max(out_len_temp), self.max_utt_len)

		for row in rows:
			if s_id < len(row)-1:

				cut_row = row[s_id:e_id]
				in_row = cut_row[0:-1]
				out_row = cut_row[-1]

				context_utts.append([self.pad_to(utt, max_in_len) for utt, floor, feat in in_row])
				context_lens.append(len(in_row))

				out_utt, out_floor, out_feat = out_row
				out_utts.append(self.pad_to(out_utt, max_out_len))
				out_lens.append(len(self.pad_to(out_utt, max_out_len, do_pad=False)))

				out_floors.append(out_floor)
				out_das.append(out_feat[0])
			else:
				print(row)
				raise ValueError("S_ID %d larger than row" % s_id)


		vec_out_das = np.array(out_das)
		vec_floors = np.array(out_floors)
		vec_out_lens = np.array(out_lens)
		vec_outs = np.array(out_utts)
		vec_context_lens = np.array(context_lens)
	

		dial_utt_len = min(context_lens)

		vec_context = []
		for b_id in range(self.batch_size):
			# vec_context.append(pad_cont(context_utts[b_id]))
			vec_context.append(context_utts[b_id][-dial_utt_len:])

		vec_context = np.array(vec_context)

		return vec_context, vec_context_lens, vec_floors, topics, None, None, vec_outs, vec_out_lens, vec_out_das


	def _prepare_new_batch(self, cur_grid, prev_grid=None):
		
		b_id, s_id, e_id = cur_grid
		batch_ids = self.batch_indexes[b_id]
		rows = [self.data[idx] for idx in batch_ids]
		meta_rows = [self.meta_data[idx] for idx in batch_ids]
		topics = np.array([meta for meta in meta_rows])
		
		context_lens, context_utts, floors, out_utts, out_lens, out_floors, out_das = [], [], [], [], [], [], []

		# count length
		in_len_temp = []
		out_len_temp = []
		for row in rows:
			if s_id < len(row)-1:
				cur_row = row[s_id:e_id]
				in_row = cur_row[0:-1]
				out_row = cur_row[1:]

				for utt, floor, feat in in_row:
					in_len_temp.append(len(utt))

				for utt, floor, feat in out_row:
					out_len_temp.append(len(utt))

			else:
				print(row)
				raise ValueError("S_ID %d larger than row" % s_id)

		max_in_len = min(max(in_len_temp), self.max_utt_len)
		max_out_len = min(max(out_len_temp), self.max_utt_len)

		for row in rows:
			if s_id < len(row)-1:

				cut_row = row[s_id:e_id]
				in_row = cut_row[0:-1]
				out_row = cut_row[1:]

				context_utts.append([self.pad_to(utt, max_in_len) for utt, _, _ in in_row])
				context_lens.append(len(in_row))

				out_utts.append([self.pad_to(out_utt, max_out_len) for out_utt, _, _ in out_row])
				out_lens.append([len(self.pad_to(out_utt, max_out_len, do_pad=False)) for out_utt, _, _ in out_row])

				# out_utt, out_floor, out_feat = out_row
				# out_utts.append(self.pad_to(out_utt, max_out_len))
				# out_lens.append(len(self.pad_to(out_utt, max_out_len, do_pad=False)))
				# out_floors.append(out_floor)
				# out_das.append(out_feat[0])

			else:
				print(row)
				raise ValueError("S_ID %d larger than row" % s_id)


		# vec_out_das = np.array(out_das)
		# vec_floors = np.array(out_floors)
		# vec_out_lens = np.array(out_lens)
		# vec_outs = np.array(out_utts)

		vec_context_lens = np.array(context_lens)
		dial_utt_len = min(context_lens)

		vec_context = []
		vec_outs = []
		vec_out_lens = []

		for b_id in range(self.batch_size):
			
			vec_context.append(context_utts[b_id][-dial_utt_len:])
			vec_outs.append(out_utts[b_id][-dial_utt_len:])
			vec_out_lens.append(out_lens[b_id][-dial_utt_len:])

		vec_context = np.array(vec_context)
		vec_outs = np.array(vec_outs)
		vec_out_lens = np.array(vec_out_lens)

		return vec_context, vec_context_lens, None, topics, None, None, vec_outs, vec_out_lens, None




class SWDADataLoader(LongDataLoader):
	def __init__(self, name, data, meta_data, config):

		assert len(data) == len(meta_data)
		self.name = name
		self.data = data
		self.meta_data = meta_data
		self.data_size = len(data)
		self.data_lens = all_lens = [len(line) for line in self.data]
		self.max_utt_len = config.max_utt_len

		print("max_dialog_len: %d, min_dialog_len: %d, avg_dialog_len: %f" % 
			(np.max(all_lens), np.min(all_lens), float(np.mean(all_lens))))

		self.indexes = list(np.argsort(all_lens))


	def pad_to(self, tokens, size, do_pad=True):
		if len(tokens) >= size:
			return tokens[0:size-1] + [tokens[-1]]
		elif do_pad:
			return tokens + ["<pad>"] * (size-len(tokens))
		else:
			return tokens


	def _prepare_batch(self, cur_grid, prev_grid):
		
		b_id, s_id, e_id = cur_grid
		batch_ids = self.batch_indexes[b_id]
		rows = [self.data[idx] for idx in batch_ids]
		meta_rows = [self.meta_data[idx] for idx in batch_ids]
		dialog_lens = [self.data_lens[idx] for idx in batch_ids]

		topics = np.array([meta[2] for meta in meta_rows])
		cur_pos = [np.minimum(1.0, e_id/float(l)) for l in dialog_lens]

		in_len_temp = []
		out_len_temp = []
		for row in rows:
			if s_id < len(row)-1:
				cut_row = row[s_id:e_id]
				in_row = cut_row[0:-1]
				out_row = cut_row[-1]

				for utt, floor, feat in in_row:
					in_len_temp.append(len(utt))

				out_utt, out_floor, out_feat = out_row
				out_len_temp.append(len(out_utt))
			else:
				print(row)
				raise ValueError("S_ID %d larger than row" % s_id)


		max_in_len = min(max(in_len_temp), self.max_utt_len)
		max_out_len = min(max(out_len_temp), self.max_utt_len)

		context_lens, context_utts, floors, out_utts, out_lens, out_floors, out_das = [], [], [], [], [], [], []
		for row in rows:
			if s_id < len(row)-1:
				cut_row = row[s_id:e_id]
				in_row = cut_row[0:-1]
				out_row = cut_row[-1]
				out_utt, out_floor, out_feat = out_row

				context_utts.append([self.pad_to(utt, max_in_len) for utt, floor, feat in in_row])
				floors.append([int(floor==out_floor) for utt, floor, feat in in_row])
				context_lens.append(len(cut_row) - 1)

				out_utt = self.pad_to(out_utt, max_out_len, do_pad=False)
				out_utts.append(self.pad_to(out_utt, max_out_len))
				out_lens.append(len(out_utt))
				out_floors.append(out_floor)
				out_das.append(out_feat[0])
			else:
				print(row)
				raise ValueError("S_ID %d larger than row" % s_id)

		my_profiles = np.array([meta[out_floors[idx]] for idx, meta in enumerate(meta_rows)])
		ot_profiles = np.array([meta[1-out_floors[idx]] for idx, meta in enumerate(meta_rows)])
		vec_context_lens = np.array(context_lens)
		# vec_context = np.zeros((self.batch_size, np.max(vec_context_lens), self.max_utt_size), dtype=np.int32)
		# vec_floors = np.zeros((self.batch_size, np.max(vec_context_lens)), dtype=np.int32)
		# vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
		vec_outs = np.array(out_utts)
		vec_out_lens = np.array(out_lens)
		vec_out_das = np.array(out_das)
		vec_floors = np.array(out_floors)

		vec_context = []
		dial_utt_len = min(context_lens)
		for b_id in range(self.batch_size):
			# vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
			# vec_floors[b_id, 0:vec_context_lens[b_id]] = floors[b_id]
			# vec_context[b_id, 0:vec_context_lens[b_id], :] = np.array(context_utts[b_id])
			# vec_context.append(pad_cont(context_utts[b_id]))
			vec_context.append(context_utts[b_id][-dial_utt_len:])

		vec_context = np.array(vec_context)

		return vec_context, vec_context_lens, vec_floors, topics, my_profiles, ot_profiles, vec_outs, vec_out_lens, vec_out_das




	def _prepare_new_batch(self, cur_grid, prev_grid):
		
		b_id, s_id, e_id = cur_grid
		batch_ids = self.batch_indexes[b_id]
		rows = [self.data[idx] for idx in batch_ids]
		meta_rows = [self.meta_data[idx] for idx in batch_ids]
		dialog_lens = [self.data_lens[idx] for idx in batch_ids]

		topics = np.array([meta[2] for meta in meta_rows])
		cur_pos = [np.minimum(1.0, e_id/float(l)) for l in dialog_lens]

		in_len_temp = []
		out_len_temp = []
		for row in rows:
			if s_id < len(row)-1:
				cut_row = row[s_id:e_id]
				in_row = cut_row[0:-1]
				out_row = cut_row[1:]

				for utt, floor, feat in in_row:
					in_len_temp.append(len(utt))

				for out_utt, out_floor, out_feat in out_row:
					out_len_temp.append(len(out_utt))
			else:
				print(row)
				raise ValueError("S_ID %d larger than row" % s_id)


		max_in_len = min(max(in_len_temp), self.max_utt_len)
		max_out_len = min(max(out_len_temp), self.max_utt_len)

		context_lens, context_utts, floors, out_utts, out_lens, out_floors, out_das = [], [], [], [], [], [], []
		for row in rows:
			if s_id < len(row)-1:
				cut_row = row[s_id:e_id]
				in_row = cut_row[0:-1]
				out_row = cut_row[1:]

				context_utts.append([self.pad_to(utt, max_in_len) for utt, floor, feat in in_row])
				context_lens.append(len(cut_row) - 1)

				out_utts.append([self.pad_to(out_utt, max_out_len) for out_utt, _, _ in out_row])
				out_lens.append([len(self.pad_to(out_utt, max_out_len, do_pad=False)) for out_utt, _, _ in out_row])
				
				# out_floors.append(out_floor)
				# out_das.append(out_feat[0])
			else:
				print(row)
				raise ValueError("S_ID %d larger than row" % s_id)

		# my_profiles = np.array([meta[out_floors[idx]] for idx, meta in enumerate(meta_rows)])
		# ot_profiles = np.array([meta[1-out_floors[idx]] for idx, meta in enumerate(meta_rows)])
		vec_context_lens = np.array(context_lens)
		dial_utt_len = min(context_lens)
		# vec_context = np.zeros((self.batch_size, np.max(vec_context_lens), self.max_utt_size), dtype=np.int32)
		# vec_floors = np.zeros((self.batch_size, np.max(vec_context_lens)), dtype=np.int32)
		# vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
		# vec_outs = np.array(out_utts)
		# vec_out_lens = np.array(out_lens)
		# vec_out_das = np.array(out_das)
		# vec_floors = np.array(out_floors)

		vec_context = []
		vec_outs = []
		vec_out_lens = []

		
		for b_id in range(self.batch_size):
			vec_context.append(context_utts[b_id][-dial_utt_len:])
			vec_outs.append(out_utts[b_id][-dial_utt_len:])
			vec_out_lens.append(out_lens[b_id][-dial_utt_len:])

		vec_context = np.array(vec_context)
		vec_outs = np.array(vec_outs)
		vec_out_lens = np.array(vec_out_lens)

		return vec_context, vec_context_lens, None, topics, None, None, vec_outs, vec_out_lens, None


