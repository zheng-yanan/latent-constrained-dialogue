import numpy as np


txts = open("dialogues_text.txt", "r").readlines()
tops = open("dialogues_topic.txt", "r").readlines()
emots = open("dialogues_emotion.txt", "r").readlines()
acts = open("dialogues_act.txt", "r").readlines()

assert len(txts) == len(tops) == len(emots) == len(acts) 



# average turn per dialog
def avg_turn_per_dialog(lines):

	turn_count = []
	for line in lines:
		split = line.lower().replace(" ' ", "'").strip().split("__eou__")[:-1]
		turn_count.append(len(split))
	
	print("max_turn: %d avg_turn: %.4f min_turn: %d" % 
		(np.max(turn_count), np.mean(turn_count), np.min(turn_count)))
	# max_turn: 35 avg_turn: 7.8503 min_turn: 2

	
# average tokens per utterance/turn
def avg_tokens_per_turn(lines):
	token_count = []
	for line in lines:
		split = line.lower().replace(" ' ", "'").strip().split("__eou__")[:-1]
		for sent in split:
			token_count.append(len(sent.strip().split(" ")))

	print("max_utt_len: %d avg_utt_len: %.4f min_utt_len: %d" % 
		(np.max(token_count), np.mean(token_count), np.min(token_count)))
	# max_utt_len: 278 avg_utt_len: 13.3163 min_utt_len: 1



avg_turn_per_dialog(txts)
avg_tokens_per_turn(txts)



num_total = len(txts)

for idx in range(num_total):

	line = txts[idx]
	line = line.lower().replace(" ' ", "'").strip().split("__eou__")[:-1]

	act = acts[idx].replace("\n", "").strip().split(" ")
	act = [int(item) - 1 for item in act]

	emot = emots[idx].replace("\n", "").strip().split(" ")
	emot = [int(item) for item in emot]

	top = int(tops[idx]) - 1

