import pickle as pkl 
import sklearn

txt_file = open("dialogues_text.txt", "r")
act_file = open("dialogues_act.txt", "r")
emot_file = open("dialogues_emotion.txt", "r")
top_file = open("dialogues_topic.txt", "r")

lines = txt_file.readlines()
acts = act_file.readlines()
emots = emot_file.readlines()
tops = top_file.readlines()

assert len(lines) == len(acts) == len(emots) == len(tops)

total_count = len(lines)
print("Total Count: %d." % total_count)



data = []

for idx in range(total_count):

	line = lines[idx].replace("\n", "").replace(" ' ", "'").lower().strip().split("__eou__")[:-1]
	line = [s.strip() for s in line]
	# list of string

	act = acts[idx].replace(" \n", "").strip().split(" ")
	act = [int(item)-1 for item in act]
	# list of int, [0, 3]

	emot = emots[idx].replace(" \n", "").strip().split(" ")
	emot = [int(item) for item in emot]
	# list of int, [0, 6]

	topic = tops[idx].replace("\n", "").strip()
	topic = int(topic)-1
	# int, [0, 9]

	if (len(line) == len(act) == len(emot)) == False:
		continue

	utt_num = len(line)
	dial_obj = {"topic": topic, "utts": []}

	flag = True
	for j in range(utt_num):

		floor = int(flag)
		flag = not flag

		utts_line = line[j].strip()
		utts_act = act[j]
		utts_emot = emot[j]

		utts_item = {"floor": floor, 
					"text": utts_line,
					"act": utts_act,
					"emot": utts_emot}

		dial_obj["utts"].append(utts_item)


	data.append(dial_obj)


print("Remaining Count: %d." % len(data))
new_data = sklearn.utils.shuffle(data)

f = open("dailydialog.pkl", "wb")
pkl.dump(new_data, f)


test = new_data[:1500]
valid = new_data[1500:3000]
train = new_data[3000:]

print("train/valid/test: %d/%d/%d" % (len(train), len(valid), len(test)))

ret = {"train": train, "valid": valid, "test": test}
	
ff = open("dailydialog_split.pkl", "wb")
pkl.dump(ret, ff)


# Total Count: 13118.
# Remaining Count: 13117.
# train/valid/test: 10117/1500/1500