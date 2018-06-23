import pickle 

id_to_prediction_dict = {}
id_to_actual_dict = {}

with open('models/8-layer/pkls/id_to_prediction_dict.pkl', 'rb') as f: 
	id_to_prediction_dict = pickle.load(f)

with open('models/8-layer/pkls/id_to_actual_dict.pkl', 'rb') as f: 
	id_to_actual_dict = pickle.load(f)


for i in range(1, 6):
	score = 0
	for key in id_to_prediction_dict.keys():
		x = 0
		for actual in id_to_actual_dict[key]:
			if actual in id_to_prediction_dict[key]:
				x += 1
		if x >= i: 
			score += 1

	score /= len(id_to_prediction_dict.keys())

	print(i)
	print(score)
	print('')

