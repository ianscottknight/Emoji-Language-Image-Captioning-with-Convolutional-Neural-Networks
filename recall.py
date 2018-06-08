import pickle 

id_to_prediction_dict = {}
id_to_actual_dict = {}

with open('models/8-layer/pkls/id_to_prediction_dict.pkl', 'rb') as f: 
    id_to_prediction_dict = pickle.load(f)

with open('models/8-layer/pkls/id_to_actual_dict.pkl', 'rb') as f: 
    id_to_actual_dict = pickle.load(f)

score = 0

for key in id_to_prediction_dict.keys():
    for actual in id_to_actual_dict[key]:
        if actual in id_to_prediction_dict[key]:
            score += 0.2

score /= len(id_to_prediction_dict.keys())

print(score)
