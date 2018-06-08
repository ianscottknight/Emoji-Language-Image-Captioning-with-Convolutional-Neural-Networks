import pickle 

with open('models/8-layer/pkls/id_to_prediction_dict.pkl', 'rb') as f: 
    id_to_prediction_dict = pickle.load(f)

with open('models/8-layer/pkls/id_to_actual_dict.pkl', 'rb') as f: 
    id_to_actual_dict = pickle.load(f)

for key in id_to_prediction_dict.keys():
    print('Key: {}'.format(str(key)))
    print('Predicted: {}'.format(str(id_to_prediction_dict[key])))
    print('Actual: {}'.format(str(id_to_actual_dict[key])))
    print('')