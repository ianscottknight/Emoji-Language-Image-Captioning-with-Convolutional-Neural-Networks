from utils import ConvNet
from utils import CocoCaptions
import utils
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pickle


### Hyperparameters

BATCH_SIZE = utils.BATCH_SIZE
NUM_CLASSES = utils.NUM_CLASSES


test_dir = 'resized_val2017/'

ann_file_test = 'annotations/captions_val2017.json'

test_dataset = CocoCaptions(root=test_dir, annFile=ann_file_test, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)

MODEL_NAME = 'model.ckpt'

device = None
model = None

if torch.cuda.is_available(): 
    device = torch.device('cuda:0')
    model = ConvNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_NAME))
else: 
    device = 'cpu'
    model = ConvNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))


### Test the model

predicted = []
actual = []

id_to_prediction_dict = {}
id_to_actual_dict = {}

for i, (image_ids, images, labels) in enumerate(test_loader):
    images = images.to(device)

    # Forward pass
    outputs = model(images)
    softmax = torch.nn.Softmax()

    for j, scores in enumerate(softmax(outputs).tolist()):
        id_to_prediction_dict[int(image_ids[j])] = [utils.int_to_emoji_dict[x] for x in np.argsort(scores)[::-1][:5]]
    for j, label in enumerate(labels.tolist()):
        id_to_actual_dict[int(image_ids[j])] = [utils.int_to_emoji_dict[x] for x in label]

with open('id_to_prediction_dict.pkl', 'wb') as f:
    pickle.dump(id_to_prediction_dict, f)

with open('id_to_actual_dict.pkl', 'wb') as f:
    pickle.dump(id_to_actual_dict, f)



