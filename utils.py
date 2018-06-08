import json
import collections
from PIL import Image
import os
import torch 
import torch.nn as nn
import torch.utils.data as data


### Hyper parameters
NUM_EPOCHS = 40
BATCH_SIZE = 100
LR = 0.0001


EMOJI_CAPTIONS_TRAIN_FILE = 'annotations/captions_train.json'
EMOJI_CAPTIONS_TEST_FILE = 'annotations/captions_val.json'

id_to_emojis_dict = {}

with open(EMOJI_CAPTIONS_TRAIN_FILE) as f:
    d = json.load(f)
    for image_id in d: 
        id_to_emojis_dict[int(image_id)] = [emoji for emoji, score in d[str(image_id)][1]]
        
with open(EMOJI_CAPTIONS_TEST_FILE) as f:
    d = json.load(f)
    for image_id in d: 
        id_to_emojis_dict[int(image_id)] = [emoji for emoji, score in d[str(image_id)][1]]

classes = []
class_count_dict = collections.defaultdict(int)

for key in id_to_emojis_dict.keys():
    for emoji in id_to_emojis_dict[key]: 
        if emoji not in classes: 
            classes.append(emoji)
        class_count_dict[emoji] += 1
        
count = collections.Counter(class_count_dict)

### 320 classes have 10 or fewer instances
### 536 classes have 50 or fewer instances
### 628 classes have 100 or fewer instances


### Remove rare emojis

emoji_classes = [key for key in count.keys() if count[key] > 100]

for key in id_to_emojis_dict.keys():
    yes_list = []
    for emoji in id_to_emojis_dict[key]: 
        if emoji in emoji_classes:
            yes_list.append(emoji)
    if len(yes_list) < 5: 
        if len(yes_list) == 0:
            yes_list = sorted(count, key=count.get, reverse=True)[:5] # Just make yes_list into the most common emojis
        else:
            for i in range(5 - len(yes_list)):
                yes_list.append(yes_list[i])
    id_to_emojis_dict[key] = yes_list


### Condense emojis with similar/same meanings to just one emoji (per meaning)

categories_to_keep = [
    
    ['👨', '👨🏻', '👨🏿', '👱🏿'], # man
    ['👩', '👩🏻'], # woman
    ['👦', '👦🏻'], # boy 
    ['👧', '👧🏻'], # girl
    ['👶', '👶🏻', '🍼'], # baby
    ['👪', '👨\u200d👩\u200d👦\u200d👦', '👨\u200d👩\u200d👧\u200d👧', \
                '👩\u200d👩\u200d👦\u200d👦', '👩\u200d👩\u200d👧\u200d👧', '👨\u200d👨\u200d👧\u200d👧'], # family
    ['👥', '👫', '👬', '👭', '💑', '👨\u200d❤️\u200d👨'], # group of people
    ['👮', '👮🏿'], # police
    ['🏄', '🏄🏻', '🏄🏿'], # surfing
    ['✈️', '🛬', '🛩'], # airplane
    ['🛁', '🛀'], # bathtub
    ['🐎', '🐴', '🏇🏿', '🏇', '🎠'], # horse
    ['📷', '🎥', '🎦', '📹', '📽'], # camera
    ['🚎', '🚐', '🚌', '🚍'], # bus
    ['🚪', '🔒'], # door/window
    ['☂', '⛱', '🌂'], # umbrella
    ['🏠', '🏡', '🏚', '🏘' ], # house
    ['⏰', '⏲', '🕰', '⌚️'], # clock
    ['⛪️', '⛪'], # church
    ['🚴', '🚵', '🚳'], # bike
    ['🚘', '🚗', '🚙'], # car
    ['🐶', '🐕', '🐺'], # dog
    ['🚛', '🚚'], # truck
    ['🛋', '🛏'], # furniture
    ['⛸', '🏒', '🏑'], # ice-rink skating
    ['💺', '🏓', '🏀'], # table/seat/sitting/bench
    ['📞', '📲', '📱', '☎️', '📵'], # phone
    ['🐮', '🐄'], # cow
    ['🐱', '🐈'], # cat
    ['🐦', '🐧'], # bird
    ['🐭', '🐁'], # mouse
    ['🐻', '🍯'], # bear
    ['🏈', '🏉'], # football
    ['📚', '📕', '📖'], # book/reading
    ['🚖', '🚕'], # taxi
    ['👕', '👔', '👚'], # top/shirt
    ['💡', '🕯', '🕎'], # light/lantern
    ['🚢', '🛳', '🚤', '🛥'], # boat
    ['🍽', '🍴'], # plates/eating
    ['⚓', '⚓️'], # anchor
    ['😴', '💤', ], # sleep
    ['👜', '💼', '👝'], # bag
    ['🔵', '🔹'], # blue
    ['🎿', '⛷'], # skiing
    ['🏪', '🏬'], # store
    ['🤵', '♠️'], # suit
    ['🗿', '🗽'], # statue
    ['⚠️', '©️', '☣', '🚸'], # sign
    ['🚂', '🚉', '🚆', '🚋'], # train
    ['🚽', '🚾', '🚺'], # bathroom
    ['☕', '🍵'], # cup
    ['⌨', '🖲', '🎹'], # keyboard
    ['🎏', '🎣'], # kite
    ['🍸', '🏮'], # drink
    ['🗃', '☑️', '💝'], # box
    ['🚶', '🚷'], # walking/standing
    ['🎶', '🎷', '🎻'], # music/band
    ['🍻', '🍶'], # alcohol
    ['🍊', '✴️', '🔸'], # orange
    ['🌳', '🌲', '🏦'], # tree/branch
    ['🚮', '🚯'], # trash/garbage
    ['🌼', '🌻'], # flower
    ['🏕', '⛺️', '🎪', '⛺️'], # tent
    ['🏔', '⛰'], # mountain
    ['💵', '💶'], # paper/money
    ['🛄', '🛅'], # luggage
    ['🚦', '🚥'], # traffic light 
    ['👞', '👡'] # shoe
    
]

for category in categories_to_keep: 
    for key in id_to_emojis_dict.keys():
        yes_list = []
        for emoji in id_to_emojis_dict[key]: 
            if emoji in category:
                yes_list.append(category[0])
            else:
                yes_list.append(emoji)
        id_to_emojis_dict[key] = yes_list


### Remove remaining emojis that are erroneous in their semantic translation from text (checked by human)

categories_to_remove = ['🇩🇯', '🇮🇲', '🍨', '🖱', '👋', '💅', '💁🏻', '💁🏽', '💁🏿', \
                        '🙍🏿', '🙇🏿', '🚣🏼', '🇧🇧', '🇬🇺', '😦', '🤘', '🏁']

for key in id_to_emojis_dict.keys(): 
    yes_list = []
    for emoji in id_to_emojis_dict[key]:
        if emoji not in categories_to_remove: 
            yes_list.append(emoji)
    if len(yes_list) < 5: 
        if len(yes_list) == 0:
            yes_list = ['👨', '👩', '👦', '👧', '👶'] # Just make yes_list into common, neutral emojis (very few instances)
        else:
            for i in range(5 - len(yes_list)):
                yes_list.append(yes_list[i])
    id_to_emojis_dict[key] = yes_list


emoji_classes = []

for value in id_to_emojis_dict.values():
    for emoji in value:
        if emoji not in emoji_classes:
            emoji_classes.append(emoji)


### Convert emojis to integers

emoji_to_int_dict = {}
int_to_emoji_dict = {}

for i, emoji in enumerate(emoji_classes):
    emoji_to_int_dict[emoji] = i # tokens between 0 and 326
    int_to_emoji_dict[i] = emoji

id_to_classes_dict = {}
for key in id_to_emojis_dict.keys():
    id_to_classes_dict[key] = [emoji_to_int_dict[emoji] for emoji in id_to_emojis_dict[key]]


class CocoCaptions(data.Dataset):

    def __init__(self, root, annFile, transform=None, target_transform=None):
        
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = torch.tensor(id_to_classes_dict[img_id])

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_id, img, target

    def __len__(self):
        return len(self.ids)


### Convolutional neural network (two convolutional layers)
NUM_CLASSES = len(emoji_classes)

class ConvNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5))
        self.fc1 = nn.Linear(2048, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, NUM_CLASSES)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

