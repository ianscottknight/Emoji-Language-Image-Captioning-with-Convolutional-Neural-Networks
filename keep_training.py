from utils import ConvNet
from utils import CocoCaptions
import utils
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import pickle

### Hyperparameters

NUM_EPOCHS = utils.NUM_EPOCHS
BATCH_SIZE = utils.BATCH_SIZE
LR = utils.LR
NUM_CLASSES = utils.NUM_CLASSES


train_dir = 'resized_train2017/'

ann_file_train = 'annotations/captions_train2017.json'

train_dataset = CocoCaptions(root=train_dir, annFile=ann_file_train, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ConvNet(NUM_CLASSES).to(device)
model.load_state_dict(torch.load('model.ckpt'))


### Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


### Train the model

loss_log = []
total_step = len(train_loader)

for epoch in range(NUM_EPOCHS):

    losses_for_epoch = []

    for i, (image_id, images, labels) in enumerate(train_loader): 
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = 0
        for j in range(labels.shape[1]):
            loss += criterion(outputs, labels[:,j])
        losses_for_epoch.append(loss)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{} / {}], Step [{} / {}], Loss: {}'.format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))
    
    loss_log.append(losses_for_epoch)


### Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')

with open('loss_log.pkl', 'wb') as f: 
    pickle.dump(loss_log, f)

