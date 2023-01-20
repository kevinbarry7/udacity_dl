#TODO: Import packages you need
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim


def train(model, train_loader, cost, optimizer, epoch):
    model.train()
    #TODO: Add your code here to train your model
    for e in range(epoch):
        running_loss = 0
        correct = 0
        for data, target in train_loader:
            data = data.view(data.shape[0], -1)
            optimizer.zero_grad()
            pred = model(data)
            loss = cost(pred, target)
            running_loss+= loss
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")

def test(model, test_loader):
    model.eval()
    #TODO: Add code here to test the accuracy of your model
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} = {100*(correct/len(test_loader.dataset))}%)')


def create_model():
    #TODO: Add your model code here. You can use code from previous exercises
    input_size = 784
    output_size = 10

    model = nn.Sequential(nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, output_size),
    nn.LogSoftmax(dim=1))

    return model

#TODO: Create your Data Transforms
training_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

testing_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))])

#set hyperparameters
batch_size = 64
epoch = 10

#TODO: Download and create loaders for your data
trainset = datasets.MNIST('./data2', train=True, download=True, transform=training_transform)
testset = datasets.MNIST('./data2', train=False, download=True, transform=testing_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

#NOTE: Do not change any of the variable names to ensure that the training script works properly
print("Loading Model")
model=create_model()

cost = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print('Starting Model Training')
train(model, train_loader, cost, optimizer, epoch)
print('Testing Trained Model')
test(model, test_loader)