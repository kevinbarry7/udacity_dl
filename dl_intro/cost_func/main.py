from torch import nn

def create_model():
    input_size=784
    output_size=10
    model = nn.Sequential(nn.Linear(input_size, 128),
                         nn.ReLU(),
                         nn.Linear(128,64),
                         nn.ReLU(),
                         nn.Linear(64, output_size),
                         nn.LogSoftmax(dim=1))
    return model

model=create_model()