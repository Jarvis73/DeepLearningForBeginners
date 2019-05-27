import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# print(torch.__version__)
# print(torchvision.__version__)


def get_dataset():
    trainsform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.,), (255.,))])

    def get_data_dir():
        datadir = Path(__file__).parent / "data"
        datadir.mkdir(parents=True, exist_ok=True)
        return str(datadir)
    
    trainset = torchvision.datasets.FashionMNIST(root=get_data_dir(), train=True, download=True,
                                                 transform=trainsform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testset = torchvision.datasets.FashionMNIST(root=get_data_dir(), train=False, download=True,
                                                transform=trainsform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, num_workers=4)
    return trainloader, testloader


def imshow(img):
    image = img.numpy() * 255.
    plt.imshow(np.transpose(image, (1, 2, 0)), cmap="gray")
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    trainloader, testloader = get_dataset()
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(", ".join("%5s" % classes[labels[j]] for j in range(32)))

    # Construct network
    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # train model
    for epoch in range(5):

        running_loss = 0.
        for i, data in enumerate(trainloader, 1):
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i, running_loss / 500))
                running_loss = 0.

    print("Finish Training.")

    # Test model
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(8)))
    
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print("Predicted:   ", " ".join("%5s" % classes[predicted[j]] for j in range(8)))

    # test all
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))


if __name__ == "__main__":
    main()
