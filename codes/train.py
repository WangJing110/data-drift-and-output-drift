import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

TRAIN_PATH = "../dataset/showcase/china_original/train/"
TEST_PATH = "../dataset/showcase/china_original/test/"


transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    # transforms.Grayscale(),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
trainloader = DataLoader(dataset=train_dataset,
                         batch_size=100,
                         shuffle=True,
                         num_workers=0)

test_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform)
testloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0)

classes = ('direction', 'mandatory', 'prohibitory', 'speed_limit', 'warning')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(2048, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 20 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / (20 * 100)))
            running_loss = 0.0

            # dataiter = iter(testloader)
            # images, labels = dataiter.next()
            # outputs = net(images)
            # _, predicted = torch.max(outputs, 1)

            # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
            #                               for j in range(4)))

            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for i, (images, labels) in enumerate(testloader, 0):
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    sample_fname, _ = testloader.dataset.samples[i]
                    # if predicted != labels and epoch > 10:
                    #     print(sample_fname, classes[predicted[0]])
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images: %d %%' % (
                    100 * correct / total))

print('Finished Training')

PATH = './sign_net_china_ori.pth'
torch.save(net.state_dict(), PATH)
