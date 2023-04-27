import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import configuration
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torchviz import make_dot


class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.25)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.drop4(x)
        x = self.fc2(x)
        return x


def train_process(model, config, save_model):
    """
    Training/Testing process
    :param model:
    :param config:
    :param save_model:
    :return:
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(1, config.EPOCHS + 1):
        train(model, epoch, config, criterion, optimizer, save_model)
        test(model, config, criterion)
    epochs = [i for i in range(1, config.EPOCHS)]
    plot_train_curve(config, epochs)
    plot_accuracy(config, epochs)


def train(model, epoch, config, criterion, optimizer, save_model):
    model.train()
    train_correct = 0
    total_train_loss = 0
    for batch_idx, (images, labels) in tqdm(enumerate(config.train_loader), desc=f'Training model epoch:{epoch}'):
        images, labels = images.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        loss.backward()
        total_train_loss += loss

        optimizer.step()
        if batch_idx % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(config.train_loader.dataset),
                       100. * batch_idx / len(config.train_loader), loss.item()))

            if save_model: #save current model
                if config.device == 'cuda':
                    model.to('cpu')
                torch.save(model.state_dict(), 'results/model_new.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer_new.pth')
                model.to('cuda')
    config.train_losses.append(total_train_loss.cpu().detach().numpy() / config.train_counter)
    config.train_correct.append(train_correct / len(config.train_loader.dataset))


def test(model, config, criterion):
    """
    Tests the current model
    :param model: CNN model
    :param config:
    :param criterion:
    :return:
    """
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for (image, label) in config.test_loader:
            image, label = image.to(config.device), label.to(config.device)
            output = model(image)
            test_loss += criterion(output, label).item()
            test_acc += (output.argmax(1) == label).type(torch.float).sum().item()
    config.test_losses.append(test_loss / config.test_counter)
    config.test_correct.append(test_acc / len(config.test_loader.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss * config.BATCH_SIZE/ len(config.test_loader.dataset), test_acc, len(config.test_loader.dataset),
        100. * test_acc / len(config.test_loader.dataset)))


def output_predictions(network, prep_model):
    """
    Predicts the output based on the test data available
    :param network:  CNN model
    :param prep_model: dataset
    :return:
    """
    mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}
    network.eval()
    with torch.no_grad():
        example_data = next(iter(prep_model.test_loader))
        images, data = example_data
        correct_data = data[:12]

        cuda_images, data = images.to(prep_model.device), data.to(prep_model.device)
        output = network(cuda_images[:12])
        for i in range(12):
            prediction = output.data.max(1, keepdim=True)[1][i].item()
            pred = mapping[prediction]
            plt.subplot(4, 3, i + 1)
            plt.tight_layout()
            plt.imshow(images[i][0], cmap='gray', interpolation='none')
            plt.title(f'Prediction: {pred}')
            plt.xticks([])
            plt.yticks([])
        plt.show()


def plot_train_curve(data_model, epochs):
    """
    Plots the training loss for the model
    :param data_model:
    :return:
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    ax.plot(range(1, len(data_model.train_losses) + 1), data_model.train_losses, label='train_loss')
    ax.plot(range(1, len(data_model.test_losses) + 1), data_model.test_losses, label='test_loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train and Validation Loss')

    ax.legend(loc='upper right')
    plt.show()

def plot_accuracy(data_model,epochs):
    """
    Plots the accuracy of test and training sets for each epoch
    :param data_model:
    :param epochs:
    :return:
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    ax.plot(range(1, len(data_model.train_correct) + 1), data_model.train_correct, label='train_acc')
    ax.plot(range(1, len(data_model.test_correct) + 1), data_model.test_correct, label='test_acc')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train and Validation Accuracy')

    ax.legend(loc='lower right')
    plt.show()

def plot_confussion_matrix(model, data_model):
    """
    Plots confusion matrix heatmap
    :param model:
    :param data_model:
    :return:
    """
    model.eval()
    validation_labels = []
    validation_pred_probs = []
    with torch.no_grad():
        for inputs, labels in data_model.test_loader:
            inputs, labels = inputs.to(data_model.device), labels.to(data_model.device)
            outputs = model(inputs)
            validation_labels.extend(labels.cpu().numpy())
            validation_pred_probs.extend(outputs.cpu().numpy())

    validation_pred_labels = np.argmax(validation_pred_probs, axis=1)

    # Compute the confusion matrix
    confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
    print(classification_report(validation_labels, validation_pred_labels))
    class_names = list(data_model.train_loader.dataset.classes)
    sns.set()
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def save_model_diagram(model, prep_data):
    """
    Saves the model diagram as png (too large for EmotionCNN)
    :param model:
    :param prep_data:
    :return:
    """
    data = torch.randn(128, 1, 48, 48)
    data = data.to(prep_data.device)
    out = model(data)
    make_dot(out, params=dict(model.named_parameters())).render("model", format='png')
