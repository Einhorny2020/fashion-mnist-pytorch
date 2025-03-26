import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Époque {epoch+1}/{epochs} - Perte : {running_loss/len(trainloader):.4f}")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Précision sur le jeu de test : {100 * correct / total:.2f}%")

classes = ['T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
           'Sandale', 'Chemise', 'Baskets', 'Sac', 'Botte']

def show_image(img, label, prediction):
    img = img / 2 + 0.5
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Vrai : {classes[label]}\nPrévu : {classes[prediction]}")
    plt.axis("off")
    plt.show()

dataiter = iter(testloader)
images, labels = next(dataiter)
outputs = model(images)
_, preds = torch.max(outputs, 1)

for i in range(5):
    show_image(images[i], labels[i], preds[i])