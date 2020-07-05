# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

#출처: https://hyeonnii.tistory.com/244 [From the bottom]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(111)
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    pass
    
# MNIST DATASET
mnist_train = dsets.MNIST(root='MNIST_data/', train=True , transform=transforms.ToTensor(), download=True)
mnist_test  = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

# parameters
epochs = 15
batch_size = 100
learning_rate = 0.1
dropout_probability = 0.2

# dataloader
data_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# model
linear1 = torch.nn.Linear(28*28, 256, bias=True)
linear2 = torch.nn.Linear(256,256, bias=True)
linear3 = torch.nn.Linear(256,10, bias=True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=dropout_probability)

torch.nn.init.normal_(linear1.weight)
torch.nn.init.normal_(linear2.weight)
torch.nn.init.normal_(linear3.weight)

# 마지막에 relu 추가하지 않는 이유는 cross-entropy에서 softmax를 사용하기 때문.
model = torch.nn.Sequential(linear1, relu, dropout, linear2, relu, dropout, linear3).to(device)
loss_function = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train() # dropout을 사용하겠다는 의미.
for epoch in range(epochs):
    avg_loss = 0
    batch_count = len(data_loader)
    
    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)
        
        prediction = model(X)
        loss = loss_function(prediction, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += (loss / batch_count)
        pass
    
    print(epoch, avg_loss.item())
    pass
    
# gradient update를 하지않겠다는 의미.
with torch.no_grad():
    model.eval() # dropout을 사용하지 않겠다는 의미.
    
    # test dataset의 데이터 형태를 (batch*784)로 바꿔줌.
    X_test = mnist_test.data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.targets.to(device) prediction = model(X_test)
    
    # 각 배치별로 가장 높은 가능성의 숫자 클래스를 뽑아줌.
    predicted_classes = torch.argmax(prediction, 1)
    correct_count = (predicted_classes == Y_test)
    
    # 맞는 개수의 평균을 내면 정확도가 나옴.
    accuracy = correct_count.float().mean()
    print(accuracy.item())
    
    # 하나의 그림만 뽑아서 보기 위해 랜덤으로 인덱스 설정.
    r = random.randint(1, len(Y_test)-1)
    single_x = mnist_test.data[r:r+1].view(-1, 28*28).float().to(device)
    single_y = mnist_test.targets[r:r+1].to(device)
    
    prediction = model(single_x)
    print(single_y.item())
    print(torch.argmax(prediction,1).item())
    
    plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap='Greys', interpolation='nearest')
    plt.show()
    pass
