import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train(model, device, train_loader,optimizer):
    
    model.train()
    
    loss_fn = nn.CrossEntropyLoss()#becasue o softmax models
    
    
    #variables for plots
    train_acc = 0
    
    loss_items = []
    
    acc_items = []
    
    current_epoch = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        pred = model(data)
        
        loss = loss_fn(pred, target)
        
        loss.backward()
        
        optimizer.step()
        
        pred_num = torch.argmax(pred,dim=1)#to pick the actual predict number
        
        train_acc += torch.sum(pred_num == target)
        
        
        #for plots
        loss_items.append(loss.item())
        current_epoch = batch_idx
        acc_total = train_acc / len(train_loader.dataset)
        acc_items.append(acc_total)
        
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {batch_idx} \n Loss: {loss.item()} \n Acurracy: {acc_total}')
            
    return loss_items,current_epoch, acc_items



def test(model, device, test_loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    
    loss_fn = nn.CrossEntropyLoss()#to test accuracy and erros
    
    with torch.no_grad():
        for data, target in test_loader:
            
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            test_loss += loss_fn(output, target).item() 
            
            pred = output.argmax(dim=1, keepdim=True)#predicted number
             
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss} \n Acurracy {correct}')
    
def loss_graph(loss_items,final_epoch):
    x = list(range(1, final_epoch+2))
    plt.figure(figsize=(5, 2))
    plt.plot(x, loss_items, label='Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Model')
    plt.legend()
    plt.show()
    
def accuracy_graph(acc_items,final_epoch):
    x = list(range(1, final_epoch+2))
    plt.figure(figsize=(5, 2))
    plt.plot(x, acc_items, label='Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Acuuracy in %")
    plt.title('Model')
    plt.legend()
    plt.show()
    