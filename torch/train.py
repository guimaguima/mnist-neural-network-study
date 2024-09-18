import torch
import torch.nn as nn
import torch.optim as optim


def train(model, device, train_loader,optimizer):
    
    model.train()
    
    loss_fn = nn.CrossEntropyLoss()
    
    train_acc = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        pred = model(data)
        
        loss = loss_fn(pred, target)
        
        loss.backward()
        
        optimizer.step()
        
        pred_num = torch.argmax(pred,dim=1)
        
        train_acc += torch.sum(pred_num == target)
        
        if batch_idx % 10 == 0:
            acc_total = train_acc / len(train_loader.dataset)
            print(f'Train Epoch: {batch_idx} \n Loss: {loss.item()} \n Acurracy: {acc_total}')



def test(model, device, test_loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            test_loss += loss_fn(output, target).item() 
            
            pred = output.argmax(dim=1, keepdim=True)
             
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss} \n Acurracy {correct}')