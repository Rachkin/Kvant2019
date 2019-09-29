import torch
import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

#############################################
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
device = torch.device('cuda:0')

def train(net, X_train, y_train, X_test, y_test, bestnet):
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
    
    batch_size = 100
    
    best_accuracy = 0.0

    test_accuracy_history = []
    test_loss_history = []

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for epoch in range(10):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            net.train()

            batch_indexes = order[start_index:start_index+batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)
            
            preds = net.forward(X_batch) 

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()
            
            if start_index == 4:
                break

        net.eval()
        with torch.no_grad():
            test_preds = net.forward(X_test)
            test_loss_history.append(loss(test_preds, y_test).data.cpu())
    
            accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
            test_accuracy_history.append(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_net = net
    
            print(accuracy)
            print(test_loss_history[-1])
            print("==============")
    print('---------------')
    return test_accuracy_history, test_loss_history
##############################################
accuracies = {}
losses = {}

net = HarukaNet()

best_net = net

accuracies['='], losses['='] = \
    train(net, 
          X_train, y_train, X_test, y_test, best_net)

import matplotlib.pyplot as plt
for experiment_id in accuracies.keys():
    plt.plot(accuracies[experiment_id], label=experiment_id)
plt.legend()
plt.title('Validation Accuracy');
#print("1")
########################################
#X_valid = X_valid.to(device)
#y_valid = y_valid.to(device)

best_net.eval()
with torch.no_grad():
    test_preds = best_net.forward(X_valid)
    accuracy = (test_preds.argmax(dim=1) == y_valid).float().mean().data.cpu()
    print(accuracy)
    
    
torch.save(best_net.state_dict(), 'my_models/HN_c32_PP_r5.pt')