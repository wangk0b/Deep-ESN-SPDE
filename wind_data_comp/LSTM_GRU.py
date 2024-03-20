import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim
import time
import numpy as np
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
                                            
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
                                                                        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        return out, h
                                                                                                            
    def init_hidden(self, batch_size = None):
        weight = next(self.parameters()).data
        if batch_size == None:
            hidden = weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device)
        else:
            hidden = weight.new(self.n_layers, batch_size,self.hidden_dim).zero_().to(self.device)
        return hidden






class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
     super(LSTMNet, self).__init__()
     self.hidden_dim = hidden_dim
     self.n_layers = n_layers
     self.device = device
                                            
     self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
     self.fc = nn.Linear(hidden_dim, output_dim)
     self.relu = nn.ReLU()
                                                                            
    def forward(self, x, h):
       out, h = self.lstm(x, h)
       out = self.fc(self.relu(out))
       return out, h
                                                                                                            
    def init_hidden(self, batch_size = None):
        weight = next(self.parameters()).data
        if batch_size == None:
          hidden = (weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device),weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device))
        else:
            hidden = (weight.new(self.n_layers, batch_size,self.hidden_dim).zero_().to(self.device),weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden









def train(train_loader, learn_rate, hidden_dim, EPOCHS, device, batch_size = None, model_type="GRU"):
        
    # Setting common hyperparameters
    input_dim = 3173
    output_dim = 3173
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
      model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device)
    else:
      model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers, device)
    model.to(device)
                                                            
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
                                                                            
    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
      start_time = time.perf_counter()
      h = model.init_hidden()
      avg_loss = 0
      counter = 0
      for x, label in train_loader:
         counter += 1
         if model_type == "GRU":
             h = h.data                                                                                                          
         else:
             h = tuple([e.data for e in h])                                                                                        
         model.zero_grad()
         out, h = model(x.to(device).float(), h)
         loss = criterion(out, label.to(device).float())
         loss.backward()
         optimizer.step()
         avg_loss += loss.item()
         #if counter%200 == 0:
         #    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
      current_time = time.perf_counter() 
      print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
      #print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
      epoch_times.append(current_time-start_time)
      #print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model
                                                                                                                                                                                            
def evaluate(model, test_loader, device, batch_size = None):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    for test_x,test_y in test_loader:

     h = model.init_hidden()

     out, h = model(test_x.to(device), h)
     outputs.append(out.cpu().detach().numpy().reshape(-1))
     targets.append(test_y.numpy().reshape(-1))
  
    #print("Evaluation Time: {}".format(str(time.perf_counter()-start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
         sMAPE += np.mean((outputs[i]-targets[i])**2)/len(outputs)
    print("sMAPE: {}".format(sMAPE))        
    return outputs, targets, sMAPE


