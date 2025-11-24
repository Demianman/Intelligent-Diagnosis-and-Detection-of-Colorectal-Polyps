import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset

def q6():

    class myMultiLayerPerceptron_TahnActivation(nn.Module):
        def __init__(self,input_dim,output_dim):
            super().__init__()
            self.sequential = nn.Sequential(  # here we stack multiple layers together
                nn.Linear(input_dim,64),
                nn.Tanh(), # Using Tanh activation!
                nn.Linear(64,32),
                nn.Tanh(),
                nn.Linear(32,20),
                nn.Tanh(),
                nn.Linear(20,20),
                nn.Tanh(),
                nn.Linear(20,20),
                nn.ReLU(),
                nn.Linear(20,output_dim)
            )
        def forward(self,x):
            y = self.sequential(x)
            return y

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['WORLD_SIZE'] = str(world_size) # 确保 StreamingDataset 能读到
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size) # 假设单机
    os.environ['RANK'] = str(rank) # 确保 StreamingDataset 能读到

    dist.init_process_group(backend='gloo') # 'gloo' 是用于 CPU 的后端

    lr = .005
    batch_size = 64
    N_epochs = 10
    in_dim = 113
    mymodel = myMultiLayerPerceptron_TahnActivation(in_dim, 2)

    mymodel = DDP(mymodel)

    loss_fun = nn.CrossEntropyLoss()

    train_dataset_scalable = StreamingDataset(local="/Users/wuyiman/Downloads/NSL-KDD/train_mds", batch_size=batch_size, shuffle=True)


    train_dataloader = DataLoader(train_dataset_scalable, batch_size=batch_size, num_workers=1)

    optimizer = torch.optim.Adam(mymodel.parameters(),
                                 lr=lr)  # this line creates a optimizer, and we tell optimizer we are optimizing the parameters in mymodel

    losses = []  # training losses of each epoch
    accuracies = []  # training accuracies of each epoch


    for epoch in range(N_epochs):
        # Train loop
        batch_loss = []  # keep a list of losses for different batches in this epoch
        batch_accuracy = []  # keep a list of accuracies for different batches in this epoch
        for batch_samples in train_dataloader:
            x_batch = batch_samples['features']
            y_batch = batch_samples['outcome']
            # pass input data to get the prediction outputs by the current model
            prediction_score = mymodel(x_batch)

            # compute the cross entropy loss. Note that the first input to the loss_func should be the predicted scores (not probabilities), and the second input should be class labels as integers
            loss = loss_fun(prediction_score, y_batch)

            # compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # update parameters with optimizer step
            optimizer.step()

            # append the loss of this batch to the batch_loss list
            batch_loss.append(loss.detach().numpy())

            # You can also compute other metrics (accuracy) for this batch here
            prediction_label = torch.argmax(prediction_score.detach(), dim=1).numpy()
            batch_accuracy.append(np.sum(prediction_label == y_batch.numpy()) / x_batch.shape[0])


        losses.append(np.mean(np.array(batch_loss)))

        accuracies.append(np.mean(np.array(batch_accuracy)))

        if rank == 0:
            print(f"Epoch = {epoch}, train_loss={losses[-1]:.4f}, Train accuracy = {accuracies[-1] * 100:.2f}%")

    dist.destroy_process_group()



if __name__ == "__main__":
    q6()