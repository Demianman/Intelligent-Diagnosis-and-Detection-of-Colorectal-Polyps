import os
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from streaming import StreamingDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import streaming.base.util as util

class mymodel(nn.Module):
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

def main():
    util.clean_stale_shared_memory()
    print("when showing File exists, do the cleaning.")
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--world_size', default=1,
                        type=int)
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='cuda device ID. -1 represents cpu')
    parser.add_argument('-r', '--rank', default=0, type=int,
                        help='rank of the process')
    parser.add_argument('--epochs', default=2, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--master_addr', default="127.0.0.1", type=str,
                        help='address of the machine that runs the master process (process with rank 0)')
    parser.add_argument('--master_port', default="8891", type=str,
                        help='port of the machine that runs the master process (process with rank 0)')
    args = parser.parse_args()
    print(f"hello! Process # {args.rank} is being initialized! Awaiting all processes to join. ")
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(args.world_size)

    print(args)
    train(args)

def train(args):
    # Setup the communication with other processes
    #print(f"hello! Process # {args.rank} is being initialized! Awaiting all processes to join. ")
    #os.environ['MASTER_ADDR'] = args.master_addr
    #os.environ['MASTER_PORT'] = args.master_port
    # os.environ['RANK'] = str(args.rank)
    dist.init_process_group(
    	backend='gloo' if args.gpu == -1 else 'nccl',
   		init_method='env://',
    	world_size=args.world_size,
    	rank=args.rank
    )
    print(f"hello! Process # {args.rank} is running!")

    # Loading a slice of the dataset
    batch_size = 64
    train_dataset = StreamingDataset(local="/Users/wuyiman/Downloads/NSL-KDD/train_mds",
                                     batch_size=batch_size,
                                     shuffle=True)

    # Setting up device according to args.gpu. if args.gpu = -1, then device is cpu; otherwise, the device will be a cuda device.
    if args.gpu >=0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')


    # Create model and WRAP IT WITH DistributedDataParallel
    model = mymodel(input_dim=113, output_dim=2)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model)

    # Setup the loss, optimizer, loader
    lr = 0.005
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size)


    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            images = batch['features'].to(device)
            labels = batch['outcome'].to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                   )

if __name__ == '__main__':
    main()
