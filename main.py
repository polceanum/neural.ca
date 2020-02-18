from __future__ import print_function
import os
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import NCA
from emoji_loader import load_emoji
from visualize import saveVideo

def train(args, model, device, train_target, optimizer, iter):
    model.train()

    # create initial states
    batch_data = torch.zeros((
            args.batch_size,      # batch size
            train_target.size(0), # width
            train_target.size(1), # height
            args.cell_size)       # cell state size
        ).float()
    batch_data = batch_data.to(args.__device__) # send to device

    # batch_data[:, 20, 20, 3:] = 0.2 # set seed in the middle with all except RGB = 1
    
    optimizer.zero_grad()
    output, history = model(batch_data, steps=args.ca_steps)
    loss = 0
    if not args.persistence:
        loss = F.mse_loss(output[:, :, :, :4], train_target.unsqueeze(0))
    else:
        nb = 0
        for i in range(len(history)):
            out = history[i]
            if i >= args.ca_steps//2 and i % 4 == 0:
                loss += F.mse_loss(out[:, :, :, :4], train_target.unsqueeze(0))
                nb += 1
        loss /= nb
    loss.backward()
    optimizer.step()

    if iter % args.log_interval == 0:
        saveVideo(history, filename='./videos/test_'+str(iter)+'.gif')
    
    print('Train iter: {} \tLoss: {:.6f}'.format(iter, loss.item()))
    

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Neural Cellular Automata')
    parser.add_argument('--cell-size', type=int, default=16, metavar='E',
                        help='cell state size (default: 16)')
    parser.add_argument('--emoji', type=int, default=0, metavar='ST',
                        help='index of training emoji (default: 0)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='number of epochs to train (default: 100000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load-model-iter', type=int, default=0, metavar='S',
                        help='load model path (default: None)')
    parser.add_argument('--ca-steps', type=int, default=96, metavar='CAS',
                        help='load model path (default: None)')
    parser.add_argument('--interim-layers', type=int, default=0, metavar='L',
                        help='load model path (default: 0)')
    parser.add_argument('--persistence', action='store_true', default=False,
                        help='Use persistence on the second half of CA steps')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # create save and video folders
    if not os.path.exists('./videos'):
        os.makedirs('./videos')
    if not os.path.exists('./save'):
        os.makedirs('./save')

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    args.__device__ = device

    emoji = load_emoji('./data/emoji.png', args.emoji)
    
    # convert to torch tensor and send to device
    emoji = torch.from_numpy(emoji).float().to(device)

    print('training target size:', emoji.size())

    # create model
    model = NCA(cell_state_size=args.cell_size, hidden_size=128, num_layers=args.interim_layers, device=device).to(device)
    if args.load_model_iter > 0:
        model.load_state_dict(torch.load("save/nca_model_L="+str(args.interim_layers)+"_"+str(args.load_model_iter)+".pt"))
        print('Loaded model from:', "save/nca_model_L="+str(args.load_model_iter)+".pt")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    for epoch in range(args.load_model_iter+1, args.epochs + 1):
        train(args, model, device, emoji, optimizer, epoch)

        if epoch % args.log_interval == 0:
            torch.save(model.state_dict(), "save/nca_model_L="+str(args.interim_layers)+"_"+str(epoch)+".pt")