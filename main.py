import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
import yaml
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from script import dataloader, utility, earlystopping
from model import models
from envs import get_env
from script.dataloader import DataGenerator,generate_file,Emulator
from envs.scenario.shunqing import shunqing


# import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='shunqing', help='drainage scenarios')
    parser.add_argument('--simulate', action="store_true", help='if simulate rainfall events for training data')
    parser.add_argument('--data_dir', type=str, default='./envs/data/shunqing/', help='the sampling data file')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of training events')
    parser.add_argument('--processes', type=int, default=1, help='number of simulation processes')

    parser.add_argument('--loss_function', type=str, default='MeanSquaredError', help='Loss function')
    parser.add_argument('--opt', type=str, default='Adam', help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--act', action="store_true", help='if the environment contains control actions')
    parser.add_argument('--seq_in', type=int, default=12,help='==n_his')
    parser.add_argument('--seq_out', type=int, default=3,
                        help='the number of time interval for predcition, default as 3,==n_pred')

    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='shunqing', choices=['metr-la', 'pems-bay', 'shunqing'])

    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3) #kernel_size
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2]) #Chebnet
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv',
                        choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap',
                        choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Ko = args.seq_in - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])  #一个时间点

    return args, device, blocks


def data_preparate(args, device):
    n_vertex = 105

    seq = max(args.seq_in, args.seq_out)

    n_events = int(max(dG.event_id)) + 1
    train_ids = np.random.choice(np.arange(n_events), int(n_events * 0.8), replace=False)
    test_ids = [ev for ev in range(n_events) if ev not in train_ids]

    X_train, B_train, Y_train = dG.prepare(seq, train_ids)
    X_test, B_test, Y_test = dG.prepare(seq, test_ids)

    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(X_train)
    test = zscore.transform(X_test)

    x_train, y_train = dataloader.data_transform(train, args.seq_in, args.seq_out, device)
    x_test, y_test = dataloader.data_transform(test, args.seq_in, args.seq_out, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, test_iter


def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler


def train(loss, args, optimizer, scheduler, es, model, train_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], l_sum / n,  gpu_mem_alloc))


@torch.no_grad()
def test(zscore, loss, model, test_iter, args):
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(
        f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args, device, blocks, = get_parameters()

    env = get_env(args.env)()
    env_args = env.get_args()
    for k, v in env_args.items():
        if k == 'act':
            v = v & args.act
        setattr(args, k, v)


    adj = np.zeros((args.edges.max() + 1, args.edges.max() + 1))
    for u, v in args.edges:
        adj[u, v] += 1

    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()

    dG = DataGenerator(env, args.seq_in, args.seq_out, args.act, args.data_dir)
    events = generate_file(env.config['swmm_input'], env.config['rainfall'])
    if args.simulate:
        dG.generate(events, processes=args.processes, act=args.act)
        dG.save(args.data_dir)
    else:
        dG.load(args.data_dir)


    n_vertex, zscore, train_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(loss, args, optimizer, scheduler, es, model, train_iter)
    test(zscore, loss, model, test_iter, args)
