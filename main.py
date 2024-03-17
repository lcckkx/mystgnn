import time
import random

from dataloader import *
from model.models import GNN,STGCN
import argparse
from envs import get_env
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.cuda.amp import autocast, GradScaler


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
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    # Whole situation, SWMM
    parser.add_argument('--env', default='shunqing', help='drainage scenarios')
    parser.add_argument('--data_dir', type=str, default='./envs/data/shunqing/', help='the sampling data file')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of training events')
    parser.add_argument('--processes', type=int, default=5, help='number of simulation processes')
    parser.add_argument('--act', default=True, help='if the environment contains control actions')
    parser.add_argument('--seq', default=True, help='seq')

    parser.add_argument('--seq_in', type=int, default=5, help='==n_his')
    parser.add_argument('--seq_out', type=int, default=5,
                        help='the number of time interval for predcition, default as 5,==n_pred')
    parser.add_argument('--simulate', default=False, help='if simulate rainfall events for training data')

    parser.add_argument('--result_dir',type=str,default='./results/shunqing/',help='the test results')
    #train
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN','STGCN'])
    parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')                                                                   
    parser.add_argument('--epochs', type=int, default=2, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=128)   
    parser.add_argument('--weight_decay', type=float, default=1e-5)         
    parser.add_argument('--shuffle',default=True)

    args = parser.parse_args()
    print('configs: {}'.format(args))

    set_env(args.seed)

    return args

def prepare_model(args):
    loss_fn = torch.nn.MSELoss()
    if args.model == 'GCN':
        model = GNN().to(device)
    else:
        model = STGCN().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()
    return loss_fn,model,optimizer,scaler

#TODO dataloader X  Y  B, formal time later time 

def dataset(args,seq,n_events):  
    '''
    adj
    edge_index = args.edges
    edge_index = np.array(edge_index) # [2,n]
    '''
    edge_index = load_adj()# [2,n]

    # train_ids = np.random.choice(np.arange(n_events), int(n_events * args.ratio), replace=False)
    # test_ids = [ev for ev in range(n_events) if ev not in train_ids]


    train_ids = [1,2]
    test_ids = [3]   
    
    X,B,Y = dG.prepare(seq,train_ids) #(20000,12,113,3)
    Xt,Bt,Yt = dG.prepare(seq,test_ids)
    # xt,bt,yt = torch.from_numpy(xt).type(torch.FloatTensor),torch.from_numpy(bt).type(torch.FloatTensor),torch.from_numpy(yt).type(torch.FloatTensor)

    # Y = dG.prepare_Y(seq,train_ids)#(20000,113,3)
    # Yt = dG.prepare_Y(seq,test_ids)

    x ,b, y = minmax_norm(X),minmax_norm(B),minmax_norm(Y)#x_mins, x_maxs  into b,t,n,f
    xt,bt= minmax_norm(Xt),minmax_norm(Bt)

    yt, min,max = minmax_norm_y(Yt)
    
    x_train ,y_train= torch.cat((x,b),dim=3).to(device), y.to(device)
    x_test,y_test = torch.cat((xt,bt),dim=3).to(device), yt.to(device)

    #train
    train_dataset_new = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=args.batch_size, shuffle=args.shuffle)
    #test
    test_dataset_new = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=args.batch_size, shuffle=args.shuffle)

    return edge_index,train_loader,test_loader ,min,max


def train(args,loss_fn,model,optimizer,train_loader,test_loader,edge_index,scaler):
    total_train_loss = []
    total_val_loss = []
    for epoch in range(args.epochs):
        train_loss_epoch = []

        #train
        model.train()
        for x, y in tqdm(train_loader): #x（B,N,T,F）, y (20000,113,3) ncols=10 
            with autocast():
                # y_true = y[: ,: , : ,:3] #32,10,113,3
                y_pred = model(x, edge_index)     #32,105,3,10 ,     10,3
                loss = loss_fn(y_pred, y)  # Mean squared error
                l2_reg = torch.tensor(0.0,device=args.device)
                for param in model.parameters():
                    l2_reg += torch.norm(param, p=2)**2
                loss += args.weight_decay * l2_reg
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss_epoch.append(loss.item())
        train_loss = sum(train_loss_epoch) / len(train_loss_epoch)
        total_train_loss.append(train_loss)

        #val
        avg_val_loss = val(model,test_loader,edge_index,loss_fn)
        total_val_loss.append(avg_val_loss)

        print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.6f} Val Loss:{avg_val_loss:.6f} ')#- MAE :{MAE:.6f}  - WMAPE : {WMAPE:.6f}- R2 Score: {average_r2:.6f

    torch.save(model.state_dict(), os.path.join(args.result_dir, 'model.pth'))#save

    plt.figure()
    plt.plot(range(1, len(total_train_loss) + 1), total_train_loss, label='Train Loss')
    plt.plot(range(1, len(total_val_loss) + 1), total_val_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.savefig(os.path.join(args.result_dir, 'loss.png'), dpi=300)
    # plt.ylim(bottom=0) 
    np.save(os.path.join(args.result_dir,'train_losses.npy'),np.array(total_train_loss))
    np.save(os.path.join(args.result_dir,'val_losses.npy'),np.array(total_val_loss))


    

@torch.no_grad()
def val(model, test_loader,edge_index,loss_fn):
    model.eval()
    val_loss = []
    for x, y in test_loader:
        with autocast():
            y_pred = model(x, edge_index)
            loss = loss_fn(y_pred, y) 
        val_loss.append(loss.item())
    average_val_loss = sum(val_loss) / len(val_loss)

    return average_val_loss

def test(model,test_loader,edge_index,loss_fn,threshold):

        Y_true = []
        Y_pred = []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                y_true = []
                y_pred = []
                with autocast():
                    y = inverse_y(y,min,max)  #32,10,113,3
                    ypred = inverse_y(model(x,edge_index),min,max)
                y_true.append(y)
                y_pred.append(ypred)

                Y_true.extend(y_true)
                Y_pred.extend(y_pred)
        Y_true = np.concatenate(Y_true, axis=0)#32,10,105,3
        Y_pred = np.concatenate(Y_pred, axis=0)#32,10,105,3

        np.save(os.path.join(args.result_dir,'test_pred.npy'),np.array(Y_pred))
        np.save(os.path.join(args.result_dir,'test_true.npy'),np.array(Y_true))

        r2 = r2_score(Y_true.flatten(), Y_pred.flatten())
        Y_true = Y_true.reshape(-1)
        Y_pred = Y_pred.reshape(-1)
        mse = mean_squared_error(Y_true, Y_pred)
        mae = mean_absolute_error(Y_true, Y_pred)
        acc = np.mean(np.abs(Y_true - Y_pred) <= threshold)

        np.save(os.path.join(args.result_dir, 'r2_score.npy'), np.array(r2))
        np.save(os.path.join(args.result_dir, 'mse.npy'), np.array(mse))
        np.save(os.path.join(args.result_dir, 'mae.npy'), np.array(mae))
        np.save(os.path.join(args.result_dir, 'accuracy.npy'), np.array(acc))
        
        print(f'R2:{r2:.6f} - MSE: {mse:.4f} -  MAE:{mae:.4f} - Accuracy: {acc:.4f} ')

if __name__ == "__main__":
    args = get_parameters()
    env = get_env(args.env)()
    env_args = env.get_args()
    for k, v in env_args.items():
        if k == 'act':
            v = v & args.act
        setattr(args, k, v)

    swmm_start_time = time.time()
    dG = DataGenerator(env, args.seq_in, args.seq_out, args.act, args.data_dir)
    events = generate_file(env.config['swmm_input'], env.config['rainfall'])
    if args.simulate:
        dG.generate(events, processes=args.processes, act=args.act, seq=args.seq)
        dG.save(args.data_dir)
    else:
        dG.load(args.data_dir)

    seq = max(args.seq_in, args.seq_out)

    n_events = int(max(dG.event_id)) + 1
    device = args.device
    threshold = 0.1
    edge_index,train_loader,test_loader,min,max = dataset(args,seq,n_events)    #test_loader
    loss_fn,model,optimizer,scaler = prepare_model(args)


    train(args,loss_fn,model,optimizer,train_loader,test_loader,edge_index,scaler)
    test(model,test_loader,edge_index,loss_fn,threshold)
    
    #model
    for event in events:
            name = os.path.basename(event).strip('.inp')
            if os.path.exists(os.path.join(args.result_dir,name + '_states.npy')):
                states = np.load(os.path.join(args.result_dir,name + '_states.npy'))
                perfs = np.load(os.path.join(args.result_dir,name + '_perfs.npy'))
            else:
                t0 = time.time()
                states,perfs,settings = dG.simulate(event,seq,act=args.act)
                print("{} Simulation time: {}".format(name,time.time()-t0))
                np.save(os.path.join(args.result_dir,name + '_states.npy'),states)
                np.save(os.path.join(args.result_dir,name + '_perfs.npy'),perfs)
            states[...,1] = states[...,1] - states[...,-1]
            r,true = states[args.seq_out:,...,-1:],states[args.seq_out:,...,:-1]

            states = states[...,:-1]
            t0 = time.time()
            states = states[:-args.seq_out]

            model.eval()
            with torch.no_grad():
                states = torch.from_numpy(states).to(device,dtype=torch.double)
                pred = model(states,edge_index).detach().numpy()
            
            print("{} Emulation time: {}".format(name,time.time()-t0))
            true = np.concatenate([true,perfs[args.seq_out:,...]],axis=-1)  # cumflooding in performance
            np.save(os.path.join(args.result_dir,name + '_pred.npy'),pred)
            np.save(os.path.join(args.result_dir,name + '_true.npy'),true)


    













    # x = X[..., 3:]# 输入 x 包含前三个特征（h、q_us 和 q_ds）  (78000,12,105,3)
    # b = X[..., 3:4]  # 固定序列特征 r  (78000,12,105,1)
    #     # 提取目标特征
    # y = X[..., :3]  # 目标 y 包含下一个时间步的三个特征（h、q_us 和 q_ds）(78000,12,105)
    # x_input = [x,b]   #list [(78000,12,105,3)(78000,12,105,1)]

    # xt = Xt[..., [0, 1, 2]]# 输入 x 包含前三个特征（h、q_us 和 q_ds）  (78000,12,105,3)
    # bt = Xt[..., 3]  # 固定序列特征 r  (78000,12,105,1)
    #     # 提取目标特征
    # yt = Xt[..., [0, 1, 2]]  # 目标 y 包含下一个时间步的三个特征（h、q_us 和 q_ds）(78000,12,105)
    
    # #zscore
    # x = z_score_normalize(x)
    # xt = z_score_normalize(xt)
    # # 生成y
    # y = data_produce(x,args.seq_in,device)
    # yt = data_produce(xt,args.seq_in,device)

    # # 将数组保存为.npy文件
    # np.save(os.path.join(args.result_dir,'x.npy'), x)
    # np.save(os.path.join(args.result_dir,'xt.npy'), xt)
    # np.save(os.path.join(args.result_dir,'y.npy'), y)
    # np.save(os.path.join(args.result_dir,'yt.npy'), yt)
    # np.save(os.path.join(args.result_dir,'edgeidx.npy'), edge_index)
    # np.save(os.path.join(args.result_dir,'train_id.npy'),np.array(train_ids))
    # np.save(os.path.join(args.result_dir,'test_id.npy'),np.array(test_ids))
    '''
    根据提供的代码，看起来你使用了标准化（Standardization）的方法对数据进行了处理。为了进行反向缩放，你需要知道原始数据的均值和标准差。

反向缩放的步骤如下：

计算原始数据的均值和标准差。你可以使用与标准化时相同的方法计算均值和标准差。
python
original_means = means  # 原始数据的均值
original_stds = stds  # 原始数据的标准差
执行反向缩放。根据标准化公式，将标准化后的数据乘以标准差，并加上均值。
python
X_scaled = X  # 标准化后的数据
X_original = X_scaled * original_stds.reshape(1, -1, 1) + original_means.reshape(1, -1, 1)
最后，X_original 就是经过反向缩放后的原始数据。

请注意，反向缩放后的数据类型可能仍然是 np.float32，你可以根据需要进行转换。另外，确保在进行反向缩放之前，original_means 和 original_stds 的值是正确的。
    
    '''
