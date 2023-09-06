import time
from dataloader import DataGenerator,generate_file
import argparse
from envs import get_env
import numpy as np
import os
import torch
device = torch.device("cuda")

def setxy(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
    indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
    features, target = [], []
    for i, j in indices:
        features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
        target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

    return features ,target

def data_produce(data,seq_in,device):
    '''
    对数据进行处理
    '''
    
    num = data.shape[0]
    # 初始化输出目标 y
    y = np.zeros((num, seq_in, 105))

    for i in range(num):
        for j in range(seq_in):
        # 从输入 x 中提取滑动窗口对应的片段，并赋值给目标 y
            y[i, j] = data[i, j, :, 0]

# 将 x 和 y 转换为 PyTorch 的 Tensor 格式，并转移到指定设备（如 GPU）

    y = torch.Tensor(y)

    return  torch.Tensor(y)



def z_score_normalize(x):
    mean = np.mean(x,axis = 1,keepdims=True) #  (B,channel,TS,N)
    std = np.std(x,axis =1,keepdims=True)
    std[std == 0] = 1e-8
    x_normalized = (x - mean) / std
    return x_normalized


def get_parameters():
    parser = argparse.ArgumentParser()
    # Whole situation, SWMM
    parser.add_argument('--env', default='shunqing', help='drainage scenarios')
    parser.add_argument('--data_dir', type=str, default='./envs/data/shunqing/', help='the sampling data file')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of training events')
    parser.add_argument('--processes', type=int, default=5, help='number of simulation processes')
    parser.add_argument('--act', default=True, help='if the environment contains control actions')
    parser.add_argument('--seq', default=True, help='seq')
    parser.add_argument('--seq_in', type=int, default=12, help='==n_his')
    parser.add_argument('--seq_out', type=int, default=12,
                        help='the number of time interval for predcition, default as 1,==n_pred')
    parser.add_argument('--hmax', default=np.array([1.5 for _ in range(105, 4)]), help='hmax')
    parser.add_argument('--res_dir', type=str, default='./figure', help='result')
    parser.add_argument('--simulate', default=False, help='if simulate rainfall events for training data')
    parser.add_argument('--result_dir',type=str,default='./results/',help='the test results')
    args = parser.parse_args()
    print('configs: {}'.format(args))
    return args

if __name__ == "__main__":
    args = get_parameters()
    env = get_env(args.env)()
    env_args = env.get_args()
    for k, v in env_args.items():
        if k == 'act':
            v = v & args.act
        setattr(args, k, v)

    #adj
    edge_index = args.edges
    edge_index = np.array(edge_index)

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
    train_ids = np.random.choice(np.arange(n_events), int(n_events * args.ratio), replace=False)
    test_ids = [ev for ev in range(n_events) if ev not in train_ids]

    X,B,Y = dG.prepare(seq,train_ids)
    Xt,Bt,Yt = dG.prepare(seq,test_ids)

    x = np.squeeze(X,axis=2) #删掉第二维度,5>4 ,x.shape(70000,12,105,4)
    xt = np.squeeze(Xt,axis=2)
    # x = X[..., 3:]# 输入 x 包含前三个特征（h、q_us 和 q_ds）  (78000,12,105,3)
    # b = X[..., 3:4]  # 固定序列特征 r  (78000,12,105,1)
    #     # 提取目标特征
    # y = X[..., :3]  # 目标 y 包含下一个时间步的三个特征（h、q_us 和 q_ds）(78000,12,105)
    # x_input = [x,b]   #list [(78000,12,105,3)(78000,12,105,1)]

    # xt = Xt[..., [0, 1, 2]]# 输入 x 包含前三个特征（h、q_us 和 q_ds）  (78000,12,105,3)
    # bt = Xt[..., 3]  # 固定序列特征 r  (78000,12,105,1)
    #     # 提取目标特征
    # yt = Xt[..., [0, 1, 2]]  # 目标 y 包含下一个时间步的三个特征（h、q_us 和 q_ds）(78000,12,105)
    
    #zscore
    x = z_score_normalize(x)
    xt = z_score_normalize(xt)
    # 生成y
    y = data_produce(x,args.seq_in,device)
    yt = data_produce(xt,args.seq_in,device)

    # 将数组保存为.npy文件
    np.save(os.path.join(args.result_dir,'x.npy'), x)
    np.save(os.path.join(args.result_dir,'xt.npy'), xt)
    np.save(os.path.join(args.result_dir,'y.npy'), y)
    np.save(os.path.join(args.result_dir,'yt.npy'), yt)
    np.save(os.path.join(args.result_dir,'edgeidx.npy'), edge_index)
    np.save(os.path.join(args.result_dir,'train_id.npy'),np.array(train_ids))
    np.save(os.path.join(args.result_dir,'test_id.npy'),np.array(test_ids))



   







