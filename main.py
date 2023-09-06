import time
from dataloader import DataGenerator,generate_file,sliding_window,norm,load_adj,origin_window
from model.model import TemporalGNN
import argparse
from envs import get_env
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda")

def get_parameters():
    parser = argparse.ArgumentParser()
    # Whole situation, SWMM
    parser.add_argument('--env', default='shunqing', help='drainage scenarios')
    parser.add_argument('--data_dir', type=str, default='./envs/data/shunqing/', help='the sampling data file')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of training events')
    parser.add_argument('--processes', type=int, default=5, help='number of simulation processes')
    parser.add_argument('--act', default=True, help='if the environment contains control actions')
    parser.add_argument('--seq', default=True, help='seq')
    parser.add_argument('--hmax', default=np.array([1.5 for _ in range(105, 4)]), help='hmax')
    parser.add_argument('--seq_in', type=int, default=12, help='==n_his')
    parser.add_argument('--seq_out', type=int, default=12,
                        help='the number of time interval for predcition, default as 12,==n_pred')
    parser.add_argument('--simulate', default=False, help='if simulate rainfall events for training data')
    parser.add_argument('--result_dir',type=str,default='./results/',help='the test results')

    #train 
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--shuffle',default=True)



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
    # edge_index = args.edges
    # edge_index = np.array(edge_index) # [2,n]
    edge_index = load_adj()

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
    #选取train和test降雨事件
    train_ids = np.random.choice(np.arange(n_events), int(n_events * args.ratio), replace=False)
    test_ids = [ev for ev in range(n_events) if ev not in train_ids]

    X,B,Y = dG.prepare(seq,train_ids) #Y不要，B是某点降雨序列，也不要 
    Xt,Bt,Yt = dG.prepare(seq,test_ids)


    #删掉第二维度, axis =2 , 5>4 (70000,12,105,4)  into  (27399,207,2,12)
    #axis =1,4>3 ,x.shape(70000,105,4),xt.shape(20000,105,4)
    x = np.squeeze(X,axis=1) 
    xt = np.squeeze(Xt,axis=1)
    # y = np.squeeze(Y,axis=2)
    # yt = np.squeeze(Yt,axis=2)

    #norm
    x = norm(x)  #into 105,4,76614
    xt = norm(xt)
    
    # x = np.transpose(x,(0,2,3,1))
    # xt = np.transpose(xt,(0,2,3,1))
    # y = np.transpose(y,(0,2,3,1))
    # yt = np.transpose(yt,(0,2,3,1))

    x = torch.from_numpy(x) #(76614,1,105,4)
    xt = torch.from_numpy(xt)
    # y = torch.from_numpy(y) #(76614,1,105,4)
    # yt = torch.from_numpy(yt)

    x_train, y_train = sliding_window(x, args.seq_in, args.seq_out)
    x_test, y_test = sliding_window(xt, args.seq_in, args.seq_out)

    # x_train, y_train = origin_window(X, args.seq_in, args.seq_out)
    # x_test, y_test = origin_window(Xt, args.seq_in, args.seq_out)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

    #train
    train_input = np.array(x_train) # (27399, 207, 2, 12)
    train_target = np.array(y_train) # (27399, 207, 12)
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(device)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(device)  # (B, N, T)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=args.batch_size, shuffle=args.shuffle,drop_last=True)
    #test
    test_input = np.array(x_test) # (, 207, 2, 12)
    test_target = np.array(y_test) # (, 207, 12)
    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(device)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device)  # (B, N, T)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=args.batch_size, shuffle=args.shuffle,drop_last=True)


    batch_size = args.batch_size
    model = TemporalGNN(node_features=3, periods=args.seq_in, batch_size=batch_size).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

 
    train_loss_list = []
    test_loss_list = []
    for epoch in range(args.epochs):
        step = 0
        loss_list = []
         #y_hat = y_pred    ,  labels = y

        for x, y in tqdm(train_loader):
            y_pred = model(x, edge_index)         # Get model predictions
            loss = loss_fn(y_pred, y) # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step= step+ 1
            loss_list.append(loss.item())
            # if step % 100 == 0 :
            #     print('train_loss:',sum(loss_list)/len(loss_list))

        train_loss = sum(loss_list) / len(loss_list)
        train_loss_list.append(train_loss)
        
        print("Epoch {} - Train Loss: {:.6f}".format(epoch+1, train_loss))
    
        model.eval()
        step = 0
    # Store for analysis
        total_loss = []
        for x, y in test_loader:
            y_pred = model(x, edge_index)
            loss = loss_fn(y_pred, y)
            total_loss.append(loss.item())

        test_loss = sum(total_loss) / len(total_loss)
        test_loss_list.append(test_loss)
        print("Epoch {} - Test Loss: {:.6f}".format(epoch+1, test_loss))

    # print("Epoch {} train loss: {:.6f} Test Loss :{:.6f}".format(epoch+1, train_loss,test_loss))
    plt.figure()
# 绘制平滑后的训练损失曲线
    plt.plot(range(1, args.epochs+1), train_loss_list, label='Train Loss')
    plt.plot(range(1, args.epochs+1), test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss ')
    plt.grid(True)  # 添加网格线
    plt.legend()
    plt.savefig('loss.png')

    













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
