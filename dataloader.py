import numpy as np
from datetime import datetime
from swmm_api import read_inp_file
import multiprocessing as mp
import os
import torch

device = torch.device("cuda")
HERE = os.path.dirname(__file__)

def minmax_norm(X):
    #X (20000,12,113,3) y(20000,113,3)
    X = X.astype(np.float32)
    mins = np.min(X, axis=(0, 1))
    maxs = np.max(X, axis=(0, 1))
    X = (X - mins) / (maxs - mins+1e-6)
    X1 = X[: ,:,:105, :]
    return torch.from_numpy(X1).type(torch.FloatTensor)

def minmax_norm_y(y):
    y = y.astype(np.float32)#y(20000,113,3)
    mins = np.min(y, axis=(0,1))
    maxs = np.max(y, axis=(0,1))
    y = (y - mins) / (maxs - mins+1e-6)
    y1 = y[:,:,:105,:]
    return torch.from_numpy(y1).type(torch.FloatTensor), mins, maxs

def inverse_y(y_norm, mins, maxs):
    y_norm = y_norm.detach().cpu().numpy()
    y_denorm = y_norm * (maxs - mins + 1e-6) + mins
    return y_denorm

def load_adj():
    #[105,105]
    adj = np.load('./data/edgeidx.npy')
    static_edge_index = torch.tensor(adj, dtype=torch.int)
    edge_index = static_edge_index.to(torch.long).to(device)

    return edge_index

# suffix: bpswmm
# filedir: ./envs/network/shunqing/
def generate_file(file, arg):
    inp = read_inp_file(file)
    for k, v in inp.TIMESERIES.items():
        if k.startswith(arg['suffix']):
            dura = v.data[-1][0] - v.data[0][0]
            st = (inp.OPTIONS['START_DATE'], inp.OPTIONS['START_TIME'])
            st = datetime(st[0].year, st[0].month, st[0].day, st[1].hour, st[1].minute, st[1].second)
            et = (st + dura)
            inp.OPTIONS['END_DATE'], inp.OPTIONS['END_TIME'] = et.date(), et.time()
            inp.RAINGAGES['RainGage'].Timeseries = k
            if not os.path.exists(arg['filedir'] + k + '.inp'):
                inp.write_file(arg['filedir'] + k + '.inp')
    events = [arg['filedir'] + k + '.inp' for k in inp.TIMESERIES if k.startswith(arg['suffix'])]
    return events

class DataGenerator:
    def __init__(self, env, seq_in=10, seq_out=10, act=False, data_dir=None):
        self.env = env
        self.seq_in = seq_in
        self.seq_out = seq_out
        # self.if_flood= if_flood
        self.data_dir = data_dir if data_dir is not None else '.env/data/{}/'.format(env.config['env_name'])
        if act:
            self.action_table = list(env.config['action_space'].values())

    def simulate(self, event, seq_in=False, act=False):
        state = self.env.reset(event, global_state=True, seq=seq_in)
        perf = self.env.performance(seq=seq_in)
        states, perfs, settings = [state], [perf], []
        done = False
        while not done:
            setting = [table[np.random.randint(0, len(table))] for table in self.action_table] if act else None
            done = self.env.step(setting)
            state = self.env.state(seq=seq_in)
            perf = self.env.performance(seq=seq_in)
            states.append(state)
            perfs.append(perf)
            settings.append(setting)
        return np.array(states), np.array(perfs), np.array(settings) if act else None

    def state_split(self, states, perfs, settings=None):
        if settings is not None:
            # B,T,N,S
            states = states[:settings.shape[0] + 1]
            perfs = perfs[:settings.shape[0] + 1]
        h, q_totin, q_ds, r = [states[..., i] for i in range(4)]
        q_us = q_totin - r
        # B,T,N,in
        n_spl = self.seq_out #if self.recurrent else 1
        X = np.stack([h[:-n_spl],q_us[:-n_spl],q_ds[:-n_spl]],axis=-1)
        #   origin : X = np.stack([h[:-n_spl],q_us[:-n_spl],q_ds[:-n_spl]],axis=-1)
        Y = np.stack([h[n_spl:],q_us[n_spl:],q_ds[n_spl:]],axis=-1)   
        #   origin : Y = np.stack([h[n_spl:],q_us[n_spl:],q_ds[n_spl:]],axis=-1) 
        B = np.expand_dims(r[n_spl:], axis=-1)

        # Y =np.concatenate([Y,perfs[n_spl:]],axis=-1) 

        # if self.recurrent:
        X = X[:, -self.seq_in:, ...]
        B = B[:, :self.seq_out, ...]
        # Y = Y[:, :self.seq_out ,...] # :seq_out 
        
        return X, B, Y

    def generate(self, events, processes=1, seq=False, act=False):
        pool = mp.Pool(processes)
        if processes > 1:
            res = [pool.apply_async(func=self.simulate, args=(event, seq, act,)) for event in events]
            pool.close()
            pool.join()
            res = [r.get() for r in res]
        else:
            res = [self.simulate(event, seq, act) for event in events]
        self.states, self.perfs = [np.concatenate([r[i] for r in res], axis=0) for i in range(2)]
        self.settings = np.concatenate([r[2] for r in res], axis=0) if act else None
        self.event_id = np.concatenate([np.repeat(i, r[0].shape[0]) for i, r in enumerate(res)])

    def expand_seq(self, dats, seq):
        dats = np.stack([np.concatenate(
            [np.tile(np.zeros_like(s), (max(seq - idx, 0),) + tuple(1 for _ in s.shape)), dats[max(idx - seq, 0):idx]],
            axis=0) for idx, s in enumerate(dats)])
        return dats

    def prepare(self, seq=0, event=None):
        res = []
        event = np.arange(int(max(self.event_id)) + 1) if event is None else event
        for idx in event:
            num = self.event_id == idx
            #new
            # states = self.states[num]
            # perfs = self.perfs[num]
            # settings = self.settings[num] if self.settings is not None else None
            if seq > 0:
                states, perfs = [self.expand_seq(dat[num],seq) for dat in [self.states,self.perfs]]
                settings = self.expand_seq(self.settings[num],seq) if self.settings is not None else None
            x,b,y = self.state_split(states, perfs, settings)
            res.append((x.astype(np.float32),b.astype(np.float16),y.astype(np.float16)))
        return [np.concatenate([r[i] for r in res],axis=0) for i in range(3)]
    
    def prepare_Y(self, seq=0, event=None):
        res = []
        event = np.arange(int(max(self.event_id)) + 1) if event is None else event
        for idx in event:
            num = self.event_id == idx
            #new
            states = self.states[num]
            perfs = self.perfs[num]
            settings = self.settings[num] if self.settings is not None else None
            x,b,y = self.state_split(states, perfs, settings)
            res.append(y.astype(np.float32))
        return np.concatenate([r for r in res],axis=0)

    def save(self, data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        np.save(os.path.join(data_dir, 'states.npy'), self.states)
        np.save(os.path.join(data_dir, 'perfs.npy'), self.perfs)
        if self.settings is not None:
            np.save(os.path.join(data_dir, 'settings.npy'), self.settings)
        np.save(os.path.join(data_dir, 'event_id.npy'), self.event_id)

    def load(self, data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        for name in ['states', 'perfs', 'settings', 'event_id']:
            if os.path.isfile(os.path.join(data_dir, name + '.npy')):
                dat = np.load(os.path.join(data_dir, name + '.npy')).astype(np.float32)
            else:
                dat = None
            setattr(self, name, dat)




