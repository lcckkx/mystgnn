import numpy as np
from datetime import datetime
from swmm_api import read_inp_file
import multiprocessing as mp
import os
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from envs.scenario.shunqing import shunqing
import yaml
from sklearn import preprocessing

HERE = os.path.dirname(__file__)
# def load_adj():
#
#     edges = shunqing.get_edge_list()
#     adj = np.zeros((edges.max()+1,edges.max()+1))
#     for u,v in edges:
#         adj[u,v] +=1
#
#     adj = adj.tocsc()
#     #toarray
#
#     return adj


class DataGenerator:
    def __init__(self, env, seq_in=12,seq_out=1, act=False, data_dir=None):
        self.env = env
        self.seq_in = seq_in
        self.seq_out = seq_out
        # self.if_flood= if_flood
        self.data_dir = data_dir if data_dir is not None else '.env/data/{}/'.format(env.config['env_name'])
        if act:
            self.action_table = list(env.config['action_space'].values())



    def simulate(self, event,seq = False, act=False):
        state = self.env.reset(event, global_state=True, seq=self.seq)
        perf = self.env.performance(seq=seq)
        states, perfs, settings = [state],[perf], []
        done = False
        while not done:
            setting = [table[np.random.randint(0, len(table))] for table in self.action_table] if act else None
            done = self.env.step(setting)
            state = self.env.state(seq=self.seq)
            perf = self.env.performance(seq_len=seq)
            states.append(state)
            perfs.append(perf)
            settings.append(setting)
        return np.array(states),np.array(perfs), np.array(settings) if act else None

    def state_split(self, states, perfs, settings=None):
        if settings is not None:
            # B,T,N,S
            states = states[:settings.shape[0] + 1]
            perfs = perfs[:settings.shape[0] + 1]
            # B,T,n_act
            a = np.tile(np.expand_dims(settings, axis=1), [1, states.shape[1], 1])
        h, q_totin, q_ds, r = [states[..., i] for i in range(4)]
        # h,q_totin,q_ds,r,q_w = [states[...,i] for i in range(5)]
        q_us = q_totin - r
        # B,T,N,in
        n_spl = self.seq_out #if self.recurrent else 1
        X = np.stack([h[:-n_spl], q_us[:-n_spl], q_ds[:-n_spl]], axis=-1)
        Y = np.stack([h[n_spl:], q_us[n_spl:], q_ds[n_spl:]], axis=-1)
        # TODO: classify flooding
        # if self.if_flood:
        #     f = (perfs > 0).astype(int)
        #     f = np.eye(2)[f].squeeze(-2)
        #     X, Y = np.concatenate([X, f[:-n_spl]], axis=-1), np.concatenate([Y, f[n_spl:]], axis=-1)
        Y = np.concatenate([Y, perfs[n_spl:]], axis=-1)
        B = np.expand_dims(r[n_spl:], axis=-1)
        #if self.recurrent:
        X = X[:, -self.seq_in:, ...]
        B = B[:, :self.seq_out, ...]
        Y = Y[:, :self.seq_out, ...]
        if settings is not None:
            B = np.concatenate([B, a], axis=-1)
        return X, B, Y


    def generate(self, events, processes=1, seq=False, act=False):
        pool = mp.Pool(processes)
        if processes > 1:
            res = [pool.apply_async(func=self.simulate,args=(event,seq,act,)) for event in events]
            pool.close()
            pool.join()
            res = [r.get() for r in res]
        else:
            res = [self.simulate(event,seq,act) for event in events]
        self.states, self.perfs = [np.concatenate([r[i] for r in res],axis=0) for i in range(2)]
        self.settings = np.concatenate([r[2] for r in res],axis=0) if act else None
        self.event_id = np.concatenate([np.repeat(i,r[0].shape[0]) for i,r in enumerate(res)])



    def expand_seq(self,dats,seq):
        dats = np.stack([np.concatenate([np.tile(np.zeros_like(s),(max(seq-idx,0),)+tuple(1 for _ in s.shape)),dats[max(idx-seq,0):idx]],axis=0) for idx,s in enumerate(dats)])
        return dats


    def prepare(self,seq=0,event=None):
        res = []
        event = np.arange(int(max(self.event_id))+1) if event is None else event
        for idx in event:
            num = self.event_id == idx
            if seq > 0:
                states,perfs = [self.expand_seq(dat[num],seq) for dat in [self.states,self.perfs]]
                settings = self.expand_seq(self.settings[num],seq) if self.settings is not None else None
            x,b,y = self.state_split(states,perfs,settings)
            res.append((x.astype(np.float32),b.astype(np.float32),y.astype(np.float32)))
        return [np.concatenate([r[i] for r in res],axis=0) for i in range(3)]



    def save(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        np.save(os.path.join(data_dir,'states.npy'),self.states)
        np.save(os.path.join(data_dir,'perfs.npy'),self.perfs)
        if self.settings is not None:
            np.save(os.path.join(data_dir,'settings.npy'),self.settings)
        np.save(os.path.join(data_dir,'event_id.npy'),self.event_id)


    def load(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        for name in ['states','perfs','settings','event_id']:
            if os.path.isfile(os.path.join(data_dir,name+'.npy')):
                dat = np.load(os.path.join(data_dir,name+'.npy')).astype(np.float32)
            else:
                dat = None
            setattr(self,name,dat)

    # def save(self, data_dir=None):
    #     data_dir = data_dir if data_dir is not None else self.data_dir
    #     np.save(os.path.join(data_dir, 'X.npy'), self.X)
    #     np.save(os.path.join(data_dir, 'Y.npy'), self.Y)
    #     np.save(os.path.join(data_dir, 'event_id.npy'), self.event_id)
    #
    # def load(self, data_dir=None):
    #     data_dir = data_dir if data_dir is not None else self.data_dir
    #     for name in ['X', 'Y', 'event_id']:
    #         dat = np.load(os.path.join(data_dir, name + '.npy'))
    #         setattr(self, name, dat)

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



class Emulator():


    def set_norm(self,normal):
        setattr(self,'normal',normal)

    def normalize(self,dat,inverse=False):
        dim = dat.shape[-1]
        if dim >= 3:
            return dat * self.normal[...,:dim] if inverse else dat/self.normal[...,:dim]
        else:
            return dat * self.normal[...,-dim:] if inverse else dat/self.normal[...,-dim:]


    def constrain(self,y,r):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = np.squeeze(r,axis=-1)
        y = np.stack([h,q_us,q_ds],axis=-1)
        return y,r

    




# def load_data(dataset_name, len_train, len_val):
#
#
#     dataset_path = './data'
#     dataset_path = os.path.join(dataset_path, dataset_name)
#     vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))
#
#     train = vel[: len_train]
#
#     test = vel[len_train + len_val:]
#     return train, test


def data_transform(data, seq_in, seq_out, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - seq_in - seq_out

    x = np.zeros([num, 1, seq_in, n_vertex])
    y = np.zeros([num, n_vertex])

    for i in range(num):
        head = i
        tail = i + seq_in
        x[i, :, :, :] = data[head: tail].reshape(1, seq_in, n_vertex)
        y[i] = data[tail + seq_out - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
