import argparse
from dataloader import DataGenerator,generate_file
from envs import get_env
import os   
import numpy as np  
import time

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='shunqing', help='drainage scenarios')
    parser.add_argument('--result_dir',type=str,default='./results/shunqing',help='the test results')
    parser.add_argument('--data_dir', type=str, default='./envs/data/shunqing/', help='the sampling data file')
    parser.add_argument('--seq_in',type=int,default=12,help='input sequential length')
    parser.add_argument('--seq_out',type=int,default=12,help='out sequential length. if not roll, seq_out < seq_in ')
    parser.add_argument('--processes', type=int, default=5, help='number of simulation processes')
    parser.add_argument('--act', default=True, help='if the environment contains control actions')

    args = parser.parse_args()
    return args

def simulate(self,states,runoff):
        # runoff shape: T_out, T_in, N
        preds = []
        for idx,ri in enumerate(runoff):
            x = states[idx,-self.seq_in:,...]
            # x = x if self.roll else state
            if self.roll:
                # TODO: What if not recurrent
                qws,ys = [],[]
                for i in range(ri.shape[0]):
                    r_i = ri[i:i+self.seq_out]
                    y = self.predict(x,r_i)
                    q_w,y = self.constrain(y,r_i)
                    x = np.concatenate([x[1:],y[:1]],axis=0) if self.recurrent else y
                    qws.append(q_w)
                    ys.append(y)
                q_w,y = np.concatenate(qws,axis=0),np.concatenate(ys,axis=0)
            else:
                ri = ri[:self.seq_out]
                y = self.predict(x,ri)
                q_w,y = self.constrain(y,ri)
            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            preds.append(y)
        return np.array(preds)




if __name__ == "__main__":
 
    args = get_parameters()
    env = get_env(args.env)()
    env_args = env.get_args()
    for k,v in env_args.items():
        if k == 'act':
            v = v & args.act
        setattr(args,k,v)

    
    dG = DataGenerator(env,args.seq_in,args.seq_out,args.act,args.data_dir)
    events = generate_file(env.config['swmm_input'],env.config['rainfall'])


    for event in events:
        name = os.path.basename(event).strip('.inp')
        if os.path.exists(os.path.join(args.result_dir,name + '_states.npy')):
            states = np.load(os.path.join(args.result_dir,name + '_states.npy'))
            perfs = np.load(os.path.join(args.result_dir,name + '_perfs.npy'))
        else:
            t0 = time.time()
            seq = max(args.seq_in,args.seq_out)
            states,perfs,settings = dG.simulate(event,seq,act=args.act)
            print("{} Simulation time: {}".format(name,time.time()-t0))
            np.save(os.path.join(args.result_dir,name + '_states.npy'),states)
            np.save(os.path.join(args.result_dir,name + '_perfs.npy'),perfs)

        states[...,1] = states[...,1] - states[...,-1]
        r,true = states[args.seq_out:,...,-1:],states[args.seq_out:,...,:-1]
        states = states[...,:-1]
        t0 = time.time()
        states = states[:-args.seq_out]
        # pred = emul.simulate(states,r)
        print("{} Emulation time: {}".format(name,time.time()-t0))
        true = np.concatenate([true,perfs[args.seq_out:,...]],axis=-1)
        np.save(os.path.join(args.result_dir,name + '_runoff.npy'),r)
        np.save(os.path.join(args.result_dir,name + '_true.npy'),true)



