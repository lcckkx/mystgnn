import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal import STConv
device = torch.device("cuda")


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features,periods,batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,
                           out_channels=64,
                           periods=periods,
                           batch_size=batch_size
                           ) 
        self.linear1 = torch.nn.Linear(1,64)
        self.tgnn1 = TGCN2(in_channels=64,#+32
                           out_channels=3,  #3                      
                           batch_size=batch_size
                           )
        # self.linear = torch.nn.Linear(64,periods)
        # self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index,b):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        # x_selected = x[:, :, :3, :]
        h = self.tgnn(x, edge_index) # x [B, 105, 3, 10]   
        #h[B,105,32]
        h = F.relu(h)
        h = torch.unsqueeze(h,dim=-2)#h [B,105,1,32]
        b =b.transpose(-1,-2) #into B,N,T,F
        R = self.linear1(b)   #R[b,105,10,32] into 
        R = F.relu(R)
        h = torch.cat((R[:,:,:1,:]+h, R[:,:,1:,:]), dim=-2)
        h_acc = torch.zeros(R.shape[0],R.shape[1],0,3).to(device)

        for i in range(10):
            hi = self.tgnn1(h[:,:,i,:], edge_index) # x [b, 105, 3, 12]
            hi=torch.unsqueeze(hi,dim=-2)
            # hi = self.dropout(hi)
            h_acc = torch.concat((h_acc,hi),dim=-2)
        h_acc = torch.transpose(h_acc,2,3)  # 32,105,10,3 >32,105,3,10
    
        return h_acc


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        
        self.tgnn = A3TGCN(in_channels=4,
                           out_channels=64,
                           periods=5,
                           )
        self.linear = torch.nn.Linear(1,64)
        self.dropout = nn.Dropout(p=0.5)
        self.tgnn1 = TGCN1(in_channels=64,#+32
                           out_channels=3,  #3               
                           )
        self.tgnn2 = A3TGCN(in_channels=64,
                           out_channels=3,
                           periods=5,
                           )
        
        self.fcs = nn.ModuleList([
        nn.Sequential(
        nn.Linear(2*32, 64),
        nn.ReLU(),
        self.dropout,
        nn.Linear(64, 3)
                        ) for _ in range(113)
                                                ])
        
        self.tgnn3 = torch.nn.Linear(64,3)

    def forward(self, x, edge_index): #input : x [B,10,113,4]  b [B,10,113,1]
        # x1 = x[:,:,:,:3] #[B,10,113,3]
        b = x[:,:,:,-1:]#[B,10,113,1]
        h = F.relu(self.tgnn(x, edge_index)) # x [B, 10, 113,4] #h [B,113,64]  ,应该是x，不是x1

        h = torch.unsqueeze(h,dim=-3)# into h [B,1,113,64]
        R = F.relu(self.linear(b)) #[B,10,113,64]
        h = torch.cat((R[:,:1,:,:]+h, R[:,1:,:,:]), dim=-3) #h [B,10,113,64]
        h_acc = F.relu(self.tgnn3(h))

        # h_acc = torch.zeros(R.shape[0],0,R.shape[2],3).to(device) #[B,0,113,3]
        # for i in range(10):
        #     hi = self.tgnn1(h[:,i,:,:], edge_index) # h [b, 10, 113, 64] hi[B,113,3]
        #     # hi=torch.unsqueeze(hi,dim=-3) # h [b, 1, 113, 3]
        #     h_acc = torch.concat((h_acc,hi),dim=-3)

        # pred = []
        # for k in range(h.shape[1]):
        #     pred.append(self.fcs[k](torch.flatten(h[:,k,:],start_dim=1)))
        # pred = torch.stack(pred,dim=1) #(256,113,3)
        # #h = self.linear(h)
        # h= pred.unsqueeze(1)
        # h = h.expand(-1,10,-1,-1)

        return h_acc


class NN(nn.Module):
    def __init__(self,input,output):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(input, 64) 
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, output) 

    def forward(self, x, b):  #x (256, 105, 3, 10)
        h = x.reshape(x.size(0), -1, x.size(-2)) #into (256,N,3) 
        R = b.reshape(b.size(0), -1, b.size(-2)) #into (256,N,1)
        ht = torch.cat((R,h), dim=-1)
        ht = self.fc1(ht)
        ht = self.relu(ht)
        ht = self.fc2(ht)
        return ht


class TGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0],X.shape[1], self.out_channels).to(X.device)   #x（20000,113,4）
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H): #H [B,113,64] test [B,113,64]
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=2)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=2)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=2)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class TGCN1(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, 
                 improved: bool = False, cached: bool = False, 
                 add_self_loops: bool = True):
        super(TGCN1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(in_channels=self.in_channels,  out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
    
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 113, 3)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)

        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)

        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)

        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
       
        H = self._set_hidden_state(X, H) #(128,105,64)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)#(128,105,64)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)#(128,105,64)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H


class TGCN2(torch.nn.Module):


    def __init__(self, in_channels: int, out_channels: int, 
                 improved: bool = False, cached: bool = False, 
                 add_self_loops: bool = True):
        super(TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(in_channels=self.in_channels,  out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            # can infer batch_size from X.shape, because X is [B, N, F]
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)

        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)

        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)

        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H) #(128,105,64)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)#(128,105,64)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)#(128,105,64)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H


class A3TGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
    ):
        super(A3TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:


        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(X[:,period,:,:], edge_index, edge_weight, H) #Hacc [B,113,64]
        return H_accum


class A3TGCN2(torch.nn.Module):

    def __init__(
        self,
        in_channels: int, 
        out_channels: int,  
        periods: int, 
        batch_size:int, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super(A3TGCN2, self).__init__()

        self.in_channels = in_channels  # 2
        self.out_channels = out_channels # 32
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels,  
            batch_size=self.batch_size,
            improved=self.improved,
            cached=self.cached, 
            add_self_loops=self.add_self_loops)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward( 
        self, 
        X: torch.FloatTensor,
        edge_index: torch.LongTensor, 
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):

            H_accum = H_accum + probs[period] * self._base_tgcn( X[:, :, :, period], edge_index, edge_weight, H) #([32, 207, 32]

        return H_accum
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, n_outputs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # print(input_seq.shape)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        preds = []
        pred1, pred2, pred3 = self.fc1(output), self.fc2(output), self.fc3(output)
        pred1, pred2, pred3 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :]

        # pred = torch.cat([pred1, pred2], dim=0)
        pred = torch.stack([pred1, pred2, pred3], dim=0)
        # print(pred.shape)

        return pred


class STGCN(nn.Module):

    def __init__(self):
        super(STGCN, self).__init__()
        self.conv1 = STConv(num_nodes = 113,
                            in_channels=4, 
                            hidden_channels=32,
                            out_channels=64, 
                            kernel_size=3, K=1)
        self.conv2 = STConv(num_nodes = 113,
                            in_channels=64, 
                            hidden_channels=32,
                            out_channels=16, 
                            kernel_size=3, K=1)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(2,64)
        self.fc2 = nn.Linear(64,10)
        
        self.fcs = nn.ModuleList()
        for k in range(113):
            self.fcs.append(nn.Sequential(nn.Linear(2*16,64),
                                          nn.ReLU(),
                                          nn.Linear(64,3)
                                          )
            )
        
    def forward(self, x, edge_index):
        # x(batch_size, seq_len, num_nodes, in_channels)
        x, edge_index = x.to(device), edge_index.to(device)
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = self.dropout(F.relu(self.conv2(x, edge_index)))

        #x[B,2,113,3]
        # x = x.transpose(1,3)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = x.transpose(1,3)
        pred = []

        
        for k in range(x.shape[2]):
            pred.append(self.fcs[k](torch.flatten(x[:,:,k,:],start_dim=1)))
        pred = torch.stack(pred,dim=0) #(113,256,3)
        pred = pred.transpose(1,0)

        
        return pred

