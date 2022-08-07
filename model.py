import torch
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self,
                 hid_dim=2048):
        super(GraphConv, self).__init__()
        self.hid_dim = hid_dim
        self.W = nn.Parameter(torch.zeros(hid_dim, hid_dim), requires_grad=True)
        self.act = nn.ReLU()
        
    def forward(self, x, A):
        b, t, c = x.size()
        W = self.W.unsqueeze(0).repeat(b, 1, 1)
        o = torch.bmm(A, x)
        o = torch.bmm(o, W)
        return o
    
class ACGNet(nn.Module):
    def __init__(self,
                 num_layers=2,
                 hid_dim=2048):
        super(ACGNet, self).__init__()
        self.hid_dim = hid_dim
        self.gcn = nn.ModuleList(
            [GraphConv(hid_dim) for _ in range(num_layers)]
        )
    
    def make_ssg(self, x):
        '''
            make similarity graph
            input: feature(x) : b, t, c
            output: ssg: b, t, t
        '''
        x_t = x.permute(0, 2, 1)
        
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True) + 1e-9
        x_t_norm = torch.norm(x_t, p=2, dim=1, keepdim=True) + 1e-9
        
        ssg = torch.bmm((x/x_norm), (x_t/x_t_norm))
        return ssg
    
    def make_tdg(self, x, Z=10):
        '''
            make temporal diffusion graph
            input: x: b, t, c
                   Z: hyperparameter, an integer indicating diffusion degree 
        '''
        b, t, c = x.size()
        tmp_vecs = [torch.arange(t, dtype=torch.float32) - i for i in range(t)]
        g = torch.abs(torch.stack(tmp_vecs))
        g = torch.max(Z-g, torch.zeros_like(g))
        tgd = 1. - g / Z
        tgd = tgd.unsqueeze(0).repeat(b, 1, 1).to(x.device)
        return tgd
    
    def make_osg(self, x, alpha=1., lamda=0.85, K=50):
        '''
            make overall sparse graph
        '''
        b, t, c = x.size()
        A = 0.5 * (self.make_ssg(x) + alpha * self.make_tdg(x))
        _, idx = torch.topk(A, k=K, dim=-1, largest=True)
        mask = torch.zeros_like(A)
        mask = mask.scatter(dim=-1, index=idx, value=1.)
        A = A * mask
        A = torch.where(A > lamda, A, torch.full_like(A, fill_value=0, dtype=A.dtype))
        return A
    
    def forward(self, x):
        A_prime = self.make_osg(x)
        A_hat = A_prime / (torch.sum(A_prime, dim=-1, keepdim=True) + 1e-9)
        F_avg = torch.bmm(A_hat, x)
        for i, layer in enumerate(self.gcn):
            if i == 0:
                F_conv = layer(x, A_hat)
            else:
                F_conv = layer(F_conv, A_hat)
        return x, x + F_avg + F_conv, A_prime