import torch
from torch import nn
from torch.nn import functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, num_characristics, proj_dim, hidden_size=25, GRU_num_layers=1) -> None:
        super().__init__()

        self.linear = nn.Linear(num_characristics, proj_dim)
        self.GRU = nn.GRU(proj_dim, hidden_size, GRU_num_layers)

    
    def forward(self, x, zeta):
        #x: shape = [batch_size, Ns, T, C]
        x = self.linear(x)
        h_proj,  = F.leaky_relu(x, zeta)
        #h_proj: shape = [batch_size, Ns, T, proj_dim]


        B, Ns, T, C = h_proj.shape
        h_proj = torch.reshape(h_proj, shape=(B*Ns, T, C))
        _, h_gru = self.GRU(h_proj)
        h_gru = torch.reshape(x, shape=(B, Ns, -1))

        #h_gru: shape = [batch_size, Ns, hidden_size]
        return h_gru
        

class FactorEncoder(nn.Module):
    def __init__(self, num_portfolio, num_stocks, num_factors) -> None:
        super().__init__()

        self.portfolio_layer = nn.Sequential(
            nn.Linear(num_stocks, num_portfolio),
            nn.Softmax()
        )

        self.map_layer_mu = nn.Linear(num_factors, num_portfolio)
        self.map_layer_sigma = nn.Sequential(
            nn.Linear(num_factors, num_portfolio),
            nn.Softplus()
        )

    def forward(self, y, e):
        '''y: shape = [batch_size, Ns]
           e: shape = [batch_size, Ns, hidden_size]
        '''
        a_p = self.portfolio_layer(e)
        # a_p:shape = [batch_size, Ns, num_portfolio]

        y_p = torch.einsum('bn,bnm->bm', y, a_p)

        mu_post = torch.squeeze(self.map_layer_mu(y_p))
        sigma_post = torch.squeeze(self.map_layer_sigma(y_p))

        return mu_post, sigma_post

class FactorDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.