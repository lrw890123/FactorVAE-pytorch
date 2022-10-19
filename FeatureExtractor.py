import torch
from torch import nn
from torch.nn import functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, num_characristics, proj_dim=20, hidden_size=25, GRU_num_layers=1) -> None:
        super(FeatureExtractor, self).__init__()

        self.linear = nn.Linear(num_characristics, proj_dim)
        self.GRU = nn.GRU(proj_dim, hidden_size, GRU_num_layers)

    
    def forward(self, x):
        #x: shape = [batch_size, Ns, T, C]
        x = self.linear(x)
        h_proj,  = F.leaky_relu(x)
        #h_proj: shape = [batch_size, Ns, T, proj_dim]


        B, Ns, T, C = h_proj.shape
        h_proj = torch.reshape(h_proj, shape=(B*Ns, T, C))
        _, h_gru = self.GRU(h_proj)
        h_gru = torch.reshape(x, shape=(B, Ns, -1))

        #h_gru: shape = [batch_size, Ns, hidden_size]
        return h_gru
        

class FactorEncoder(nn.Module):
    def __init__(self, num_portfolio=30, hidden_size=25, num_factors=25) -> None:
        super(FactorEncoder, self).__init__()

        self.portfolio_layer = nn.Sequential(
            nn.Linear(hidden_size, num_portfolio),
            nn.Softmax()
        )

        self.map_layer_mu = nn.Linear(num_factors, num_portfolio)
        self.map_layer_sigma = nn.Sequential(
            nn.Linear(num_factors, num_portfolio),
            nn.Softplus()
        )

    def forward(self, e, y):
        '''
        y: shape = [batch_size, Ns]
        e: shape = [batch_size, Ns, hidden_size]
        '''
        a_p = self.portfolio_layer(e)
        # a_p: shape = [batch_size, num_stocks, num_portfolio]

        y_p = torch.einsum('bn,bnm->bm', y, a_p).unsqueeze(0)
        # y_p: shape = [batch_size, num_portfolio]
        mu_post = torch.squeeze(self.map_layer_mu(y_p))
        sigma_post = torch.squeeze(self.map_layer_sigma(y_p))

        '''
        mu_post, sigma: shape = [batch_size, num_factors]
        '''
        return mu_post, sigma_post

class AlphaLayer(nn.Module):
    def __init__(self, hidden_size=25) -> None:
        super(AlphaLayer, self).__init__()

        self.h_alpha_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.mu_alpha_layer = nn.Linear(hidden_size, 1)
        self.sigma_alpha_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )

    def forward(self, e):
        # e: shape = [batch_size, num_stocks, hidden_size]

        h_alpha = self.h_alpha_layer(e)
        # h_alpha: shape = [batch_size, num_stocks, hidden_size]
        
        mu_alpha = self.mu_alpha_layer(h_alpha)
        sigma_alpha = self.sigma_alpha_layer(h_alpha)
        # mu_alpha, sigma_alpha: shape = [batch_size, num_stocks]

        return mu_alpha, sigma_alpha

class FactorDecoder(nn.Module):
    def __init__(self, hidden_size=25, num_factors=25) -> None:
        super(FactorDecoder, self).__init__()

        self.alpha_layer = AlphaLayer(hidden_size)
        self.beta_layer = nn.Linear(num_factors, hidden_size)

    def forward(self, e, z):
        '''
        '''
        mu_z, sigma_z = z[0].unsqueeze(-1), z[1].unsqueeze(-1)
        # mu_z.shape, sigma_z.shape == (K, 1)
        
        mu_alpha, sigma_alpha = self.alpha_layer(e)
        mu_alpha = mu_alpha.unsqueeze(-1)
        sigma_alpha = sigma_alpha.unsqueeze(-1)
        
        beta = self.beta_layer(e)

        mu_y = (mu_alpha + torch.bmm(beta, mu_z)).squeeze(-1)
        sigma_y = torch.sqrt(torch.square(sigma_alpha) + torch.bmm(torch.square(beta), torch.square(sigma_z))).squeez(-1)

        return mu_y, sigma_y


class FactorPredictor(nn.Module):
    def __init__(self, hidden_size=25, num_factors=25) -> None:
        super(FactorPredict,self).__init__()
        self.w_key = torch.randn(num_factors, 1)
        self.w_key.requires_grad = True
        self.w_value = torch.randn(num_factors, 1)
        self.w_value.requires_grad = True

        self.q = torch.randn(hidden_size,)
        self.q.requires_grad = True

        self.mu_prior = nn.Linear(hidden_size, 1)
        self.sigma_prior = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()
            )

    def forward(self, e):
        e_ = e.unsqueeze(1)
        k = torch.einsum('kd,bdnh->bknh', self.w_key, e_)
        v = torch.einsum('kd,bdnh->bknh', self.w_value, e_)

        q_norm = torch.norm(self.q, p=2, dim=-1, keepdim=True)
        k_norm = torch.norm(k, p=2, dim=-1, keepdim=True)

        a_att = torch.einsum('h,bknh->bkn', self.q, k)/(q_norm*k_norm)
        a_att= torch.maximum(0., a_att)
        a_att = a_att/torch.maximum(1e-6, torch.reduce_sum(a_att, 1e-6, axis=-1, keepdims=True))

        h_muti = torch.einsum('bkn, bknh->bkh', a_att, v)

        mu_prior = self.mu_prior(h_muti).squeeze(-1)
        sigma_prior = self.sigama_prior(h_muti).squeeze(-1)

        return mu_prior, sigma_prior    

class FactorVAE(nn.Module):
    def __init__(self, num_characristics, num_portfolios=30, num_factors=25, proj_dim=20, hidden_size=25, GRU_num_layers=1):
        super(FactorVAE, self).__init__()

        self.feature_extractor = FeatureExtractor(num_characristics=num_characristics, proj_dim=proj_dim, 
                                                  hidden_size=hidden_size, GRU_num_layers=GRU_num_layers)

        self.factor_encoder = FactorEncoder(num_portfolio=num_portfolios, hidden_size=hidden_size, num_factors=nums_factor)
        self.factor_predictor = FactorPredictor(hidden_size=hidden_size, num_factors=num_factors)
        self.factor_decoder = FactorDecoder(hidden_size=hidden_size, num_factors=num_factors)

    def forward(self, x, y=None, training=False):

        if training:
            if y is None:
                raise ValueError("`y` must be stock future return!")
            e = self.feature_extractor(x)
            z_post = self.factor_encoder(e, y)
            z_prior = self.factor_predictor(e)
            y_rec = self.factor_decoder(e, z_post)
            y_pre = self.factor_decoder(e, z_prior)

            return z_post, z_prior, y_rec, y_pre

        else:
            e = self.feature_extractor(x)
            z_prior = self.factor_predictor(e)
            y_pre = self.factor_decoder(e, z_prior)
            
            return y_pre


