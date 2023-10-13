import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import math

def get_ddpm_schedules(T=1000, type='linear'):
    """ Returns beta_sched, alpha_sched, bar_alpha_sched
        as described by vanilla ddpm paper, followed exactly
    """
    assert type == 'linear' or 'cosine'

    if type == 'linear':
        print('using Linear Beta Schedule')
        beta_sched = torch.linspace(0.0001, 0.02, T)

    else:
        print('using Cosine Beta Schedule')
        max_beta = 0.999
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        beta_sched = []
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            beta_sched.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        beta_sched = torch.tensor(beta_sched, dtype=torch.float32)
    
    alpha_sched = torch.tensor([1.0],dtype=torch.float32) - beta_sched
    bar_alpha_sched = alpha_sched.clone()
    for i in range(1, len(bar_alpha_sched)):
        bar_alpha_sched[i] = bar_alpha_sched[i] * bar_alpha_sched[i-1]
    
    return beta_sched, alpha_sched, bar_alpha_sched

##closed form formula for forward diffusion at any given timestep, given x_0
class closed_forward_diffusion(nn.Module):
    def __init__(self, bar_alpha, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.bar_alpha = bar_alpha
        self.bar_alpha = self.bar_alpha.to(device)
        self.device = device
    def forward(self, x_0, t):
        """ 
            Dataloader samples t for every image and applies this function across the batch
            Takes batch as input 
            O(1) forward diffusion process, given by equation:
            q(x_t|x0) = Normal(xt; sqrt(bar_alpha_t)x0, (1-bar_alpha_t)I)

            x0: images, shape: (b, 3, 32,32), if cifar10
            t: sampled times from catigorical distribution from 0 to 1000-1, shape: (b)
            bar_alpha: schedule proposed by paper, a function of the beta schedule

            returns the noised x0 and the unscaled epsilon noise
        """
        batch_bar_alpha_ts = self.bar_alpha[t]
        batch_means = torch.sqrt(batch_bar_alpha_ts).unsqueeze(-1,).unsqueeze(-1).unsqueeze(-1) * x_0
        assert len(batch_means.shape) == 4
        ##they use a multivariate normal with covariance as scaled identity matrix
        ##this is equivalent to an individual guassian for each pixel, i.e. zero covariance
        ##just make sure you input the std, not the variance into the univariate distribution

        batch_vars = (torch.tensor([1.0],dtype=torch.float32).to(self.device)-batch_bar_alpha_ts)
        batch_stds = torch.sqrt(batch_vars).view(batch_bar_alpha_ts.shape[0],1,1,1).repeat(1,3,x_0.shape[-2], x_0.shape[-1]) 

        ##make the distribution
        dist = Normal(torch.zeros_like(batch_means), torch.ones_like(batch_means))
        epsilon  = dist.sample().to(self.device)

        assert epsilon.shape == batch_stds.shape

        x_t = batch_means + (batch_stds*epsilon)

        return x_t, epsilon        

##positional encodings are used from the transformer paper - at every resblock of the unet
class pos_encoding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, t, d=64):
        """ same as transformer positional encoding, completely vectorized
        """
        
        #sin & cos indices
        sin_i = 2*torch.arange(d//2 + (1 if d % 2 == 1 else 0), dtype=torch.int).to(t.device)
        cos_i = 2*torch.arange(d//2, dtype=torch.int).to(t.device) + 1

        #use indices to calculate positional encoding
        #even indices = sin(pos/10000^{i/d})
        #odd indices =  cos(pos/10000^{i/d})
        pe = torch.arange(d).to(t.device).view(1, d).repeat(t.shape[0], 1)*1.0
        
        pe[:, sin_i] = torch.sin(t.view(-1,1).repeat(1,sin_i.shape[0])/(10000**(pe[:, sin_i]/d)))
        pe[:, cos_i] = torch.cos(t.view(-1,1).repeat(1,cos_i.shape[0])/(10000**(pe[:, cos_i]/d)))
        return pe


