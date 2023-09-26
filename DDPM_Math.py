import torch
from torch.distributions.normal import Normal
import torch.nn as nn

def get_ddpm_schedules(T=1000):
    """ Returns beta_sched, alpha_sched, bar_alpha_sched
        as described by vanilla ddpm paper, following exactly
    """
    beta_sched = torch.linspace(0.0001, 0.02, T)
    alpha_sched = 1 - beta_sched

    bar_alpha_sched = alpha_sched.clone()
    for i in range(1, len(bar_alpha_sched)):
        bar_alpha_sched[i] = bar_alpha_sched[i] * bar_alpha_sched[i-1]
    
    return beta_sched, alpha_sched, bar_alpha_sched

##closed form formula for forward diffusion at any given timestep, given x_0
class closed_forward_diffusion(nn.Module):
    def __init__(self, bar_alpha, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.bar_alpha = bar_alpha
        self.bar_alpha.to(device)
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
        batch_means = torch.sqrt(batch_bar_alpha_ts).view(t.shape[0], 1, 1, 1) * x_0
        ##they use a multivariate normal with covariance as scaled identity matrix
        ##this is equivalent to an individual guassian for each pixel, i.e. zero covariance
        ##just make sure you input the std, not the variance into the univariate distribution

        batch_vars = (1-batch_bar_alpha_ts)
        batch_stds = torch.sqrt(batch_vars).view(batch_bar_alpha_ts.shape[0],1,1,1).repeat(1,3,x_0.shape[-2], x_0.shape[-1]) 

        ##make the distribution
        dist = Normal(torch.zeros_like(batch_means), torch.ones_like(batch_means))
        epsilon  = dist.sample()

        assert(epsilon.shape == batch_stds.shape, 'epsilon noise and batch_stds shape mismatch')

        x_t = batch_means + (batch_stds*epsilon)

        return x_t, epsilon        

##positional encodings are used from the transformer paper - at every resblock of the unet
##look at readme of how I better implemented the time embeddings for ddpm; what were these authors on? Not the good stimulants it seems
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