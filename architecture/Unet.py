##Regular UNet has 4 down/upscals, for cifar10, they use only 3 levels: 32x32 -> 4x4 -> 32x32

##Necesary Assumptions, normalization choice changes:
##this is the setup they had in the paper, the only difference is I'm using layernorm for the convs, and layernorm after the self attention
##and I'm using handcrafted number of channels and handcrafted projection of time embeddings, everything else is the exact same as the paper
##everything 'handcrafted' is due to lack of details of the paper, assuming my intuition is enough to determine these numbers
##due to channel choice difference, my model has 41 million params, rather than 35 million params
##it's also unclear the number of heads they used for multihead attention, but 4 seems to be the common go-to, using that here

import torch
import torch.nn as nn
from DDPM_Math import pos_encoding, get_ddpm_schedules
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##do not change name of ResBlock class, string of class name used in cutom Sequential object below
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, t_dim=512, add_time=False):
        super().__init__()
        self.add_time = add_time
        self.time_dim = t_dim
        if self.add_time:
            self.time_projector = nn.Linear(t_dim, in_channels)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU())
        self.resid_connector = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.in_channels = in_channels
        self.pos_encoder = pos_encoding()
    def forward(self, xb, t):
        #time embedding works like an extra bias for the convolutions of previous resblock
        ##output will be (b, in_channels)
        #after unsqueezing this becomes (b, in_channels, 1, 1), as to broadcaast along channels
        if self.add_time:
            time_embedding = self.time_projector(self.pos_encoder(t, self.time_dim)).unsqueeze(-1).unsqueeze(-1)
            xb = xb + time_embedding
        return self.network(xb) + self.resid_connector(xb)

class ResSelfAttention(nn.Module):
    def __init__(self, embed_dim=16*16, num_heads=4):
        """
            For use of self attention at 16x16 resolution by default
            includes a residual connection and normalization after
            default to layer norm because it works well
            group/batch norm optional hyperparameters
            default to batch first, why wouldn't you?
            NLPers are cringe for not defaulting batch dim first
        """
        super().__init__()

        self.flatten = nn.Flatten(-2,-1) #flattens (...,w,h) to (...,embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, xb):
        ##flatten to (b,c,embed_dim)
        xb_flat = self.flatten(xb)
        attended, _ = self.self_attention(xb_flat,xb_flat,xb_flat)
        ##normalize the out + residual
        norm = self.norm(attended+xb_flat)
        ##reshape the output to (b,c,w,h)
        return norm.view(xb.shape[0], xb.shape[1], xb.shape[2], xb.shape[3])

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1)
        self.resizer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, xb):
        xb = self.upsampler(xb)
        return self.conv(xb) + self.resizer(xb)
    
##have to make my own sequential, because we have to pass t through our resblocks and not through attn blocks
class Sequential(nn.Module):
    def __init__(self, layers):
        """
        layers must be a tuple of the layers
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, xb, t):
        for layer in self.layers:
            ##if it's a resblock, we also have to pass in t 
            if str(type(layer)) == "<class 'architecture.Unet.ResBlock'>":
                xb = layer(xb, t)
            ##otherwise, it's a self attention + normalization layer
            else:
                xb = layer(xb)
        return xb

class UNet(nn.Module):
    def __init__(self, T=1000, t_dim=64, sched_type='linear'):
        super().__init__()
        print("UNet instantiated")
        self.sched_type = sched_type
        self.T = T
        self.beta_sched, self.alpha_sched, self.bar_alpha_sched = get_ddpm_schedules(T=self.T, type=sched_type)
        ##ENCODER instance vars     beta_sched, alpha_sched, bar_alpha_sched      #32x32
        self.encodeB = Sequential((ResBlock(3,32,32), ResBlock(32,64,64, add_time=True)))
        self.MaxPoolB = nn.MaxPool2d(2)   #16x16
        self.encodeM = Sequential((ResBlock(64,128,128),ResSelfAttention(embed_dim=16*16), ResBlock(128,256,256, add_time=True)))
        self.MaxPoolM = nn.MaxPool2d(2)   #8x8
        self.encodeS = Sequential((ResBlock(256,256,256), ResBlock(256,512,512, add_time=True)))
        self.MaxPoolS = nn.MaxPool2d(2)   #4x4
        
        ##Connector
        self.Connector = Sequential((ResBlock(512,512,512),ResBlock(512,512,1024, add_time=True))) ##should be 4 x 4 with 1024 feature maps
        
        ##Decoder instance vars - notice inputs are twice as big as should be for the convblock, normally
        ##This is because of the concatonation that happens with UNet 
        self.upsamplerS = Upsampler(1024, 512) #8x8
        self.decodeS = Sequential((ResBlock(1024, 512, 512), ResBlock(512, 512, 512, add_time=True)))
        self.upsamplerM = Upsampler(512, 256) #16x16
        self.decodeM = Sequential((ResBlock(512, 256, 256), ResSelfAttention(embed_dim=16*16), ResBlock(256, 128, 128, add_time=True)))
        self.upsamplerB = Upsampler(128, 64)  #32x32
        self.decodeB = Sequential((ResBlock(128, 64, 64),ResBlock(64, 64, 64, add_time=True)))
        
        ##from #feature maps to 3 channels for image reconstruction from latent space
        self.to_img = ResBlock(64, 64, 3)
        
    def forward(self, xb, t):
        ##make sure the batch sizes match x and t
        assert xb.shape[0] == t.shape[0]

        #pass through encoder
        b = self.encodeB(xb, t)
        xb = self.MaxPoolB(b)
        m = self.encodeM(xb, t)
        xb = self.MaxPoolM(m)
        s = self.encodeS(xb, t)
        xb = self.MaxPoolS(s)
        #pass through connector
        xb = self.Connector(xb, t)
        
        #pass through decoder
        xb = self.upsamplerS(xb)
        xb = self.decodeS(torch.cat((s,xb), dim=1), t)
        xb = self.upsamplerM(xb)
        xb = self.decodeM(torch.cat((m,xb), dim=1), t)
        xb = self.upsamplerB(xb)
        xb = self.decodeB(torch.cat((b,xb), dim=1), t)
        return self.to_img(xb, t)

            
    def DDIM_Sample(self, num_imgs=1, steps=1000, res=(32,32), x0_min=-1, x0_max=1, ddpm=False, upper_bound=False,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    ret_steps=None):
        """
            Function assums hyperparameter n is 0. This means DDIM sampling only. 
            This hyperparameter can be added later, but currently, I don't see the point.
            Using assumption all elements x0 is in clamped range [x0_min, x0,max]
            This assumption would be wrong if elements have their own unique ranges-
            if that is the case -such as in action spaces- , code must be modified to accomidate
        """
        ##cannot use cosine schedule with upperbound=True, results in a sqrt of a negative number with ddim formula
        if self.sched_type == 'cosine':
            assert upper_bound == False

        ##ensure that training steps is divible by ddim steps
        assert (self.T % steps) == 0

        ##meh, easy to read and understand, not gonna sit here
        ##and type out an explanation for these lines of code for you
        self.to(device)
        self.eval()
        self.bar_alpha_sched = self.bar_alpha_sched.to(device)
        self.beta_sched = self.beta_sched.to(device)
        returns = []

        ##dist to be used for sampling both xt initially nose is in shape (b,3,w,h)
        dist = Normal(torch.zeros(num_imgs, 3, res[0], res[1]), torch.ones(num_imgs, 3, res[0], res[1]))
        curr_x = dist.sample().to(device)
        ##have to append the first step :)
        returns.append(curr_x)
        skip_by = self.T // steps
        ##ddpm sampling only on same markov chain as training
        if ddpm:
            skip_by = 1 
        with torch.no_grad():
            for i in range(1, steps+1)[::-1]:
                ##S now equals steps to 1, i.e. for steps = 100, steps = 100 to 1
                ##indices of t and bar alpha are expected to be 0 to (step_size-1)
                ##indices of the t and tdelta, and labels of the curr_t
                curr_t = (i * skip_by)-1
                
                if i == (steps) and ddpm:
                    assert curr_t == (1000 - 1)
                curr_t_labels = torch.tensor([curr_t]).to(device).repeat(num_imgs)
                next_t = ((i-1) * skip_by)-1

                ##ddpm noise addition vars
                z = dist.sample().to(device)
                var = 0.0
                sigma = 0.0

                ##don't add noise for the final result :), otherwise, add noise according to schedule
                if ddpm and i!=-1:
                    if upper_bound:
                        var = self.beta_sched[curr_t]
                    else:
                        var = ((1-self.bar_alpha_sched[next_t])/(1-self.bar_alpha_sched[curr_t]))*self.beta_sched[curr_t]
                    sigma = torch.sqrt(var)
                
                ##predict total noise -> remember ddim cuts the bs and says we're predicting all noise at every timestep
                pred_noise = self.forward(curr_x, curr_t_labels)

                ##predict x0 
                pred_x0 = (curr_x - (torch.sqrt(1-self.bar_alpha_sched[curr_t])*pred_noise))/torch.sqrt(self.bar_alpha_sched[curr_t])
                
                ##have to clamp within range of x0, otherwise this error in the model will carry on to future denoising steps
                pred_x0 = torch.clamp(pred_x0, min=x0_min, max=x0_max)
                
                ##if next timestep is 0, we're already done (-1 because zero indexed)
                if next_t == -1:
                    curr_x = pred_x0
                    break
                
                ##scale down by next_bar_alpha_sched to get the mean of x_next_t
                mean_x_next = torch.sqrt(self.bar_alpha_sched[next_t])* pred_x0
                
                if not ddpm:
                    assert var == 0 and sigma == 0
                ##add back the std accumulation to x_next to get an estimate of true x_next
                x_next_t_std = (torch.sqrt(1-self.bar_alpha_sched[next_t] - var)*pred_noise)
                assert x_next_t_std.shape == mean_x_next.shape
                x_next = mean_x_next + x_next_t_std + (sigma*z)

                ##reset for next iter
                curr_x = x_next
                
                #hold the xts from ret_steps, rest is easy to understand
                if ret_steps != None:
                    if i in ret_steps and i != 0:
                        returns.append(curr_x)            
            returns.append(curr_x)
            if ret_steps == None:
                return curr_x
            else:
                return returns[::-1]