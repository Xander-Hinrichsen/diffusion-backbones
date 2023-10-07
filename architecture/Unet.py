#unet implementation taken from my cityscapes segmentation, so we know it works well.
##augmentations to this -> have to add in the transformer sinusoildal encodings into ever resblock
##ended up doing time encoding between the channels like the paper did originally. I'm a sellout
##original unet from my work had 5 levels, so it reduced res side length by 2^5. Changing to 3 levels for cifar10: 32x32 -> 4x4 -> 32x32

import torch
import torch.nn as nn
from DDPM_Math import pos_encoding, get_ddpm_schedules
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, t_dim=64):
        super().__init__()
        self.time_dim = t_dim
        self.time_projector = nn.Linear(t_dim, in_channels)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.resid_connector = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.in_channels = in_channels
        self.pos_encoder = pos_encoding()
    def forward(self, xb, t):
        #time embedding works like an extra bias for the convolutions of previous resblock
        ##output will be (b, in_channels)
        #after unsqueezing this becomes (b, in_channels, 1, 1), as to broadcaast along channels
        time_embedding = self.time_projector(self.pos_encoder(t, self.time_dim)).unsqueeze(-1).unsqueeze(-1)
        xb = xb + time_embedding
        return self.network(xb) + self.resid_connector(xb)

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1)
        self.resizer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, xb):
        xb = self.upsampler(xb)
        return self.conv(xb) + self.resizer(xb)
    
##have to make my own sequential, because we have to pass t through our resblocks
class Sequential(nn.Module):
    def __init__(self, layers):
        """
        layers must be a tuple of the layers
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, xb, t):
        for i in range(len(self.layers)):
            xb = self.layers[i](xb, t)
        return xb

class UNet(nn.Module):
    def __init__(self, T=1000, t_dim=64):
        super().__init__()
        self.T = T
        self.beta_sched, self.alpha_sched, self.bar_alpha_sched = get_ddpm_schedules(T=self.T)
        ##ENCODER instance vars     beta_sched, alpha_sched, bar_alpha_sched      #32x32
        self.encodeB = Sequential((ResBlock(3,32,32), ResBlock(32,64,64)))
        self.MaxPoolB = nn.MaxPool2d(2)   #16x16
        self.encodeM = Sequential((ResBlock(64,128,128),ResBlock(128,256,256)))
        self.MaxPoolM = nn.MaxPool2d(2)   #8x8
        self.encodeS = Sequential((ResBlock(256,256,256), ResBlock(256,512,512)))
        self.MaxPoolS = nn.MaxPool2d(2)   #4x4
        
        ##Connector
        self.Connector = Sequential((ResBlock(512,512,512),ResBlock(512,512,1024))) ##should be 4 x 4 with 512 feature maps
        
        ##Decoder instance vars - notice inputs are twice as big as should be for the convblock, normally
        ##This is because of the concatonation that happens with UNet 
        self.upsamplerS = Upsampler(1024, 512) #8x8
        self.decodeS = Sequential((ResBlock(1024, 512, 512), ResBlock(512, 512, 512)))
        self.upsamplerM = Upsampler(512, 256) #16x16
        self.decodeM = Sequential((ResBlock(512, 256, 256), ResBlock(256, 128, 128)))
        self.upsamplerB = Upsampler(128, 64)  #32x32
        self.decodeB = Sequential((ResBlock(128, 64, 64),ResBlock(64, 64, 64)))
        
        ##from #feature maps to 3 channels for image reconstruction from latent space
        self.to_img = ResBlock(64, 64, 3)
        
    def forward(self, xb, t):
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

    def DDPM_Sample(self, num_imgs=1, t=1000, res=(32,32), upper_bound=False, probabilistic=True,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                     ret_steps=None):
        self.to(device)
        self.eval()
        self.bar_alpha_sched.to(device)
        self.alpha_sched.to(device)
        returns = []
        with torch.no_grad():
            dist = Normal(torch.zeros(num_imgs, 3, res[0], res[1]), torch.ones(num_imgs, 3, res[0], res[1]))
            curr_x = dist.sample().to(device)
            for i in range(t)[::-1]:
                z = dist.sample().to(device)
                var = 0
                sigma = 0
                if i!=0 and probabilistic:
                    if upper_bound:
                        var = self.beta_sched[i]
                    else:
                        var = ((1-self.bar_alpha_sched[i-1])/(1-self.bar_alpha_sched[i]))*self.beta_sched[i]
                    sigma = torch.sqrt(var)

                curr_x = (1/torch.sqrt(self.alpha_sched[i]) * 
                          (curr_x - (((1-self.alpha_sched[i])/(torch.sqrt(1-self.bar_alpha_sched[i])))* self.forward(curr_x, torch.tensor([i]).to(device).repeat(num_imgs))))) + (sigma*z)
                if ret_steps != None:
                    if i in ret_steps and i != 0:
                        returns.append(curr_x)
            returns.append(curr_x)
            if ret_steps == None:
                return curr_x
            else:
                return returns[::-1]
            
    def DDPM_Sample(self, num_imgs=1, total_steps=100, n=1, upper_bound=False,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                     ret_steps=None):
        pass