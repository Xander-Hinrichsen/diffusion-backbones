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
    def __init__(self, in_channels, mid_channels, out_channels, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_projector = nn.Linear(time_dim, in_channels)
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
        time_embedding = self.time_projector(self.pos_encoder(t, self.time_dim)).view(t.shape[0],self.in_channels,1,1)
        xb = xb + time_embedding
        return self.network(xb) + self.resid_connector(xb)

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Upsampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1))
    def forward(self, xb):
        return self.Upsampler(xb)

class UNet(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        self.beta_sched, self.alpha_sched, self.bar_alpha_sched = get_ddpm_schedules(T=self.T)
        ##ENCODER instance vars     beta_sched, alpha_sched, bar_alpha_sched      #32x32
        self.encodeB = ResBlock(3,64,64)   
        self.MaxPoolB = nn.MaxPool2d(2)   #16x16
        self.encodeM = ResBlock(64,128,128)
        self.MaxPoolM = nn.MaxPool2d(2)   #8x8
        self.encodeS = ResBlock(128,256,256)
        self.MaxPoolS = nn.MaxPool2d(2)   #4x4
        
        ##Connector
        self.Connector = ResBlock(256,512,512) ##should be 4 x 4 with 512 feature maps
        
        ##Decoder instance vars - notice inputs are twice as big as should be for the convblock, normally
        ##This is because of the concatonation that happens with UNet 
        self.upsamplerS = Upsampler(512, 256) #8x8
        self.decodeS = ResBlock(512, 256, 256)
        self.upsamplerM = Upsampler(256, 128) #16x16
        self.decodeM = ResBlock(256, 128, 128)
        self.upsamplerB = Upsampler(128, 64)  #32x32
        self.decodeB = ResBlock(128, 64, 64)
        
        ##from #feature maps to 3 channels for image reconstruction from latent space
        self.to_img = nn.Conv2d(64, 3, kernel_size=1)
        
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
        return self.to_img(xb)

    def DDPM_Sample(self, num_imgs=1, t=1000, res=(32,32), upper_bound=False, probabilistic=True,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.to(device)
        self.eval()
        self.bar_alpha_sched.to(device)
        self.alpha_sched.to(device)
        with torch.no_grad():
            dist = Normal(torch.zeros(num_imgs, 3, res[0], res[1]), torch.ones(num_imgs, 3, res[0], res[1]))
            curr_x = dist.sample().to(device)
            for i in range(t)[::-1]:
                z = dist.sample().to(device)
                sigma = 0
                if i!=0 and probabilistic:
                    if upper_bound:
                        sigma = self.beta_sched[i]
                    else:
                        sigma = ((1-self.bar_alpha_sched[i-1])/(1-self.bar_alpha_sched[i]))*self.beta_sched[i]

                curr_x = (1/torch.sqrt(self.alpha_sched[i]) * 
                          (curr_x - (((1-self.alpha_sched[i])/(torch.sqrt(1-self.bar_alpha_sched[i])))* self.forward(curr_x, torch.tensor([i]).to(device).repeat(num_imgs))))) + (sigma*z)
            return curr_x