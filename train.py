##Todo -> FID and InceptionScore

##images are expected to be scaled linearly to [-1,1]
##basic imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import wandb
import copy #for EMA model updating

##dataset/visualization 
import torchvision
import torchvision.transforms as tfs
import numpy as np

##my algos
from DDPM_Math import closed_forward_diffusion
from architecture.Unet import UNet
from DDPM_Utils import unroll_samples

##cifar10 dataset easy import
data_aug = tfs.Compose([tfs.RandomHorizontalFlip(p=0.5), tfs.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root='./data', transform=data_aug, train=True, download=True)

##device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##wandb
wandb.init(project="U-VIT-S (Deep) Unconditional Cifar-10")

##hyperparameters
##just going to use the hyperparams from the paper, optim is adam
epochs = 10000
lr = 2e-4
batch_size=128
ema_decay = 0.999
dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

##model, avg_model for ema, & unsupervised learnining 'labeler'
model = UNet(sched_type='cosine')
# avg_model = copy.deepcopy(model).to(device).eval()
forward_diffuser = closed_forward_diffusion(model.bar_alpha_sched)

hyperparams = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "T":1000}
wandb.log(hyperparams)
##simple loss from paper
loss_fn = nn.MSELoss()

##training loop
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
#sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=len(dl))
loss_per_epoch = []
for epoch in range(epochs):
    epoch_loss = []
    for i, x_0 in enumerate(dl):
        model.train()
        ##x_0 is expected to be [-1,1], ToTensor() used before made it [0,1]
        x_0 = x_0[0].to(device)
        x_0 = (2*x_0)-1

        ##sample timesteps of shape (b)
        t = torch.randint(0, 1000, (x_0.shape[0],))
        t = t.to(device)

        #training
        ##use closed formula forward diffusion
        ##x_t is our input, epsilon is our label
        x_t, epsilon = forward_diffuser(x_0,t)

        preds = model(x_t, t)
        assert preds.shape == epsilon.shape
        loss = loss_fn(preds, epsilon)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(),0.1)
        optim.step()
        #sched.step(i)

        epoch_loss.append(loss.mean().item())
        del(x_0);del(x_t);del(t);del(epsilon)
        torch.cuda.empty_cache()

        ##EMA model updating -> update our average model on moving average basis
        # model.eval()
        # with torch.no_grad():
        #     for avg_layer, fast_layer in zip(avg_model.state_dict().values(),model.state_dict().values()):
        #         avg_layer = ema_decay*avg_layer + (1-ema_decay)*fast_layer
        
        ##log the last scheduled lr
        #wandb.log({"lr_scheduled": sched.get_last_lr()[0]})
    loss_per_epoch.append(np.mean(epoch_loss))
    wandb.log({"epoch loss": loss_per_epoch[epoch]})

    ##evaluate model
    model.eval()

    #only log samples every 20 epochs, takes too long to sample otherwise
    if (epoch % 10) == 0:
        ##ddpm sampling, fast model
        sample_time_list = model.DDIM_Sample(num_imgs=10, ddpm=True, upper_bound=False, ret_steps=[0, 10, 20, 40, 60, 80, 100, 150, 250, 350, 500, 750, 1000])
        mantage_img = unroll_samples(sample_time_list)
        ##convert from torch [-1,1] to Image [0,255]
        wandb_friendly_img = Image.fromarray(np.array(((mantage_img.detach().cpu()+1)/2).permute(1,2,0)*255, dtype=np.uint8))
        wandb.log({"ddpm, upperbound=False, T=1000: 0, 10, 20, 40, 60, 80, 100, 150, 250, 350, 500, 750, 1000": wandb.Image(wandb_friendly_img)})

        # ##ddim sampling, full T
        # sample_time_list = model.DDIM_Sample(num_imgs=10, steps = model.T, ret_steps=[0, 10, 20, 40, 60, 80, 100, 150, 250, 350, 500, 750, 1000])
        # mantage_img = unroll_samples(sample_time_list)
        # ##convert from torch [-1,1] to Image [0,255]
        # wandb_friendly_img = Image.fromarray(np.array(((mantage_img.detach().cpu()+1)/2).permute(1,2,0)*255, dtype=np.uint8))
        # wandb.log({"ddim T=1000: 0, 10, 20, 40, 60, 80, 100, 150, 250, 350, 500, 750, 1000": wandb.Image(wandb_friendly_img)})
##
        ##ddim sampling, fast model, 500 steps
        sample_time_list = model.DDIM_Sample(num_imgs=10, steps=500, ret_steps=[0, 10, 15, 20, 25, 30, 40, 50, 80, 100, 200, 300, 500])
        mantage_img = unroll_samples(sample_time_list)
        ##convert from torch [-1,1] to Image [0,255]
        wandb_friendly_img = Image.fromarray(np.array(((mantage_img.detach().cpu()+1)/2).permute(1,2,0)*255, dtype=np.uint8))
        wandb.log({"ddim, T=500: 0, 10, 15, 20, 25, 30, 40, 50, 80, 100, 200, 300, 500": wandb.Image(wandb_friendly_img)})

        ##ddim sampling, fast model, 250 steps
        sample_time_list = model.DDIM_Sample(num_imgs=10, steps=250, ret_steps=[0, 5, 10, 15, 20, 30, 40, 80, 150, 250])
        mantage_img = unroll_samples(sample_time_list)
        ##convert from torch [-1,1] to Image [0,255]
        wandb_friendly_img = Image.fromarray(np.array(((mantage_img.detach().cpu()+1)/2).permute(1,2,0)*255, dtype=np.uint8))
        wandb.log({"ddim, T=250: 0, 5, 10, 15, 20, 30, 40, 80, 150, 250": wandb.Image(wandb_friendly_img)})

        ##ddim sampling, fast model, 100 steps
        sample_time_list = model.DDIM_Sample(num_imgs=10, steps=100, ret_steps=[0, 5, 10, 15, 20, 25, 30, 40, 100])
        mantage_img = unroll_samples(sample_time_list)
        ##convert from torch [-1,1] to Image [0,255]
        wandb_friendly_img = Image.fromarray(np.array(((mantage_img.detach().cpu()+1)/2).permute(1,2,0)*255, dtype=np.uint8))
        wandb.log({"ddim, T=100: 0, 5, 10, 15, 20, 25, 30, 40, 100": wandb.Image(wandb_friendly_img)})

        ##ddim sampling, fast model, 50 steps
        sample_time_list = model.DDIM_Sample(num_imgs=10, steps=50, ret_steps=[0, 2, 4, 6, 8, 16, 32, 50])
        mantage_img = unroll_samples(sample_time_list)
        ##convert from torch [-1,1] to Image [0,255]
        wandb_friendly_img = Image.fromarray(np.array(((mantage_img.detach().cpu()+1)/2).permute(1,2,0)*255, dtype=np.uint8))
        wandb.log({"ddim, T=50: 0, 2, 4, 6, 8, 16, 32, 50": wandb.Image(wandb_friendly_img)})

        ##ddim sampling, fast model, 20 steps
        sample_time_list = model.DDIM_Sample(num_imgs=10, steps=20, ret_steps=[0, 1, 2, 3, 4, 5, 7, 10, 16, 20])
        mantage_img = unroll_samples(sample_time_list)
        ##convert from torch [-1,1] to Image [0,255]
        wandb_friendly_img = Image.fromarray(np.array(((mantage_img.detach().cpu()+1)/2).permute(1,2,0)*255, dtype=np.uint8))
        wandb.log({"ddim, T=20: 0, 1, 2, 3, 4, 5, 7, 10, 16, 20": wandb.Image(wandb_friendly_img)})

    if (epoch % 50 == 0) and epoch != 0:
        # torch.save(avg_model.state_dict(), "trained_models/avg_model_epoch" + str(epoch) + ".pth")
        torch.save(model.state_dict(), "trained_models/fast_model_epoch" + str(epoch) + ".pth")
