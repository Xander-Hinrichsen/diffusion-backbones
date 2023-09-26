##images are expected to be scaled linearly to [-1,1]
##basic imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import wandb

##dataset/visualization 
import torchvision
import torchvision.transforms as tfs
import numpy as np

##my algos
from DDPM_Math import closed_forward_diffusion
from architecture.Unet import UNet
from DDPM_Utils import unroll_samples

##cifar10 dataset easy import
train_dataset = torchvision.datasets.CIFAR10(root='./data', transform=tfs.ToTensor(), train=True, download=True)

##device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##wandb
wandb.init(project="vanilla_ddpm")

##hyperparameters
##just going to use the hyperparams from the paper, optim is adam
epochs = 10000
lr = 2e-4
batch_size=128
dl = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
model = UNet()
forward_diffuser = closed_forward_diffusion(model.bar_alpha_sched)

hyperparams = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "T":1000, "beta_schedule": "1e-4 to 0.02"}
wandb.log(hyperparams)
##simple loss from paper
loss_fn = nn.MSELoss()

##training loop
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
loss_per_epoch = []
for epoch in range(epochs):
    epoch_loss = []
    model.train()
    for x_0, _ in dl:
        ##x_0 is expected to be [-1,1], ToTensor() used before made it [0,1]
        x_0 = x_0.to(device)
        x_0 = (2*x_0)-1

        ##sample timesteps of shape (b)
        t = torch.randint(0, 1000, (x_0.shape[0],))
        t = t.to(device)

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

        epoch_loss.append(loss.mean().item())
        del(x_0);del(x_t);del(t);del(epsilon)
        torch.cuda.empty_cache()
    
    loss_per_epoch.append(np.mean(epoch_loss))
    wandb.log({"epoch loss": loss_per_epoch[epoch]})

    ##evaluate model
    model.eval()
    sample_time_list = model.DDPM_Sample(num_imgs=10, ret_steps=[0, 10, 20, 40, 60, 80, 100, 150, 250, 350, 500, 750, 1000])
    mantage_img = unroll_samples(sample_time_list)
    ##convert from torch [-1,1] to Image [0,255]
    wandb_friendly_img = Image.fromarray(np.array(((mantage_img.detach().cpu()+1)/2).permute(1,2,0)*255, dtype=np.uint8))
    wandb.log({"epoch rollout": wandb.Image(wandb_friendly_img)})
    if epoch % 20 == 0:
        torch.save(model.state_dict(), "trained_models/epoch" + str(epoch) + ".pth")