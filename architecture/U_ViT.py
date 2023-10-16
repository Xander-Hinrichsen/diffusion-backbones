##Vanilla vit implementation
##note that once the vit is instantiated with an image patch and resultion determined, these cannot be changed
##first step is to split image into patches
##Entire Vit order of components and operations, in pytorch terms:
#1. break image into patches
#2. use shared linear projection into dimension of each model, for each patch (project into d_model)
#3. prepend the time token across the entire batch
#4. add positional embeddings
#5. layernorm
#6. feed through transformer encoder -> that uses long skip connections
#7. layernorm
#8. linear projection of (B, num_tokens-1, d_model) back to (B, num_tokens-1, c*patch_h*patch_w)
#9. undo patchify to get (B, c, height, width)
#10. conv 3x3 -> maybe try with a residual connection

##apply layer norm before the encoder and before the classification head

import torch
import torch.nn as nn
from DDPM_Math import get_ddpm_schedules
from torch.distributions import Normal


class U_ViT(nn.Module):
    def __init__(self, img_res=(32,32), patch_size=(2,2), img_channels=3,
                d_model=512, dim_feedforward=2048, nhead=8, num_layers=17,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                T=1000, sched_type='cosine'):
        super().__init__()
        self.device = device
        ##ensure imsize % patch size == 0
        assert (img_res[0]*img_res[1]) % (patch_size[0]*patch_size[1]) == 0
        ##ensure num_layers is odd and at least 3
        assert (num_layers % 2) == 1 and num_layers >= 3 
        
        print("U-ViT instantiated")
        self.sched_type = sched_type
        self.T = T
        self.beta_sched, self.alpha_sched, self.bar_alpha_sched = get_ddpm_schedules(T=self.T, type=sched_type)

        ##number of tokens equals num_patches +1 (+1 because of time token)
        self.num_tokens = ((img_res[0]*img_res[1]) // (patch_size[0]*patch_size[1])) + 1

        ########### patches and their linear projector ###########
        def patchify(batch, patch_size=patch_size):
            """
                input: must be a batch of images (B, C, H, W)
                returns: batch of images cut into flattened patches
                -> final result: (b, H//patch_h* W//patch_w, C*patch_h*patch_w)
                -> elements from each channel kept contiguous in final flat result
            """
            assert len(batch.shape) == 4

            ##slicing batch into patches
            batch = batch.unfold(-2,patch_size[0], patch_size[0]).unfold(-2,patch_size[1], patch_size[1])

            ##shape is now (B, C, H', W', patch_h, patch_w)
            ##repermute so you have (B, H', W', C, patch_h, patch_w)
            ##.contiguous() necessary for .view()
            batch = batch.permute(0,2,3,1,4,5).contiguous()
            
            ##shape is now (B, H', W', C, patch_h, patch_w)
            ##view in (B, num_tokens, C*patch_elements)
            return batch.view(batch.shape[0], batch.shape[1]*batch.shape[2], batch.shape[3]*patch_size[0]*patch_size[1])
        self.patchify = patchify

        self.patch_projector = nn.Linear(patch_size[0]*patch_size[1]*img_channels, d_model)

        ########### time tokens ###########
        self.time_tokens = nn.Embedding(T,d_model)

        ########### Learned Position Embeddings ###########
        self.pos_embeddings = nn.Embedding(self.num_tokens, d_model)

        ########### Layer norm for transformer input ###########
        self.input_norm = nn.LayerNorm(d_model)

        ########### U-ViT Encoder ###########
        num_encoder_layers = num_layers // 2
        self.encoder = nn.ModuleList()
        for i in range(num_encoder_layers):
            self.encoder.append(nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, batch_first=True))

        ########### U-ViT Connector ###########
        self.connector = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, batch_first=True)

        ########### U-Vit connection projections ###########
        self.connect_linears = nn.ModuleList()
        for i in range(num_encoder_layers):
            self.connect_linears.append(nn.Linear(2*d_model, d_model))

        ########### U-ViT 'Decoder' ###########
        num_decoder_layers = num_encoder_layers
        self.decoder = nn.ModuleList()
        for i in range(num_decoder_layers):
            self.decoder.append(nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, batch_first=True))

        ########### Layer norm for transformer output ###########
        self.output_norm = nn.LayerNorm(d_model)

        ########### Linear Projection back to image dimensions ###########
        self.mlp_head = nn.Linear(d_model, img_channels*patch_size[0]*patch_size[1])

        ########### Undo patchify ###########
        self.folder = torch.nn.Fold(output_size=img_res, kernel_size=patch_size, stride=patch_size)
        def undo_patchify(batch, img_res=img_res, img_channels=3, patch_size=patch_size, folder=self.folder):
            """
                input: must be a batch of (B, num_tokens, C*patch_elements)
                returns: restoration of image tensor shape (B,C,H,W)
            """
            h_prime = img_res[0] // patch_size[0]
            w_prime = img_res[1] // patch_size[1]

            ##shape of input is (B, num_tokens, C*patch_elements)
            ##we want (B, H', W', C, patch_h, patch_w)
            batch = batch.view(batch.shape[0], h_prime,w_prime, img_channels, patch_size[0],patch_size[1])

            ##we want (B, C, patch_h, patch_w, H', W')
            batch = batch.permute(0,3,4,5,1,2)
            
            ##shape is now (B, C, patch_h*patch_w, H'*W') for the folder - just how it handles it
            ##we want      (B, C* patch_h*patch_w, H'*W') for the folder - just how it handles it
            batch = batch.view(batch.shape[0], img_channels*patch_size[0]*patch_size[1],h_prime*w_prime)
            ##shape is now (B, C*patch_h*patch_w, H'*W') for the folder - just how it handles it
            batch = folder(batch)
            ##shape is now (B, C, height, width)
            return batch
        self.undo_patchify = undo_patchify
        
        ########### final Conv for output ###########
        self.conv = nn.Conv2d(3,3, kernel_size=3, padding=1)

    def forward(self, imgs, t):
        #1. break image into patches
        patches = self.patchify(imgs)
        #2. use shared linear projection into dimension of each model, for each patch (project into d_model)
        patch_embeddings = self.patch_projector(patches)
        #3. prepend the time token across the entire batch cat (B, 1, d_model), (B, num_patches, d_model)
        time_tokens = self.time_tokens(t).unsqueeze(1)
        embeddings = torch.cat((time_tokens, patch_embeddings), dim=1)
        #4. add positional embeddings
        embeddings+= self.pos_embeddings(torch.arange(self.num_tokens).to(self.device))
        #5. layernorm
        encoder_in = self.input_norm(embeddings)

        #6. feed through transformer encoder -> that uses long skip connections
        encoder_outs = []
        for encoder_layer in self.encoder:
            encoder_in = encoder_layer(encoder_in)
            encoder_outs.append(encoder_in)

        decoder_in = self.connector(encoder_outs[-1])
        
        for i in range(len(self.decoder)):
            decoder_in = self.connect_linears[i](torch.cat((encoder_outs.pop(), decoder_in), dim=2))
            decoder_in = self.decoder[i](decoder_in)

        #7. layernorm
        decoder_out = self.output_norm(decoder_in)

        #8. linear projection of (B, num_tokens-1, d_model) back to (B, num_tokens-1, c*patch_h*patch_w)
        ## we don't want the time token as output
        mlp_out = self.mlp_head(decoder_out[:,1:,:])
        ##transformer out shape (B, num_tokens, d_model), grab just class tokens

        #9. undo patchify to get (B, C, height, width)
        mlp_out = self.undo_patchify(mlp_out)

        #10. conv 3x3 -> maybe try with a residual connection
        return self.conv(mlp_out) + mlp_out
    
    def DDIM_Sample(self, num_imgs=1, steps=1000, res=(32,32), x0_min=-1, x0_max=1, ddpm=False, upper_bound=False,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    ret_steps=None):
        """
            DDIM sampling at an arbritrary number of denoising steps
            DDPM sampling can be turned on by setting ddpm=True
            DDPM sampling uses full T=1000 denoising steps and adds back noise perturbations as usual
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
