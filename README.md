# ddpm-vanilla
ddpm from scratch to make sure I know all of the moving parts, going to only run on frogs and cars from cifar-10 with a basic resUnet. Hopefully 4090 really is that much faster than a v100, or training may take a while.

things I'm going to attempt different from / in addition to the vanilla DDPM paper:
1. Time embeddings -> they said they just add the positional encodings within each resblock in the paper -> this is a lie; they project them and do weird shit in their implementation. I'm going to try to 1. Project with learned linear layer into the resolution space, and use as an extra channel for each linear layer -> not only learn the positional encodings, but also learn how to choose how to use them in pixel space 2. can also do this without the projection and just use the positional encoding formulation to get a dim equal to the current resolution
2. Make regular inference and DDIM inference (difference of one line of code)
