from typing import List

import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F

class MLP(nn.Module):
    """standard fully connected mlp"""
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dim_hidden: List[int],
                 act_fn: nn.Module=nn.LeakyReLU(),
                 batch_norm: bool=False,
                 dropout: float=0.) -> None:
        super().__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.act_fn = act_fn
        self.batch_norm = batch_norm
        self.dropout = dropout

        modules = []
        dim_in = dim_input
        for dim_out in dim_hidden:
            modules.append(nn.Linear(dim_in, dim_out))
            modules.append(act_fn)
            if self.batch_norm:
                modules.append(nn.BatchNorm1d(dim_out))
            if self.dropout > 0.:
                modules.append(nn.Dropout(p=self.dropout))
            dim_in = dim_out
        modules.append(nn.Linear(dim_in, dim_output))
        
        self.model = nn.Sequential(*modules)
        self.modules = modules

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class VAE(nn.Module):

    recon_loss_fn = nn.BCEWithLogitsLoss()

    def __init__(self,
                 encoders: List[nn.Module],
                 decoder: nn.Module,
                 z_priors: List[D.Distribution]):
        super().__init__()

        self.encoders = nn.ModuleList(encoders)
        # register encoders as part of model
        # for ei, encoder in enumerate(encoders):
        #     setattr(self, f'encoder_{ei}', encoder)
        self.decoder = decoder
        self.z_priors = z_priors

        self.recon_loss_fn = nn.BCEWithLogitsLoss()

    def encode(self, x):
        pz_given_xs = []
        for encoder in self.encoders:
            latent_logits = encoder(x)
            assert latent_logits.shape[1] % 2 == 0
            size = latent_logits.shape[1] // 2
            mu, lv = latent_logits[:, :size], latent_logits[:, size:]
            pz_given_x = D.Normal(loc=mu, scale=F.softplus(lv) + 1e-4)
            pz_given_xs.append(pz_given_x)
        return pz_given_xs

    def decode(self, latents):
        decoder_input = torch.cat(latents, dim=1)
        recon = self.decoder(decoder_input)
        self.x_recons_ = recon
        return recon

    def forward(self, x, return_pz=False):
        # infer q(z|x)
        pz_given_xs = self.encode(x)
        latents = []

        # sample q(z|x)
        for pz in pz_given_xs:
            latent = pz.rsample()
            latents.append(latent)

        # get p(x|z)
        z_given_x = torch.cat(latents, dim=1)
        recon = self.decoder(z_given_x)
        if return_pz:
            return recon, pz_given_xs
        else:
            return recon

    def calc_z_loss(self, pz_given_xs):
        z_loss = torch.tensor(0.)
        for pz, qz in zip(*[pz_given_xs, self.z_priors]):
            loss = D.kl.kl_divergence(pz, qz)
            z_loss += loss.mean()
        return z_loss

    def calc_nelbo(self, x, x_recon, pz_given_xs, wx=1., wz=1.):
        z_loss = self.calc_z_loss(pz_given_xs)
        x_loss = self.recon_loss_fn(x_recon, x)
        return z_loss * wx + x_loss * wz


class MIVAE(VAE):
    def calc_mi_loss(self, 
                      pz_given_xs: List[D.Distribution], 
                      p_prior: float=0.5) -> torch.Tensor:
        "pass to get mutual info loss to be used as regularizer on top of standard vae loss"
        num_zs = len(pz_given_xs)
        # decide if using p(z) or p(z|x) per z in latent space
        use_posteriors = torch.rand(num_zs).lt(p_prior)
        z_samples = [None] * num_zs
        for zi in range(num_zs):
            if use_posteriors[zi]:
                z_samples[zi] = pz_given_xs[zi].rsample()
            else:
                z_samples[zi] = self.z_priors[zi].rsample(
                        sample_shape=pz_given_xs[zi].loc.shape)
        # generate p(x|z)
        x_recon = self.decode(z_samples)
        # re-infer
        pz_given_x_recons = self.encode(x_recon)

        # calc mi loss for things not drawn from prior
        losses = [None] * num_zs
        zi = 0
        for zi in range(num_zs):
            pz_given_x_recon = pz_given_x_recons[zi]
            pz_given_x = pz_given_xs[zi]
            losses[zi] = D.kl.kl_divergence(pz_given_x_recon, pz_given_x)
            # don't include zs drawn from prior
            losses[zi] = losses[zi].mean() * use_posteriors[zi]
        loss = torch.mean(torch.stack(losses))
        return loss


if __name__ == '__main__':
    from torch.optim import Adam, Adamax

    from data import dataset, data_loader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DIM_X = 784

    DIM_ZS = [128, 128]
    num_z = len(DIM_ZS)
    dim_z = sum(DIM_ZS)
    Z_PRIORS = [D.Normal(torch.tensor(0.), torch.tensor(1.))] * num_z

    encoder0 = MLP(DIM_X, DIM_ZS[0] * 2, [1024, 1024])
    encoder1 = MLP(DIM_X, DIM_ZS[1] * 2, [1024, 1024])
    decoder = MLP(dim_z, DIM_X, [128])

    vae = MIVAE(
            encoders=[encoder0, encoder1],
            decoder=decoder,
            z_priors=Z_PRIORS)

    optimizer = Adamax(params=vae.parameters(), lr=1e-3)

    step = 0
    for xx, yy in data_loader:
        vae.train()
        optimizer.zero_grad()

        xx = xx.to(device)
        x_recon, pz_given_xs = vae(xx, return_pz=True)
        nelbo = vae.calc_nelbo(xx, x_recon, pz_given_xs)
        mi_loss = vae.calc_mi_loss(pz_given_xs)
        # TODO: adversarial net can be used to keep mi_loss recon on natural manifold of data points

        if step % 100 == 0:
            print(f'loss step {step}: {nelbo.item():.2f}')
        nelbo.backward()
        optimizer.step()
        step += 1
