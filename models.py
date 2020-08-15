from typing import List

import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt


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
        return z_loss * wz + x_loss * wx


class MIVAE(VAE):
    """mutual information loss function added as method to a vae"""
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
            # don't include zs drawn from prior losses[zi] = losses[zi].mean() * use_posteriors[zi] loss = torch.mean(torch.stack(losses)) return loss class MIVAESharedEncoder(MIVAE):
    """MIVAE with shared encoder for easier partitioning.  
    cannot use with pretrained encoders"""
    def __init__(self,
                 encoder: List[nn.Module],
                 decoder: nn.Module,
                 dim_encoder_output: int,
                 z_priors: List[D.Distribution],
                 dim_zs: List[int]) -> None:
        assert len(dim_z_priors) == len(z_priors)
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.dim_encoder_output = dim_encoder_output
        self.z_priors = z_priors
        self.dim_zs = dim_zs

        self.num_zs = len(z_priors)

        # init linear read outs for every arg for each distr z
        self.readouts = [None] * self.num_zs
        # which distribution this z is drawn from
        self.distrs = [None] * self.num_zs
        # constraints on the params for this z's distr
        self.constraints = [None] * self.num_zs

        for zi in range(self.num_zs):
            self.readouts[zi] = {}
            self.constraints[zi] = {}
            self.distrs[zi] = type(self.z_priors[zi])
            for arg, constr in z_prior.arg_constraints.items():
                # make transform from shared logits for this param for this z|x
                self.readouts[zi][arg] = nn.Linear(dim_encoder_output, self.dim_zs[zi])
                # save constraints for this param for this z's distr
                self.constraints[zi][arg] = constr

    def fit_logits_to_constraint(self, logits, constraint, small=1e-6):
        """transform logits to fit a distribution param's constraints"""
        if isinstance(constraint, D.constraints._Simplex):
            output = torch.softmax(logits, dim=-1)
        elif isinstance(constraint, D.constraints._GreaterThan):
            output = F.softplus(logits + constraint.lower_bound) + small
        elif isinstance(constraint, D.constraints._Interval):
            mul = constraint.upper_bound - constraint.lower_bound
            output = torch.sigmoid(logits) * mul + constraint.lower_bound
        elif isinstance(constraint, D.constraints._Real):
            output = logits
        return output

    def encode(self, x):
        pz_given_xs = [None] * self.num_zs
        # get shared logits read out by each z
        latent_logits = encoder(x)
        for zi in range(self.num_zs):
            zi_readouts = self.readouts[zi]
            values = {}
            for arg, transform in zi_readouts.items():
                logits = transform(latent_logits)
                values[arg] = self.fit_logits_to_constraint(
                        logits=logits, 
                        constraint=self.constraints[zi][arg])
            # init distr param'd by transformed and constrained logits for this z
            distr = self.distrs[zi]
            pz_given_xs[zi] = distr(**values)
        return pz_given_xs

    self.calc_mi_loss = MIVAE.calc_mi_loss


def viz_recon(axes, imgs, recons):
    # fig, axes = plt.subplots(8, 8 * 2)
    i, j = 0, 0
    mats = []
    for img, recon in zip(*[imgs, recons]):
        axes[i, j].clear()
        axes[i, j+8].clear()

        axes[i, j].matshow(img, vmin=0., vmax=1.)
        axes[i, j+8].matshow(recon, vmin=0., vmax=1.)
        i = (i + 1) % 8
        j = (j + 1) % 8
    plt.pause(0.01)
    plt.draw()
    plt.show()


if __name__ == '__main__':
    from torch.optim import Adam, Adamax

    from data import dataset, data_loader
    from viz import prep_ims_for_imshow

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DIM_X = 784

    DIM_ZS = [1] * 29
    num_z = len(DIM_ZS)
    dim_z = sum(DIM_ZS)
    Z_PRIORS = [D.Normal(torch.tensor(0.), torch.tensor(1.))] * num_z

    vae = MIVAESharedEncoder(
            encoder=encoder,
            decoder=decoder,
            dim_encoder_output=512,
            z_priors=Z_PRIORS,
            dim_zs=DIM_ZS)

    # encoder0 = MLP(DIM_X, DIM_ZS[0] * 2, [64])
    # encoder1 = MLP(DIM_X, DIM_ZS[1] * 2, [64])
    # encoder2 = MLP(DIM_X, DIM_ZS[2] * 2, [64])
    # encoder3 = MLP(DIM_X, DIM_ZS[3] * 2, [64])

    # vae = MIVAE(encoder: List[nn.Module],
    #              decoder: nn.Module,
    #              dim_encoder_output: int,
    #              z_priors: List[D.Distribution],
    #              dim_zs: List[int]) -> None:

    decoder = MLP(dim_z, DIM_X, [512])

    optimizer = Adamax(params=vae.parameters(), lr=1e-3)

    step = 0

    fig, axes = plt.subplots(1, 2)
    mats = [None] * 2
    mats[0] = axes[0].matshow(np.zeros([28, 28]), cmap='bone', vmin=0., vmax=1.)
    mats[1] = axes[1].matshow(np.zeros([28, 28]), cmap='bone', vmin=0., vmax=1.)

    for xx, yy in data_loader:
        vae.train()
        optimizer.zero_grad()

        xx = xx.to(device)
        x_recons, pz_given_xs = vae(xx, return_pz=True)
        nelbo = vae.calc_nelbo(xx, x_recons, pz_given_xs, wz=0.1)
        mi_loss = vae.calc_mi_loss(pz_given_xs)
        
        loss = nelbo + mi_loss * 0.1
        # TODO: adversarial net can be used to keep mi_loss recon on natural manifold of data points

        if step % 100 == 0:
            print(f'loss step {step}: {nelbo.item():.2f}')

            with torch.no_grad():
                plt.ion()
                nxs = xx.numpy().reshape(-1, 1, 28, 28)
                nxs = prep_ims_for_imshow(nxs)

                nrecons = x_recons[0].sigmoid().numpy().reshape(28, 28)
                nrecons = prep_ims_for_imshow(nrecons)

                x = xx[0].numpy().reshape(28, 28)
                mats[0].set_data(nxs)
                mats[1].set_data(nrecons)
                plt.pause(0.01)
                plt.draw()
                plt.show()
                plt.ioff()

        loss.backward()
        optimizer.step()
        step += 1
