from typing import Tuple
import torch
from functools import reduce
import norse.torch as snn


def normalized_linspace(length, dtype=None, device=None):
    if isinstance(length, torch.Tensor):
        length = length.to(device, dtype)
    first = -(length - 1.0) / length
    return torch.arange(length, dtype=dtype, device=device) * (2.0 / length) + first


def normal_pdf(x, mu, sigma):
    return (torch.pi * sigma) * torch.exp(-0.5 * ((x - mu.unsqueeze(-1)) / sigma) ** 2)


def make_gauss(means, shape, sigma, normalize=True):
    dims = [
        torch.linspace(-1, 1, s).repeat(*means.shape[:-1], 1).to(means.device)
        for s in shape
    ]
    images = [normal_pdf(dim, mu, sigma) for dim, mu in zip(dims, means.unbind(-1))]
    gauss_list = [x.unsqueeze(len(means.shape) - i) for i, x in enumerate(images)]
    gauss = reduce(lambda a, b: torch.mul(a, b), gauss_list)

    if normalize:
        return gauss / gauss.sum()
    else:
        return gauss


class JensenShannonLoss(torch.nn.Module):
    def forward(self, x, y):
        reg = 0
        for tx, ty in zip(x, y):
            reg = reg + _js(tx, ty, 2)
        return reg


class KLLoss(torch.nn.Module):
    def forward(self, x, y):
        reg = 0
        for tx, ty in zip(x, y):
            reg = reg + _kl(tx, ty, 2)
        return reg


class VarianceLoss(torch.nn.Module):
    def forward(self, x, y):
        reg = 0
        for tx, ty in zip(x, y):
            reg = reg + (tx.var() - ty.var()).abs().mean()
        return reg


def _kl(p, q, ndims):
    eps = 1e-24
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims), unsummed_kl)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


class DSNT(torch.nn.Module):
    def __init__(self, resolution: Tuple[int, int]):
        super().__init__()
        self.resolution = resolution
        self.probs_x = (
            torch.linspace(-1, 1, resolution[1]).repeat(resolution[0], 1).flatten()
        )
        self.probs_y = (
            torch.linspace(-1, 1, resolution[0]).repeat(resolution[1], 1).T.flatten()
        )

    def forward(self, x: torch.Tensor, s=None):
        if not x.device == self.probs_x.device:
            self.probs_x = self.probs_x.to(x.device)
            self.probs_y = self.probs_y.to(x.device)
        co_1 = (x.flatten(-2) * self.probs_x).sum(-1)
        co_2 = (x.flatten(-2) * self.probs_y).sum(-1)

        return torch.stack((co_2, co_1), -1), None


class DSNTLI(torch.nn.Module):
    def __init__(self, resolution: Tuple[int, int]):
        super().__init__()
        self.resolution = resolution
        self.probs_x = (
            torch.linspace(-1, 1, resolution[1]).repeat(resolution[0], 1).flatten()
        )
        self.probs_y = (
            torch.linspace(-1, 1, resolution[0]).repeat(resolution[1], 1).T.flatten()
        )
        self.li_tm = torch.nn.Parameter(torch.tensor([0.9, 0.9]))
        # self.lin = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor, state=None):
        if not x.device == self.probs_x.device:
            self.probs_x = self.probs_x.to(x.device)
            self.probs_y = self.probs_y.to(x.device)
        co_1 = (x.flatten(-2) * self.probs_x).sum(-1)
        co_2 = (x.flatten(-2) * self.probs_y).sum(-1)

        cos = torch.stack((co_2, co_1), -1)
        if state is None:
            state = torch.zeros(2, device=x.device)

        out = []
        for t in cos:
            state = state - (state * self.li_tm) + t
            out.append(state.clone())

        return torch.stack(out), state


class PixelActivityToCoordinate(torch.nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    def image_to_normalized(self, coordinate):
        return (
            (coordinate * 2 + 1)
            / torch.tensor(self.resolution, device=coordinate.device)
        ) - 1

    def forward(self, x: torch.nn.Module, _: torch.nn.Module, y_im: torch.nn.Module):
        maxes = x.flatten(-2).argmax(-1)
        rows = maxes % self.resolution[0]
        columns = maxes // self.resolution[0]
        co_pixel = torch.cat((rows, columns), -1)
        co = self.image_to_normalized(co_pixel)
        loss = torch.nn.functional.l1_loss(x, y_im)
        return co, loss
