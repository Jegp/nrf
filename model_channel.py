from typing import Any, List, Tuple, Sequence
from functools import reduce
import math
import torch
import norse.torch as norse


def register_lif_parameters(
    p: norse.LIFBoxParameters, scale: float, shape: torch.Size
) -> Tuple[norse.LIFBoxParameters, torch.nn.ParameterList]:
    ts = torch.nn.Parameter(
        sample_time_parameters(torch.arange(2, 8), shape).to(p.tau_mem_inv.device)
    )
    tm = torch.nn.Parameter(
        sample_time_parameters(torch.arange(2, 8), shape).to(p.tau_mem_inv.device)
    )
    p = norse.LIFParameters(tau_mem_inv=tm, v_th=p.v_th)
    p_list = torch.nn.ParameterList([ts, tm])
    return p, p_list


def register_li_parameters(
    p: norse.LIBoxParameters, scale: float, shape: torch.Size
) -> Tuple[norse.LIBoxParameters, torch.nn.ParameterList]:
    delta = torch.distributions.Normal(0, scale).sample(shape).to(p.tau_mem_inv.device)
    tm = p.tau_mem_inv + delta
    param = torch.nn.Parameter(tm)
    p = norse.LIBoxParameters(tau_mem_inv=param)  # , tau_syn_inv=ts)
    return p, param


def sample_time_parameters(scales: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    dist = torch.distributions.Categorical(scales)
    return torch.exp(dist.sample(shape))


class DuplicateChannelLayer2d(torch.nn.Module):
    def __init__(self, duplications: List[int], unsqueeze: bool = True):
        super().__init__()
        self.duplications = duplications
        self.unsqueeze = unsqueeze

    def forward(self, x: torch.Tensor):
        if self.unsqueeze:
            x = x.unsqueeze(1)
        return x.repeat(1, self.duplications, 1, 1, 1)


class TemporalScaleChannel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sc,
        activation,
        activation_map,
        init_scheme,
        time_constant_scaling: float,
        max_time_constant: float,
        **kwargs,
    ):
        super().__init__()
        self.sc = sc
        time_constants = None
        t_list = None
        if activation == "ReLU":
            t_list = [
                [
                    DuplicateChannelLayer2d(1),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.05),
                ]
            ] * self.sc
        if isinstance(init_scheme, str) and (
            init_scheme == "uniform" or init_scheme.startswith("ablation")
        ):
            if activation == norse.LIBoxCell or activation == norse.LIFBoxCell:
                # Uniform
                time_constants = (
                    1000
                    / norse.functional.receptive_field.temporal_scale_distribution(
                        4,
                        c=time_constant_scaling,
                        min_scale=(1000 / max_time_constant) ** 2,
                    )
                )
                uniform_tau = torch.distributions.uniform.Uniform(
                    time_constants[-1], time_constants[0] + 1e-4
                )
                taus = uniform_tau.sample((sc,))
                time_constants = torch.stack(
                    [
                        torch.full(
                            [out_channels, 1, 1],
                            tau,
                            dtype=torch.float32,
                        )
                        for tau in taus
                    ]
                )
        elif isinstance(init_scheme, int):  # Fix time constants
            time_constants = torch.full(
                [sc, out_channels, 1, 1], init_scheme, dtype=torch.float32
            )
        else:
            time_constants = None
        if t_list is None:
            # Initialized time constants
            if time_constants is None:
                time_constants = (
                    1000
                    / norse.functional.receptive_field.temporal_scale_distribution(
                        4,
                        c=time_constant_scaling,
                        min_scale=(1000 / max_time_constant) ** 2,
                    )
                )
                if sc == 1:
                    time_constants = time_constants[-1:]
            t_list = []
            for n_tau in range(sc):
                t_list.append(
                    [
                        norse.TemporalReceptiveField(
                            (out_channels, 1, 1),
                            1,
                            activation=activation,
                            activation_state_map=activation_map,
                            time_constants=(
                                time_constants[n_tau]
                                if time_constants is not None
                                else None
                            ),
                        ),
                        (
                            torch.nn.ReLU()
                            if activation == norse.LIBoxCell
                            else torch.nn.Identity()
                        ),
                        torch.nn.Dropout(0.05),
                    ]
                )
        self.t_rfs = [norse.SequentialState(*l) for l in t_list]

    def forward(self, x, state=None):
        out = []
        if state is None:
            state = [None] * x.shape[1]
        for i, t in enumerate(x.unbind(1)):
            y, state[i] = self.t_rfs[i](t, state[i])
            out.append(y)
        return torch.concat(out, dim=1), state


class ParallelState(torch.nn.ModuleList):
    def __init__(self, *modules, dim: int = 0):
        super().__init__(modules)
        self.dim = dim

    def forward(self, x, state=None):
        out = []
        if state is None:
            state = [None] * len(self)
        for i, t in enumerate(x.unbind(self.dim)):
            z = self[i](t)
            if isinstance(z, tuple):
                state[i] = z[1]
                out.append(z[0])
            else:
                out.append(z)
        return torch.stack(out, dim=self.dim), state


class SpatioTemporalRF(torch.nn.Module):
    def __init__(
        self, c_in, sc_in, sc, size, activation, activation_map, init_scheme, **kwargs
    ):
        super().__init__()
        pool2d = torch.nn.MaxPool2d(2)
        rf_parameters = norse.functional.receptive_field.spatial_parameters(
            scales=2 ** torch.arange(kwargs.get("n_scales", 4)).float(),
            angles=torch.linspace(0, torch.pi * 2, kwargs.get("n_angles", 1)).float(),
            ratios=1 + 1.5 ** torch.arange(kwargs.get("n_ratios", 1)).float(),
            derivatives=kwargs.get("derivatives", 2),
        )
        rf = norse.functional.receptive_field.spatial_receptive_fields_with_derivatives(
            rf_parameters, size, domain=5
        )

        fields = [
            torch.nn.Conv2d(
                c_in, len(rf), kernel_size=size, bias=False, padding=kwargs["padding"]
            )
            for _ in range(sc_in)
        ]
        for f in fields:
            f.weight.data += f.weight.min()

        for field in fields:
            field.weight = torch.nn.Parameter(rf.unsqueeze(1).repeat(1, c_in, 1, 1))
        norms = [torch.nn.BatchNorm2d(c_in) for _ in range(sc_in)]

        temporal_field = TemporalScaleChannel(
            c_in,
            fields[0].out_channels,
            sc,
            activation,
            activation_map,
            init_scheme,
            learn_timeconstants=True,
            **kwargs,
        )
        self.block = norse.SequentialState(
            norse.Lift(pool2d, dim=1),
            ParallelState(*norms, dim=1),
            ParallelState(*fields, dim=1),
            temporal_field,
        )

    def forward(self, x, state=None):
        return self.block(x, state)


class ShapesRFModel(torch.nn.Module):
    def __init__(
        self,
        classes: int,
        init_scheme: str,
        resolution: torch.Tensor,
        input_frames: int = 2,  # Input channels
        activation=norse.LIBoxCell,
        activation_p=norse.LIBoxParameters(),
        classifier_p=norse.LIBoxParameters(),
        max_time_constant: float = 1000,
        time_constant_scaling: float = 2,
    ):
        super().__init__()
        kernel_size = 11
        self.shapes = [
            resolution // 2,
            (resolution // 2) // 2,
            ((resolution // 2) // 2) // 2,
            ((resolution // 2) // 2) // 2,
            ((resolution // 2) // 2) // 2 + (kernel_size - 1),
        ]
        self.out_shape = self.shapes[-1]

        if isinstance(activation_p, norse.LIBoxParameters):
            activation_map = lambda x: norse.LIBoxParameters(
                tau_mem_inv=x.to(activation_p.v_leak.device),
                v_leak=activation_p.v_leak,
            )
        elif isinstance(activation_p, norse.LIFBoxParameters):
            activation_map = lambda x: norse.LIFBoxParameters(
                tau_mem_inv=x.to(activation_p.v_leak.device),
                v_leak=activation_p.v_leak,
                v_th=activation_p.v_th
                * (x.to(activation_p.v_leak.device).detach() * 0.001),
                v_reset=activation_p.v_reset,
                method=activation_p.method,
            )
        elif activation_p is None:
            activation_map = None
        else:
            raise ValueError(f"Unknown parameter type: {activation_p}")

        self.rf0 = norse.SequentialState(
            # RF1
            SpatioTemporalRF(
                c_in=input_frames,
                sc_in=1,
                sc=1,
                size=kernel_size,
                activation=activation,
                activation_map=activation_map,
                init_scheme=init_scheme,
                n_scales=2,
                n_angles=2,
                n_ratios=2,
                derivatives=1,
                padding="same",
                max_time_constant=max_time_constant,
                time_constant_scaling=time_constant_scaling,
            ),
            DuplicateChannelLayer2d(4, unsqueeze=False),
            # RF2
            SpatioTemporalRF(
                c_in=32,
                sc_in=4,
                sc=4,
                size=kernel_size,
                activation=activation,
                activation_map=activation_map,
                init_scheme=init_scheme,
                n_scales=2,
                n_angles=3,
                n_ratios=2,
                derivatives=2,
                padding="same",
                max_time_constant=max_time_constant,
                time_constant_scaling=time_constant_scaling,
            ),
        )

        sigma = 10
        if classifier_p is None:
            classifier_activation = [torch.nn.ReLU()]
        else:
            if isinstance(init_scheme, str) and init_scheme == "ablation_low":
                classifier_p = norse.LIBoxParameters(
                    tau_mem_inv=torch.tensor(
                        [250], device=classifier_p.tau_mem_inv.device
                    ),
                    v_leak=classifier_p.v_leak,
                )
            elif isinstance(init_scheme, str) and init_scheme == "ablation_high":
                classifier_p = norse.LIBoxParameters(
                    tau_mem_inv=torch.tensor(
                        [1000], device=classifier_p.tau_mem_inv.device
                    ),
                    v_leak=classifier_p.v_leak,
                )
            self.classifier_p, self.classifier_pl = register_li_parameters(
                classifier_p, sigma, [classes, *self.out_shape]
            )
            classifier_activation = [
                norse.LIBoxCell(p=self.classifier_p),
            ]

        self.mid = norse.SequentialState(
            norse.Lift(torch.nn.MaxPool2d(2), dim=1),
            *self.conv_block(
                108,
                classes,
                4,
                kernel_size,
                activation=activation,
                activation_map=activation_map,
                init_scheme=init_scheme,
                sizes=self.shapes[-2],
                padding="same",
                max_time_constant=max_time_constant,
                time_constant_scaling=time_constant_scaling,
            ),
        )
        self.classifier = norse.SequentialState(
            torch.nn.ConvTranspose2d(4 * classes, classes, kernel_size, bias=False),
            torch.nn.BatchNorm2d(classes),
            *classifier_activation,
            torch.nn.Dropout(0.05),
        )

    @staticmethod
    def conv_block(
        c_in,
        c_out,
        sc,
        kernel_size,
        activation,
        activation_map,
        init_scheme,
        sizes,
        max_time_constant,
        time_constant_scaling,
        **kwargs,
    ):
        block = [
            norse.Lift(
                torch.nn.Conv2d(c_in, c_out, kernel_size, bias=False, **kwargs), dim=1
            ),
            TemporalScaleChannel(
                c_in,
                c_out,
                sc,
                activation,
                activation_map,
                init_scheme,
                max_time_constant=max_time_constant,
                time_constant_scaling=time_constant_scaling,
                **kwargs,
            ),
            torch.nn.Dropout(0.05),
        ]
        return block

    def forward(self, x, state=None):
        coo = []
        activity0 = []
        activity1 = []
        activity2 = []
        for t in x:
            y, state, activity = self.forward_step(t, state)
            coo.append(y)
            activity0.append(activity[0])
            activity1.append(activity[1])
            activity2.append(activity[2])
        return (
            torch.stack(coo),
            state,
            (torch.stack(activity0), torch.stack(activity1), torch.stack(activity2)),
        )

    def forward_step(self, x, state=None):
        if state is None:
            state = (None, None, None)
        sr, sm, sc = state
        z, sr = self.rf0(x.unsqueeze(1), sr)
        a, sm = self.mid(z, sm)
        # Concatenate temporal and spatial channels
        b = a.flatten(1, 2)
        # Position head
        c, sc = self.classifier(b, sc)
        return c, (sr, sm, sc), (z, a, c)
