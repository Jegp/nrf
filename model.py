from collections import defaultdict
from typing import Any, List, Tuple, Sequence
from dataclasses import dataclass
import torch
import norse.torch as norse


@dataclass
class SpatialLayerParameters:
    rf_parameters: torch.Tensor
    kernel_size: int
    channels_in: int
    padding: str = "same"

    @property
    def channels_out(self):
        return len(self.rf_parameters)


@dataclass
class SpatioTemporalRFParameters:
    spatial_p: SpatialLayerParameters
    activation: str
    init_scheme: str
    tau: float

    device: str = "cuda"


@dataclass
class SpatioTemporalChannelParameters:
    spatial_layers: List[SpatialLayerParameters]
    channels_out: int
    tau: float

    activation: str
    init_scheme: str

    padding: str = "same"
    device: str = "cuda"


@dataclass
class SpatioTemporalModelParameters:
    n_scales: int
    n_angles: int
    n_ratios: int
    n_derivatives: int
    activation: str
    init_scheme: str
    channels_in: int
    n_classes: int
    resolution: int

    channel_layers: int = 2
    padding: str = "same"
    device: str = "cuda"


class TemporalRF(torch.nn.Module):

    def __init__(self, tau: float, activation: str, init_scheme: str, device: str):
        """
        A single temporal receptive field.
        """
        super().__init__()
        temporal_layers = []
        if activation.lower() == "relu":
            temporal_layers.append(torch.nn.ReLU())
        elif activation.lower() == "lif":
            self.register_parameter(
                "tau_mem_inv",
                torch.nn.Parameter(torch.as_tensor(tau, device=device).float()),
            )
            p = norse.LIFBoxParameters(tau_mem_inv=self.tau_mem_inv, v_th=torch.tensor([0.1], device=device))
            temporal_layers.append(norse.LIFBoxCell(p))
        elif activation.lower() == "li":
            self.register_parameter(
                "tau_mem_inv",
                torch.nn.Parameter(torch.as_tensor(tau, device=device).float()),
            )
            p = norse.LIBoxParameters(tau_mem_inv=self.tau_mem_inv)
            temporal_layers.append(norse.LIBoxCell(p))
            temporal_layers.append(torch.nn.ReLU())
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        self.temporal = norse.SequentialState(*temporal_layers)

    def forward(self, x: torch.Tensor, state=None):
        x, state = self.temporal(x, state)
        return x, state


class SpatioTemporalRF(torch.nn.Module):

    def __init__(self, p: SpatioTemporalRFParameters):
        """
        A single receptive field for a spatio-temporal layer.
        """
        super().__init__()
        self.downsample = torch.nn.AvgPool2d(2)

        self.spatial = torch.nn.Conv2d(
            p.spatial_p.channels_in,
            p.spatial_p.channels_out,
            kernel_size=p.spatial_p.kernel_size,
            bias=False,
            padding=p.spatial_p.padding,
        ).to(p.device)
        if p.init_scheme == "rf":
            weights = norse.SpatialReceptiveField2d(
                p.spatial_p.channels_in,
                p.spatial_p.kernel_size,
                p.spatial_p.rf_parameters,
                p.spatial_p.padding,
                optimize_fields=False
            ).to(p.device).weights
            self.spatial.weight.data = weights

        self.temporal = TemporalRF(p.tau, p.activation, p.init_scheme, p.device).to(
            p.device
        )

    def forward(self, x: torch.Tensor, state=None):
        x = self.downsample(x)
        x = self.spatial(x)
        x, state = self.temporal(x, state)
        return x, state


class SpatioTemporalChannel(torch.nn.Module):

    def __init__(self, p: SpatioTemporalChannelParameters):
        """
        A channel with multiple receptive fields.
        """
        super().__init__()
        layer_parameters = [
            SpatioTemporalRFParameters(l, p.activation, p.init_scheme, p.tau)
            for l in p.spatial_layers
        ]
        self.spatiotemporal = norse.SequentialState(
            *[SpatioTemporalRF(l) for l in layer_parameters],
        )
        self.output_layers = torch.nn.Sequential(
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(
                layer_parameters[-1].spatial_p.channels_out,
                p.channels_out,
                kernel_size=3,
                bias=False,
                padding=p.padding,
            ),
        ).to(p.device)

    def forward(self, x: torch.Tensor, state=None):
        activations, state = self.spatiotemporal(x, state)
        return self.output_layers(activations), state, activations


class SpatioTemporalModel(torch.nn.Module):

    def __init__(self, p: SpatioTemporalModelParameters):
        """
        A model with multiple spatio-temporal channels.
        """
        super().__init__()
        channel_taus = (
            1000
            / norse.functional.receptive_field.temporal_scale_distribution(
                p.n_scales, min_scale=1.2  # Corresponds to < 913
            )
        )
        if p.init_scheme == "uniform":  # Uniform channel taus
            channel_taus.uniform_(channel_taus.min(), channel_taus.max())

        scales = (2 ** torch.arange(p.n_scales)).float()
        angles = torch.linspace(
            0, torch.pi * 2 - torch.pi / p.n_angles, p.n_angles
        ).float()
        ratios = (1 + 1.5 ** torch.arange(p.n_ratios)).float()
        derivatives = p.n_derivatives

        channels = []
        for tau in channel_taus:
            for scale in scales:
                rf_parameter = norse.functional.receptive_field.spatial_parameters(
                    scale.unsqueeze(0),
                    angles,
                    ratios,
                    derivatives,
                    x=torch.tensor([0.0]),
                    y=torch.tensor([0.0]),
                ).to(p.device)
                kernel_sizes = torch.arange(5, 2 * p.channel_layers + 5, 2).flip(0)
                layer_parameters = []
                for i in range(p.channel_layers):
                    channels_in = (
                        p.channels_in if i == 0 else layer_parameters[-1].channels_out
                    )
                    layer_p = SpatialLayerParameters(
                        rf_parameter,
                        kernel_size=kernel_sizes[i].item(),
                        channels_in=channels_in,
                    )
                    layer_parameters.append(layer_p)
                channel_p = SpatioTemporalChannelParameters(
                    layer_parameters,
                    channels_out=p.n_classes,
                    tau=tau,
                    activation=p.activation,
                    init_scheme=p.init_scheme,
                    padding=p.padding,
                    device=p.device,
                )
                channels.append(SpatioTemporalChannel(channel_p))

        self.channels = torch.nn.ModuleList(channels).to(p.device)

        classifier_activation = "li" if p.activation == "lif" else p.activation
        self.classifier = norse.SequentialState(
            torch.nn.ConvTranspose2d(p.n_classes, p.n_classes, 5, bias=True),
            torch.nn.BatchNorm2d(p.n_classes),
            TemporalRF(900, classifier_activation, p.init_scheme, p.device),
            torch.nn.Dropout(0.1),
        ).to(p.device)

        with torch.no_grad():
            self.out_shape = self.forward(
                torch.zeros(1, 1, p.channels_in, *p.resolution, device=p.device)
            )[0].shape

    def forward(self, x: torch.Tensor, state=None):
        if state is None:
            channel_state = [None] * len(self.channels)
            classifier_state = None
        else:
            channel_state, classifier_state = state

        output_stack = []
        channel_activations = defaultdict(list)
        for t in x:
            channel_outputs = []

            for i, channel in enumerate(self.channels):
                channel_out, channel_state[i], channel_activation = channel(t, channel_state[i])
                channel_outputs.append(channel_out)
                channel_activations[i].append(channel_activation.mean())

            channel_outputs = torch.stack(channel_outputs, dim=1)
            channel_merged = channel_outputs.mean(dim=1)  # Average the channels
            classifier_out, classifier_state = self.classifier(
                channel_merged, classifier_state
            )
            output_stack.append(classifier_out)

        states = [*channel_state, classifier_state]
        activations = [torch.stack(v).mean() for v in channel_activations.values()]

        return (
            torch.stack(output_stack),
            (channel_state, classifier_state),
            activations,
        )
