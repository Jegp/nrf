from collections import defaultdict
from typing import Union, Optional, List
from dataclasses import dataclass
import torch
import norse.torch as norse


@dataclass
class SpatialLayerParameters:
    rf_parameters: torch.Tensor
    kernel_size: int
    channels_in: int
    channels_out: int
    init_scheme: str
    device: str
    padding: str = "same"


@dataclass
class SpatioTemporalRFParameters:
    spatial_layer: Union["SpatialRF", SpatialLayerParameters]
    temporal_layer: Optional["TemporalRF"]
    activation: str
    init_scheme: str
    tau: float

    device: str


@dataclass
class SpatioTemporalChannelParameters:
    spatial_layers: List[Union["SpatialRF", SpatialLayerParameters]]
    temporal_layer: Optional["TemporalRF"]
    channels_out: int
    tau: float

    activation: str
    init_scheme: str

    device: str
    padding: str = "same"


@dataclass
class SpatioTemporalModelParameters:
    n_temporal_scales: int
    n_spatial_scales: int
    n_angles: int
    n_angles_grow: bool
    n_ratios: int
    n_derivatives: int
    separate_spatial_channels: bool
    activation: str
    init_scheme: str
    channels_in: int
    n_classes: int
    resolution: int
    weight_sharing: bool

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
            p = norse.LIFBoxParameters(
                tau_mem_inv=self.tau_mem_inv,
                v_th=torch.tensor([0.1], device=device),
                alpha=torch.tensor([10.0], device=device),
            )
            temporal_layers.append(norse.LIFBoxCell(p, dt=0.001))
        elif activation.lower() == "li":
            self.register_parameter(
                "tau_mem_inv",
                torch.nn.Parameter(torch.as_tensor(tau, device=device).float()),
            )
            p = norse.LIBoxParameters(tau_mem_inv=self.tau_mem_inv)
            temporal_layers.append(norse.LIBoxCell(p, dt=0.001))
            temporal_layers.append(torch.nn.ReLU())
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        self.temporal = norse.SequentialState(*temporal_layers)

    def forward(self, x: torch.Tensor, state=None):
        x, state = self.temporal(x, state)
        return x, state


class SpatialRF(torch.nn.Module):

    def __init__(self, p: SpatialLayerParameters):
        """
        A single spatial receptive field.
        """
        super().__init__()
        self.spatial = torch.nn.Conv2d(
            p.channels_in,
            p.channels_out,
            kernel_size=p.kernel_size,
            bias=False,
            padding=p.padding,
        ).to(p.device)
        if p.init_scheme == "rf":
            weights = norse.functional.receptive_field.spatial_receptive_fields_with_derivatives(
                p.rf_parameters, p.kernel_size, domain=5
            )
            if self.spatial.weight.shape[0] > self.spatial.weight.shape[1]:
                weights = weights.unsqueeze(1).repeat(1, p.channels_in, 1, 1)
            else:
                weights = weights.unsqueeze(0).repeat(p.channels_out, 1, 1, 1)
            self.spatial.weight.data = weights
        self.channels_out = p.channels_out

    def forward(self, x: torch.Tensor):
        return self.spatial(x)


class SpatioTemporalRF(torch.nn.Module):

    def __init__(self, p: SpatioTemporalRFParameters):
        """
        A single receptive field for a spatio-temporal layer.
        """
        super().__init__()

        self.downsample = torch.nn.AvgPool2d(2)
        self.spatial = (
            p.spatial_layer
            if isinstance(p.spatial_layer, SpatialRF)
            else SpatialRF(p.spatial_layer)
        )
        self.temporal = (
            p.temporal_layer
            if p.temporal_layer is not None
            else TemporalRF(p.tau, p.activation, p.init_scheme, p.device)
        )
        # self.bn = torch.nn.BatchNorm2d(self.spatial.channels_out)

    def forward(self, x: torch.Tensor, state=None):
        x = self.downsample(x)
        x = self.spatial(x)
        # x = self.bn(x)
        x, state = self.temporal(x, state)
        return x, state


class SpatioTemporalChannel(torch.nn.Module):

    def __init__(self, p: SpatioTemporalChannelParameters):
        """
        A channel with multiple receptive fields.
        """
        super().__init__()
        layer_parameters = [
            SpatioTemporalRFParameters(
                spatial_layer=l,
                temporal_layer=p.temporal_layer,
                activation=p.activation,
                init_scheme=p.init_scheme,
                tau=p.tau,
                device=p.device,
            )
            for l in p.spatial_layers
        ]
        self.spatiotemporal = norse.SequentialState(
            *[SpatioTemporalRF(l) for l in layer_parameters],
        )
        self.output_layer = torch.nn.AvgPool2d(2)
        # self.output_layers = torch.nn.Sequential(
        #     torch.nn.AvgPool2d(2),
        #     torch.nn.Conv2d(
        #         layer_parameters[-1].spatial_p.channels_out,
        #         p.channels_out,
        #         kernel_size=5,
        #         bias=False,
        #         padding=p.padding,
        #     ),
        # ).to(p.device)

    def forward(self, x: torch.Tensor, state=None):
        activations, state = self.spatiotemporal(x, state)
        return self.output_layer(activations), state, activations
        # return self.output_layers(activations), state, activations


class SpatioTemporalModel(torch.nn.Module):

    def __init__(self, p: SpatioTemporalModelParameters):
        """
        A model with multiple spatio-temporal channels.
        """
        super().__init__()
        channel_taus = (
            1000
            / norse.functional.receptive_field.temporal_scale_distribution(
                p.n_temporal_scales, min_scale=30, c=2  # Corresponds to < 0.2
            ).to(p.device)
        )
        if p.init_scheme == "uniform":  # Uniform channel taus
            channel_taus.uniform_(channel_taus.min(), channel_taus.max())

        scales = (2 ** torch.arange(p.n_spatial_scales)).float()

        def calculate_spatial_parameters(
            scales_specific: torch.Tensor,
        ):
            angles = torch.linspace(
                0, torch.pi * 2 - torch.pi / p.n_angles, p.n_angles
            ).float()
            ratios = (2 ** torch.arange(p.n_ratios)).float().sqrt()
            fields = norse.functional.receptive_field.spatial_parameters(
                scales_specific,
                angles,
                ratios,
                p.n_derivatives,
                x=torch.tensor([0.0]),
                y=torch.tensor([0.0]),
            ).to(p.device)
            if p.n_angles_grow > 0:
                new_fields = []
                for i in range(1, p.n_ratios):
                    new_fields.append(
                        norse.functional.receptive_field.spatial_parameters(
                            scales_specific,
                            torch.linspace(
                                0,
                                torch.pi * 2 - torch.pi / p.n_angles,
                                p.n_angles + p.n_angles_grow * i,
                            ).float(),
                            ratios[i].unsqueeze(0),
                            p.n_derivatives,
                            x=torch.tensor([0.0]),
                            y=torch.tensor([0.0]),
                        ).to(p.device)
                    )
                new_fields = torch.cat(new_fields)
                fields = torch.cat([fields, new_fields]).unique(dim=0)
            return fields

        # Create spatial receptive fields
        spatial_layers = []
        kernel_sizes = [5, 5, 3][: p.channel_layers]
        if p.separate_spatial_channels:
            for scale in scales:
                layer_parameters = []
                for i in range(p.channel_layers):
                    channels_in = (
                        p.channels_in if i == 0 else layer_parameters[-1].channels_out
                    )
                    spatial_parameters = calculate_spatial_parameters(
                        scale.unsqueeze(0)
                    )
                    spatial_p = SpatialLayerParameters(
                        spatial_parameters,
                        kernel_size=kernel_sizes[i],
                        channels_in=channels_in,
                        channels_out=(
                            3 if i == p.channel_layers - 1 else len(spatial_parameters)
                        ),
                        init_scheme=p.init_scheme,
                        device=p.device,
                    )
                    if p.weight_sharing:
                        layer_parameters.append(SpatialRF(spatial_p))
                    else:
                        layer_parameters.append(spatial_p)

                spatial_layers.append(layer_parameters)
        else:
            spatial_parameters = calculate_spatial_parameters(scales)
            layer_parameters = []
            for i in range(p.channel_layers):
                channels_in = (
                    p.channels_in if i == 0 else layer_parameters[-1].channels_out
                )
                spatial_p = SpatialLayerParameters(
                    spatial_parameters,
                    kernel_size=kernel_sizes[i],
                    channels_in=channels_in,
                    channels_out=(
                        3 if i == p.channel_layers - 1 else len(spatial_parameters)
                    ),
                    init_scheme=p.init_scheme,
                    device=p.device,
                )
                if p.weight_sharing:
                    layer_parameters.append(SpatialRF(spatial_p))
                else:
                    layer_parameters.append(spatial_p)
            spatial_layers.append(layer_parameters)

        # Create spatio-temporal channels
        channels = []
        for tau in channel_taus:
            for spatial_rfs in spatial_layers:
                st_p = SpatioTemporalChannelParameters(
                    spatial_layers=spatial_rfs,
                    temporal_layer=(
                        TemporalRF(tau, p.activation, p.init_scheme, p.device)
                        if p.weight_sharing
                        else None
                    ),
                    channels_out=p.n_classes,
                    tau=tau,
                    activation=p.activation,
                    init_scheme=p.init_scheme,
                    padding=p.padding,
                    device=p.device,
                )
                channels.append(SpatioTemporalChannel(st_p))
        self.channels = torch.nn.ModuleList(channels).to(p.device)

        classifier_activation = "li" if p.activation == "lif" else p.activation
        self.classifier = norse.SequentialState(
            torch.nn.ConvTranspose2d(p.n_classes, p.n_classes, 5, bias=False),
            TemporalRF(
                channel_taus.max(), classifier_activation, p.init_scheme, p.device
            ),
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
                channel_out, channel_state[i], channel_activation = channel(
                    t, channel_state[i]
                )
                channel_outputs.append(channel_out)
                channel_activations[i].append(channel_activation.mean())

            channel_outputs = torch.stack(channel_outputs, dim=1)
            channel_merged = channel_outputs.mean(dim=1)  # Average the channels
            classifier_out, classifier_state = self.classifier(
                channel_merged, classifier_state
            )
            output_stack.append(classifier_out)

        activations = [torch.stack(v).mean() for v in channel_activations.values()]

        return (
            torch.stack(output_stack),
            (channel_state, classifier_state),
            activations,
        )
