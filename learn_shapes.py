from argparse import ArgumentParser
from itertools import chain
import pathlib
from types import SimpleNamespace

import torch
import norse.torch as norse
import pytorch_lightning as pl

from datasets.datasets.dataset import ShapeDataset

from model import SpatioTemporalModel, SpatioTemporalModelParameters, TemporalRF
from loss import *
from visualization import *


def int_or_str(value):
    try:
        return int(value)
    except:
        return value


class ShapesModel(pl.LightningModule):
    def __init__(self, args: SimpleNamespace):
        super().__init__()
        self.args = args

        # Network
        n_classes = 3
        n_scales = 4
        n_angles = 3
        n_ratios = 3
        n_derivatives = 1
        if args.net == "lif":
            activation = "lif"
        elif args.net == "li":
            activation = "li"
        elif args.net == "ann":
            activation = "relu"
        else:
            raise ValueError("Unknown network type " + args.net)

        # Network
        p = SpatioTemporalModelParameters(
            n_scales=n_scales,
            n_angles=n_angles,
            n_ratios=n_ratios,
            n_derivatives=n_derivatives,
            activation=activation,
            init_scheme=args.init_scheme,
            channels_in=2 if args.sum_frames else (args.stack_frames * 2),
            resolution=args.resolution,
            n_classes=n_classes,
            channel_layers=2,
            device=args.device,
        )
        self.net = SpatioTemporalModel(p)
        self.out_shape = torch.tensor(self.net.out_shape[-2:], device=args.device)

        # Regularization
        if args.regularization == "js":
            self.regularization = JensenShannonLoss()
        elif args.regularization == "kl":
            self.regularization = KLLoss()
        elif args.regularization == "var":
            self.regularization = VarianceLoss()
        else:
            self.regularization = lambda a, b: torch.as_tensor(0.0)
        # Rectification
        if args.rectification == "softmax":
            self.rectification = torch.nn.Softmax(dim=-1)
        elif args.rectification == "sigmoid":
            self.rectification = torch.nn.Sigmoid()
        elif args.rectification == "relu":
            self.rectification = torch.nn.ReLU()
        else:
            self.rectification = torch.nn.Identity()
        self.regularization_scale = args.regularization_scale
        # Coordinate
        if args.coordinate == "dsnt":
            self.coordinate = DSNT(self.out_shape)
        elif args.coordinate == "dsntli":
            self.coordinate = DSNTLI(self.out_shape)
        else:
            self.coordinate = PixelActivityToCoordinate(args.resolution)

        self.resolution = torch.tensor(args.resolution, device=args.device)
        self.lr = args.lr
        self.lr_step = args.lr_step
        self.warmup = args.warmup
        self.optimizer = args.optimizer

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Network")
        parser.add_argument("--net", type=str)
        parser.add_argument("--n_scales", type=int)
        parser.add_argument("--n_angles", type=int, default=3)
        parser.add_argument("--n_ratios", type=int, default=2)
        parser.add_argument(
            "--regularization",
            type=str,
            choices=["none", "js", "kl", "var"],
            default="js",
        )
        parser.add_argument(
            "--rectification",
            type=str,
            choices=["softmax", "relu", "id", "sigmoid"],
            default="softmax",
        )
        parser.add_argument(
            "--regularization_scale",
            type=float,
            default=1e-4,
        )
        parser.add_argument("--regularization_activity_mean", type=float, default=1e-3)
        parser.add_argument("--regularization_activity_scale", type=float, default=1e-3)
        parser.add_argument(
            "--coordinate",
            type=str,
            choices=["dsnt", "dsntli", "pixel"],
            default="dsntli",
            help="Method to reduce 2d surface to coordinate",
        )
        parser.add_argument(
            "--init_scheme",
            type=int_or_str,
            default="rf",
            help="Init method for spatial and temporal RFs",
        )
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument(
            "--lr_step", type=str, default="step", choices=["step", "ca", "none"]
        )
        parser.add_argument("--lr_decay", type=float, default=0.95)
        parser.add_argument("--v_leak", type=float, default=0.0)
        parser.add_argument("--v_th", type=float, default=0.3)
        parser.add_argument(
            "--optimizer",
            type=str,
            choices=["adagrad", "adam", "rmsprop", "sgd", "spacetime"],
            default="adam",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-6,
        )
        return parent_parser

    def extract_kernels(self, net, f=lambda x: x.clone().detach().cpu()):
        ks = []
        for m in net.children():
            ks += self.extract_kernels(m)

        if isinstance(net, norse.Lift):
            ks += self.extract_kernels(net.lifted_module)
        elif isinstance(net, torch.nn.Conv2d) or isinstance(
            net, torch.nn.ConvTranspose2d
        ):
            ks.append(net.weight.clone().detach().cpu())
        return ks

    def extract_time_constants(self, m, fn):
        l = []
        if isinstance(m, list):
            for i in m:
                l += self.extract_time_constants(i, fn)
            return l

        children = list(m.children())
        if len(children) > 0:
            l += self.extract_time_constants(children, fn)

        if isinstance(m, TemporalRF) and hasattr(m, "tau_mem_inv"):
            l.extend([fn(m.tau_mem_inv)])
            return l  # Return early

        return l

    def normalized_to_image(self, coordinate):
        return ((coordinate + 1) * self.resolution.to(self.args.device)) * 0.5

    def image_to_normalized(self, coordinate):
        return ((coordinate * 2) / self.resolution.to(self.args.device)) - 1

    def extract_batch(self, batch):
        x_warmup, x, y_co = batch
        y_co = y_co.permute(1, 0, 2, 3)  # TBCP
        y_co_norm = self.image_to_normalized(y_co)
        return x_warmup.float(), x.float(), y_co, y_co_norm

    def calc_coordinate(self, activations, state=None):
        rectified = self.rectification(activations.flatten(3)).reshape(
            activations.shape
        )
        coordinate = self.coordinate(rectified, state)
        return rectified, coordinate

    def extract_lif_states(self, state):
        if isinstance(state, norse.LIFBoxFeedForwardState):
            return state.v.mean()
        elif isinstance(state, list) or isinstance(state, tuple):
            return torch.stack([self.extract_lif_states(x) for x in state]).mean()
        else:
            return torch.as_tensor(0, device=self.args.device, dtype=torch.float32)

    def extract_spatio_temporal_parameters(self, module):
        spatial = []
        temporal = []
        for l in module.children():
            if (
                isinstance(l, norse.LIBoxCell)
                or isinstance(l, norse.LIFBoxCell)
                or isinstance(l, norse.TemporalReceptiveField)
            ):
                temporal.extend(l.parameters())
            elif isinstance(l, torch.nn.Sequential) or isinstance(
                l, norse.SequentialState
            ):
                s, t = self.extract_spatio_temporal_parameters(l)
                spatial.extend(s)
                temporal.extend(t)
            else:
                spatial.extend(l.parameters())
        return spatial, temporal

    def forward(self, warmup, x, prepend_warmup: bool = False):
        # Warmup
        with torch.no_grad():
            out_warmup, s, _ = self.net(warmup)
            _, (co_warmup, co_s) = self.calc_coordinate(out_warmup)

        # Predict
        out, _, activity = self.net(x, s)
        if self.args.net == "lif":
            snn_reg = torch.stack(activity)
        else:
            snn_reg = torch.tensor([0.0])
        out, (out_co, _) = self.calc_coordinate(out, co_s)  # Replace out w/ rectified
        if prepend_warmup:
            return torch.cat((out_warmup, out_co)), torch.cat((co_warmup, out_co))
        return out, out_co, snn_reg

    def calc_loss(self, out, out_co, y_co, snn_reg):
        # Norm
        loss_co = torch.norm(out_co - y_co, p=2, dim=-1)

        # Regularization
        y_gauss = make_gauss(
            y_co,
            self.out_shape.to(self.device),
            0.06,
            normalize=True,
        )
        loss_reg = (
            self.regularization(out.squeeze(), y_gauss.squeeze())
            * self.args.regularization_scale
        )

        # Spike regularization
        spike_reg = (
            (self.args.regularization_activity_mean - out).mean() ** 2
            + (self.args.regularization_activity_mean - snn_reg).mean() ** 2
        ) * self.args.regularization_activity_scale

        return loss_co, loss_reg, spike_reg, y_gauss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        warmup, x, y_co, y_co_norm = self.extract_batch(batch)
        warmup = warmup.permute(1, 0, 2, 3, 4)
        x = x.permute(1, 0, 2, 3, 4)
        out, out_co, snn_reg = self.forward(warmup, x)
        loss_co, loss_reg, spike_reg, y_gauss = self.calc_loss(
            out, out_co, y_co_norm, snn_reg
        )
        loss = loss_co.mean() + loss_reg.mean() + spike_reg.mean()

        # Visualize every 1000 steps
        if self.global_step % 1000 == 0:
            # Log prediction
            self.show_prediction(
                x[-1, 0, 0],
                y_co[-1, 0, 0],
                out[-1, 0, 0],
                self.normalized_to_image(out_co[-1, 0, 0]),
                y_gauss[-1, 0, 0],
            )

        self.log("train/loss", loss.mean(), sync_dist=True)
        self.log("train/norm", loss_co.mean(), sync_dist=True)
        self.log("train/reg", loss_reg.mean(), sync_dist=True)
        self.log("train/spike_reg", spike_reg.mean(), sync_dist=True)
        for i, layer in enumerate(snn_reg):
            self.log(f"train/out/{i}", layer.mean(), sync_dist=True)
        self.log(f"train/out/{len(snn_reg)}", out.mean(), sync_dist=True)
        if self.lr_schedulers() is not None:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)

        dic = {"loss": loss.mean(), "norm": loss_co.mean(), "reg": loss_reg.mean()}
        return dic

    def validation_step(self, batch, batch_idx):
        warmup, x, y_co, y_co_norm = self.extract_batch(batch)
        warmup = warmup.permute(1, 0, 2, 3, 4)
        x = x.permute(1, 0, 2, 3, 4)
        out, out_co, snn_reg = self.forward(warmup, x)
        loss_co, loss_reg, spike_reg, y_gauss = self.calc_loss(
            out, out_co, y_co_norm, snn_reg
        )
        loss = loss_co.mean() + loss_reg.mean()
        dic = {
            "loss": loss.mean(),
            "hp_metric": loss.mean(),
            "norm": loss_co.mean(),
            "reg": loss_reg.mean(),
            "spike_reg": spike_reg.mean(),
        }

        self.log("val/loss", loss.mean(), sync_dist=True)
        args = self.args
        with open(
            f"{args.name}_{pathlib.Path(args.data_root).name}.csv",
            "a",
        ) as fp:
            fp.write(
                f"{self.global_step},{loss.mean():.10f},{self.trainer.logger.version}\n"
            )
        self.log("hp_metric", loss.mean(), sync_dist=True)
        self.log("val/norm", loss_co.mean(), sync_dist=True)
        self.log("val/reg", loss_reg.mean(), sync_dist=True)
        self.log("val/spike_reg", spike_reg.mean(), sync_dist=True)
        return dic

    def configure_optimizers(self):
        params = list(
            chain(
                self.net.parameters(),
                self.coordinate.parameters(),
                self.rectification.parameters(),
            )
        )
        if self.optimizer == "adagrad":
            optims = [
                torch.optim.Adagrad(
                    params, lr=self.lr, weight_decay=self.args.weight_decay
                )
            ]
        elif self.optimizer == "adam":
            optims = [
                torch.optim.Adam(
                    params, lr=self.lr, weight_decay=self.args.weight_decay
                )
            ]
        elif self.optimizer == "sgd":
            optims = [
                torch.optim.SGD(params, lr=self.lr, weight_decay=self.args.weight_decay)
            ]
        elif self.optimizer == "spacetime":
            spatial, temporal = self.extract_spatio_temporal_parameters(self.net)
            optims = [
                torch.optim.Adam(
                    spatial, lr=self.lr, weight_decay=self.args.weight_decay
                ),
                torch.optim.Adam(
                    temporal,
                    lr=self.lr * self.args.lr_temporal_factor,
                    weight_decay=self.args.weight_decay,
                ),
            ]
        else:
            optims = [torch.optim.RMSprop(params, lr=self.lr, weight_decay=1e-5)]
        if self.lr_step == "step":
            steppers = [
                torch.optim.lr_scheduler.ExponentialLR(x, self.args.lr_decay)
                for x in optims
            ]
        elif self.lr_step == "ca":
            steppers = [
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    x, T_0=50, T_mult=2
                )
                for x in optims
                for x in optims
            ]
        elif self.lr_step == "none":
            steppers = []
        else:
            raise ValueError("Unknown stepper")
        return optims, steppers

    def on_after_backward(self) -> None:
        if self.global_step % 1000 > 0:  # Only log every 1000 steps
            return
        # Extract time constant gradients
        tc_and_gradients = self.extract_time_constants(
            self.net,
            lambda m: (
                m.clone().detach().cpu(),
                m.grad.clone().detach().cpu(),
            ),
        )
        for i, (t, g) in enumerate(tc_and_gradients):
            self.log(f"hist/grad/{i}", g, self.global_step)
            self.log(f"hist/tau/{i}", t, self.global_step)

        # Extract kernel gradients
        kernel_gradients = self.extract_kernels(
            self.net, lambda x: x.grad.clone().cpu()
        )
        for i, l in enumerate(kernel_gradients):
            self.logger.experiment.add_histogram(
                f"hist/conv{i}/grad", l, self.global_step
            )

    def show_prediction(self, x, x_co, y_im, y_co_pred, y_expected):
        im = render_prediction(
            x.detach().cpu(),
            x_co.detach().cpu(),
            y_im.detach().cpu().squeeze(),
            y_co_pred.detach().cpu(),
            y_expected.detach().cpu(),
        )
        try:
            self.logger.experiment.add_image(
                f"image/prediction", im, self.global_step, dataformats="HWC"
            )
            ks = self.extract_kernels(self.net)
            for i, k in enumerate(ks):
                kernel_image = render_kernels(k)
                self.logger.experiment.add_image(
                    f"image/kernels/{i}", kernel_image, self.global_step
                )

        except Exception as e:
            print(f"Failure to log step {self.global_step}", e)


def train(config, args, callbacks=[]):
    args_dict = {**vars(args), **config}  # Overwrite args values in case of tuning
    args = SimpleNamespace(**args_dict)

    args.resolution = torch.tensor([300, 300])

    train_data = torch.utils.data.DataLoader(
        ShapeDataset(
            args.data_root,
            t=args.timesteps,
            pose_delay=args.network_delay,
            stack=args.stack_frames,
            sum_frames=args.sum_frames,
            train=True,
            file_filter=args.dataset_filter,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
        prefetch_factor=2,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    val_data = torch.utils.data.DataLoader(
        ShapeDataset(
            args.data_root,
            t=args.timesteps,
            pose_delay=args.network_delay,
            stack=args.stack_frames,
            sum_frames=args.sum_frames,
            train=False,
            file_filter=args.dataset_filter,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        prefetch_factor=2,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    args.sum_suffix = "S" if args.sum_frames else ""
    args.name = f"{args.net}_{args.init_scheme}_{args.stack_frames}{args.sum_suffix}"
    logger = pl.loggers.TensorBoardLogger(
        args.log_root, name=f"{args.name}_({pathlib.Path(args.data_root).name})"
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
    )
    device = trainer.strategy.root_device
    setattr(args, "device", device)
    model = ShapesModel(args)
    trainer.fit(model, train_data, val_data)


def main(args):
    torch.set_float32_matmul_precision("medium")
    checkpoint_save = pl.callbacks.ModelCheckpoint(save_top_k=5, monitor="val/norm")
    train({}, args, [checkpoint_save])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "data_root",
        type=str,
        help="Location of the dataset to use for training and testing",
    )
    parser.add_argument("log_root", type=str, help="Root directory for logging")
    parser.add_argument("--timesteps", type=int, default=40)
    parser.add_argument("--network_delay", type=int, default=1)
    parser.add_argument("--stack_frames", type=int, default=1)
    parser.add_argument("--sum_frames", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dataset_filter", type=str, default=None)
    parser = ShapesModel.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
