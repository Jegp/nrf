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
        """
        Constructs a model that can learn to predict the position of N shapes in a 2-dimensional grid.

        Arguments:
          args (SimpleNamespace): A namespace containing parameters detailed in the arguments processed
                using PyTorchLightning.
        """
        super().__init__()
        self.args = args

        # Network
        derivatives = [(0, 0), (1, 0), (0, 1), (1, 1)]
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
            n_temporal_scales=args.n_temporal_scales,
            n_spatial_scales=args.n_spatial_scales,
            n_angles=args.n_angles,
            n_angles_grow=args.n_angles_grow,
            n_ratios=args.n_ratios,
            derivatives=derivatives,
            separate_spatial_channels=args.separate_spatial_channels,
            weight_sharing=args.weight_sharing,
            activation=activation,
            init_scheme_spatial=args.init_scheme_spatial,
            init_scheme_temporal=args.init_scheme_temporal,
            channels_in=2 if args.sum_frames else (args.stack_frames * 2),
            resolution=args.resolution,
            n_classes=args.n_classes,
            channel_layers=3,
            device=args.device,
            dropout=args.dropout,
            skip_connections=args.skip_connections,
            batch_normalization=args.batch_normalization
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
        """ Adds model-specific arguments to the argument parser. """
        parser = parent_parser.add_argument_group("Network")
        parser.add_argument("--net", type=str)
        parser.add_argument("--n_spatial_scales", type=int, required=True)
        parser.add_argument("--n_temporal_scales", type=int, required=True)
        parser.add_argument("--n_angles", type=int, required=True)
        parser.add_argument("--n_ratios", type=int, required=True)
        parser.add_argument(
            "--n_angles_grow", type=int, default=0
        )  # Whether to increase the number of angles at high excentricities
        parser.add_argument(
            "--n_classes",
            type=int,
            default=3,
            help="Number of object classes to track. Defaults to 3",
        )
        parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in the final layer")
        # Whether to separate spatial channels along with the temporal (for a total of S*T channels)
        # or keep them inside the temporal channel (for a total of T channels)
        parser.add_argument("--separate_spatial_channels", type=bool, default=False)
        parser.add_argument("--weight_sharing", type=int, default=False)
        parser.add_argument("--skip_connections", type=int, default=False)
        parser.add_argument("--batch_normalization", type=int, default=False)
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
        parser.add_argument("--regularization_activity_mean", type=float, default=5e-2)
        parser.add_argument("--regularization_activity_scale", type=float, default=5e-2)
        parser.add_argument(
            "--coordinate",
            type=str,
            choices=["dsnt", "dsntli", "pixel"],
            default="dsntli",
            help="Method to reduce 2d surface to coordinate",
        )
        parser.add_argument(
            "--init_scheme_spatial",
            type=int_or_str,
            default="rf",
            help="Init method for spatial RFs",
        )
        parser.add_argument(
            "--init_scheme_temporal",
            type=int_or_str,
            default="rf",
            help="Init method for temporal RFs",
        )
        parser.add_argument("--lr", type=float, default=5e-3)
        parser.add_argument(
            "--lr_step", type=str, default="step", choices=["step", "ca", "none"]
        )
        parser.add_argument("--lr_decay", type=float, default=0.95)
        parser.add_argument(
            "--optimizer",
            type=str,
            choices=["adagrad", "adam", "rmsprop", "sgd"],
            default="adam",
        )
        parser.add_argument("--optimizer_split", type=int, default=0)
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-6,
        )
        return parent_parser

    def extract_kernels(self, net, f=lambda x: x.clone().detach().cpu()):
        """ Extracts the spatial kernels from all the channels in the model. """
        kernels = []
        for i, channel in enumerate(net.channels):
            for n, st in enumerate(channel.spatiotemporal):
                label = f"channel/{i}/{n}"
                kernels.append((f(st.spatial.spatial.weight), label))
        kernels.append((f(net.classifier[0].weight), f"classifier"))
        return kernels

    def extract_time_constants(self, net):
        """ Extracts the time constants from all the channels in the model, if any. """
        if self.args.net == "ann":
            return {}
        taus = {}
        for i, channel in enumerate(net.channels):
            for n, st in enumerate(channel.spatiotemporal):
                if n not in taus:
                    taus[n] = []
                taus[n].append(st.temporal.temporal[0].p.tau_mem_inv)
        return taus

    def normalized_to_image(self, coordinate):
        """ Converts the coordinates from normalized space to image space. """
        return ((coordinate + 1) * self.resolution.to(self.args.device)) * 0.5

    def image_to_normalized(self, coordinate):
        """ Converts the coordinates from image space to normalized space. """
        return ((coordinate * 2) / self.resolution.to(self.args.device)) - 1

    def extract_batch(self, batch):
        """ Extracts the batch from the dataloader with coordinates in both unnormalized 
        and normalized form (normalized to (0, 1)).
        
        Arguments:
            batch (tuple): The batch of data from the dataloader.

        Returns:
            x_warmup (torch.Tensor): The warmup input tensor.
            x (torch.Tensor): The input tensor.
            y_co (torch.Tensor): The coordinates tensor.
            y_co_norm (torch.Tensor): The normalized coordinates tensor.
        """
        x_warmup, x, y_co = batch
        y_co = y_co.permute(1, 0, 2, 3)  # TBCP
        y_co_norm = self.image_to_normalized(y_co)
        return x_warmup.float(), x.float(), y_co, y_co_norm

    def calc_coordinate(self, activations, state=None):
        """ Calculates the coordinates from the activations of the model.

        Arguments:
            activations (torch.Tensor): The output of the model.
            state (torch.Tensor): The state of the model, if available. Defaults to None.

        Returns:
            rectified (torch.Tensor): The rectified activations.
            coordinate (torch.Tensor): The coordinates calculated from the activations.
        """
        rectified = self.rectification(activations.flatten(3)).reshape(
            activations.shape
        )
        coordinate = self.coordinate(rectified, state)
        return rectified, coordinate

    def extract_spatio_temporal_parameters(self, module):
        """ Recursively extracts the spatio-temporal parameters from the model. """
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

    def show_prediction(self, x, x_co, y_im, y_co_pred, y_expected):
        """ Logs a visualization of the network prediction to Tensorboard.

        Arguments:
            x (torch.Tensor): The input tensor.
            x_co (torch.Tensor): The coordinates of the input tensor.
            y_im (torch.Tensor): The image tensor.
            y_co_pred (torch.Tensor): The predicted coordinates.
            y_expected (torch.Tensor): The expected coordinates.
        """
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
            ks_file = f"{self.args.log_root}/v{self.logger.version}_{self.args.name}_kernels_{self.global_step}.dat"
            torch.save(ks, ks_file)
            for k, label in ks:
                kernel_image = render_kernels(k)
                self.logger.experiment.add_image(
                    f"kernel/{label}", kernel_image, self.global_step
                )
            ts = self.extract_time_constants(self.net)
            ts_file = f"{self.args.log_root}/v{self.logger.version}_{self.args.name}_taus_{self.global_step}.dat"
            torch.save(ts, ts_file)
            for block, taus in ts.items():
                self.logger.experiment.add_histogram(
                    f"taus/block/{block}", torch.stack(taus), self.global_step
                )

        except Exception as e:
            print(f"Failure to log step {self.global_step}", e)

    def forward(self, warmup, x, prepend_warmup: bool = False):
        """ A single forward pass of the model.
        It computes the output of the model (as a 2-d matrix) and, from that, the coordinates of the model.
        
        Arguments:
            warmup (torch.Tensor): The input tensor for the warmup phase.
            x (torch.Tensor): The input tensor for the main forward pass.
            prepend_warmup (bool): Whether to prepend the warmup output to the final output.
                Defaults to False.
        
        Returns:
            out (torch.Tensor): The output of the model.
            out_co (torch.Tensor): The predicted coordinates.
            snn_reg (list): List of SNN activations for regularization.
        """

        # Warmup
        with torch.no_grad():
            out_warmup, s, _ = self.net(warmup)
            _, (co_warmup, co_s) = self.calc_coordinate(out_warmup)

        # Predict
        out, _, activity = self.net(x, s)
        if self.args.net == "lif":
            snn_reg = activity
        else:
            snn_reg = []
        out, (out_co, _) = self.calc_coordinate(out, co_s)  # Replace out w/ rectified
        if prepend_warmup:
            return torch.cat((out_warmup, out_co)), torch.cat((co_warmup, out_co))
        return out, out_co, snn_reg

    def calc_loss(self, out, out_co, y_co, snn_reg):
        """ Calculates the loss for the model based on the output and target coordinates.
        Arguments:
            out (torch.Tensor): The output of the model.
            out_co (torch.Tensor): The predicted coordinates.
            y_co (torch.Tensor): The target coordinates.
            snn_reg (list): List of SNN activations for regularization.

        Returns:
            loss_co (torch.Tensor): The loss based on the coordinates.
            loss_reg (torch.Tensor): The regularization loss.
            spike_reg (torch.Tensor): The spike regularization loss.
            y_gauss (torch.Tensor): The Gaussian representation of the target coordinates.
        """
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
            self.regularization(out.squeeze(), y_gauss.squeeze()) * 1e-4
        )

        # Spike regularization
        act_mean = self.args.regularization_activity_mean
        act_scale = self.args.regularization_activity_scale
        # spike_reg = [(act_mean - out).mean() ** 2 * act_scale]
        spike_reg = []
        for activation in snn_reg:
            spike_reg.append((act_mean - activation).mean() ** 2 * act_scale)
        if self.args.net == "lif" and len(spike_reg) > 0:
            spike_reg = torch.stack(spike_reg).sum()
        else:
            spike_reg = torch.tensor([], device=out.device)

        return loss_co, loss_reg, spike_reg, y_gauss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """ Performs a single training step, including the forward pass and loss calculation. """
        warmup, x, y_co, y_co_norm = self.extract_batch(batch)
        warmup = warmup.permute(1, 0, 2, 3, 4)
        x = x.permute(1, 0, 2, 3, 4)
        out, out_co, snn_reg = self.forward(warmup, x)
        loss_co, loss_reg, spike_reg, y_gauss = self.calc_loss(
            out, out_co, y_co_norm, snn_reg
        )
        loss = loss_co.mean() + loss_reg.mean()
        if spike_reg.numel() > 0:
            loss = loss + spike_reg.mean()

        # Visualize every 1000 steps
        if self.global_step % 1000 == 0:
            self.show_prediction(
                x[-1, 0, 0],
                y_co[-1, 0, 0],
                out[-1, 0, 0],
                self.normalized_to_image(out_co[-1, 0, 0]),
                y_gauss[-1, 0, 0],
            )

        self.log("train/loss", loss.mean())
        self.log("train/norm", loss_co.mean())
        self.log("train/reg", loss_reg.mean())
        self.log("train/spike_reg", spike_reg.mean())
        for i, layer in enumerate(snn_reg):
            self.log(f"train/out/{i}", layer.mean())
        self.log(f"train/out/{len(snn_reg)}", out.mean())
        if self.lr_schedulers() is not None:
            if isinstance(self.lr_schedulers(), list):
                for i, scheduler in enumerate(self.lr_schedulers()):
                    self.log(f"lr/{i}", scheduler.get_last_lr()[0])
            else:
                self.log("lr", self.lr_schedulers().get_last_lr()[0])

        dic = {"loss": loss.mean(), "norm": loss_co.mean(), "reg": loss_reg.mean()}
        return dic

    def validation_step(self, batch, batch_idx):
        """ Validation step for the model. It computes the loss and logs it. """
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

        self.log("val/loss", loss.mean())
        with open(
            f"{self.args.log_root}/{self.args.name}_{pathlib.Path(self.args.data_root).name}.csv",
            "a",
        ) as fp:
            fp.write(
                f"{self.global_step},{loss_co.mean():.10f},{self.trainer.logger.version}\n"
            )
        self.log("hp_metric", loss.mean(), sync_dist=True)
        self.log("val/norm", loss_co.mean(), sync_dist=True)
        self.log("val/reg", loss_reg.mean(), sync_dist=True)
        self.log("val/spike_reg", spike_reg.mean(), sync_dist=True)
        return dic

    def configure_optimizers(self):
        """ Configures the optimizers and learning rate schedulers (steppers) for the model and returns them.
        """
        parameter_list = []
        if self.args.optimizer_split:
            s, t = self.net.spatiotemporal_parameters()
            s.extend(self.coordinate.parameters())
            s.extend(self.rectification.parameters())
            parameter_list.append((s, float(self.args.lr)))
            parameter_list.append((t, float(self.args.lr) * 1e-10))
        else:
            params = []
            params.extend(self.net.parameters())
            params.extend(self.coordinate.parameters())
            params.extend(self.rectification.parameters())
            parameter_list.append((params, float(self.args.lr)))
        optims = []
        steppers = []
        for params, lr in parameter_list:
            if self.optimizer == "adagrad":
                optims.append(
                    torch.optim.Adagrad(
                        params, lr=lr, weight_decay=self.args.weight_decay
                    )
                )
            elif self.optimizer == "adam":
                optims.append(
                    torch.optim.Adam(
                        params, lr=lr, weight_decay=self.args.weight_decay
                    )
                )
            elif self.optimizer == "sgd":
                optims.append(
                    torch.optim.SGD(params, lr=lr, weight_decay=self.args.weight_decay)
                )
            else:
                optims.append([torch.optim.RMSprop(params, lr=lr, weight_decay=1e-5)])
        
        if self.lr_step == "step":
            steppers.extend([
                torch.optim.lr_scheduler.ExponentialLR(x, self.args.lr_decay)
                for x in optims
            ])
        elif self.lr_step == "ca":
            steppers.extend([
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    x, T_0=50, T_mult=2
                )
                for x in optims
                for x in optims
            ])
        elif self.lr_step == "none":
            steppers = []
        else:
            raise ValueError(f"Unknown stepper {self.lr_step}")
        return optims, steppers


def train(args, callbacks=[]):
    """
    Initializes the training process using arguments provided from the commandline.
    The callbacks are passed to the PyTorch Lightning trainer for logging and checkpointing.

    Arguments:
        args (argparse.Namespace): The arguments parsed from the commandline.
        callbacks (list): A list of PyTorch Lightning callbacks to be used during training.
            Defaults to an empty list.
    """
    args_dict = {**vars(args)}  # Overwrite args values in case of tuning
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
        num_workers=6,
        prefetch_factor=1,
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
        num_workers=6,
        prefetch_factor=1,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    args.sum_suffix = "S" if args.sum_frames else ""
    data_root_name = pathlib.Path(args.data_root).name
    args.name = f"{args.net}_{args.init_scheme_spatial}-{args.init_scheme_temporal}_{args.stack_frames}{args.sum_suffix}_s{args.weight_sharing:b}_d{args.dropout:.2f}_n{args.batch_normalization:b}_({data_root_name})"
    logger = pl.loggers.TensorBoardLogger(
        args.log_root, name=f"{args.name}"
    )
    logger.log_hyperparams(args.__dict__)

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
    train(args, [checkpoint_save])


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
