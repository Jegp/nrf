import io
import math
from re import A

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch, torchvision
import torchvision.transforms.functional as F


def events_to_frames(frames, polarity: bool = False):
    if len(frames.shape) == 3:
        frames = frames.unsqueeze(-1).repeat(1, 1, 1, 3)
    else:
        if not polarity:
            frames = frames.abs().sum(-1)
        elif polarity:
            frames = torch.concat(
                [
                    frames,
                    torch.zeros(
                        frames.shape[0], 1, *frames.shape[2:], device=frames.device
                    ),
                ],
                dim=1,
            ).movedim(1, -1)
    frames = ((frames / frames.max()) * 255).int().clip(0, 255)
    return frames


def standardize_kernels(ks):
    max_size = max([k.shape[-1] for k in ks])
    resize = torchvision.transforms.Resize((max_size, max_size))
    return [resize(k) for k in ks]


def rearrange_kernels(kernels):
    out = []
    for ks in kernels:
        columns, rows, width, height = ks.shape

        t = torch.zeros(rows * height + rows - 1, columns * width + columns - 1)
        for column in range(columns):
            for row in range(rows):
                k = ks[column, row]
                t[
                    row * height + row : row * height + height + row,
                    column * width + column : column * width + width + column,
                ] = k
        out.append(t if rows < columns else t.permute(1, 0))
    return out


def kernel_color_norm(k):
    vmin = min(k.min(), -1e-5)
    vmax = max(k.max(), 1e-5)
    return matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)


def render_kernels(k):
    norm = kernel_color_norm(k)
    bwr = matplotlib.colormaps["bwr"]
    colors = torch.from_numpy(bwr(norm(k.flatten(0, 1)))).permute(0, 3, 1, 2)
    nrows = round(math.sqrt(len(colors)) * 1.618)
    return torchvision.utils.make_grid(colors, nrow=nrows)


def render_prediction(x, x_co, y_im, y_co_pred, y_expected):
    fig = plt.figure(figsize=(10, 5), dpi=200)
    plt.set_cmap("coolwarm")

    outer = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    b, c, d = [plt.Subplot(fig, outer[i]) for i in range(3)]
    for ax in (b, c, d):
        ax.axis("off")
        fig.add_subplot(ax)

    # Actual
    b.imshow(x.squeeze().T, cmap="gray")
    b.set_title(f"{x.sum():.0E} events ({(x.sum() * 100 / x.numel()):.0f}%)")

    # Prediction
    c.imshow(y_im.squeeze().T, cmap="gray")
    c.set_title(f"Pred. {y_co_pred[0]:.2f}x{y_co_pred[1]:.2f}")

    # Expectation
    d.imshow(y_expected.squeeze().T, cmap="gray")
    d.set_title(f"Exp. {x_co[0]:.2f}x{x_co[1]:.2f}")

    # Render
    plt.tight_layout()
    with io.BytesIO() as buf:
        fig.savefig(buf, format="raw")
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = arr.reshape((int(h), int(w), -1))
    plt.close(fig)
    return im


def render_prediction_video(frames, actual, predicted):
    inputs = [x.squeeze().detach().cpu() for x in [frames, actual, predicted]]
    video = []
    for frame, ys, xs in zip(*inputs):
        fig = plt.figure(figsize=(8, 6))
        plt.gca().axis("off")
        plt.imshow(frame.T, cmap="binary")
        # plt.imshow(frame.T, cmap="binary", extent=(0, 300, 0, 300))
        # plt.imshow(frame.T, cmap="binary", interpolation=None, extent=(0, frames.shape[-2], 0, frames.shape[-1]), aspect="auto")
        if len(ys.shape) == 1:
            ys = [ys]
        if len(xs.shape) == 1:
            xs = [xs]

        for a, p in zip(ys, xs):
            plt.plot(
                *a, marker=r"$\bigcirc$", markersize=32, markeredgewidth=5, c="#42c3f7"
            )
            plt.plot(*p, marker="x", markersize=30, markeredgewidth=5, c="red")

        plt.tight_layout()
        with io.BytesIO() as buf:
            fig.savefig(buf, format="raw")
            buf.seek(0)
            arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = arr.reshape((int(h), int(w), -1))[:, :, :3]
        video.append(im)
        plt.close()
    return torch.from_numpy(np.array(video))


def render_video(frames, filename, lossless=True, **kwargs):
    new_frames = frames.clone().cpu()
    if len(frames.shape) == 3:
        new_frames = new_frames.unsqueeze(-1).repeat(1, 1, 1, 3)
        new_frames = 255 * new_frames / new_frames.max()
    if len(frames.shape) < 3:
        raise ValueError("Dims must be >= 3")
    if lossless:
        kwargs["video_codec"] = "libx264"
        kwargs["options"] = {"crf": "1"}
    torchvision.io.write_video(filename, new_frames, **kwargs)


def animate_frames(frames, figure=None, interval: int = 20, **kwargs):
    from IPython import display

    if figure is None:
        figure, _ = plt.subplots(**kwargs)
    ax = figure.gca()

    image = ax.imshow(frames[0])  # .T)
    ax.set_axis_off()

    def animate(index):
        image.set_data(frames[index])  # .T)
        return image

    anim = FuncAnimation(figure, animate, frames=len(frames), interval=interval)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.tight_layout()
    plt.close()
