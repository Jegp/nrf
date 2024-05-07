import functools

import matplotlib
import matplotlib.pyplot as plt
import torch
import norse

size = 40

def gen_rf(derivatives):
    rff = functools.partial(norse.torch.functional.receptive_field.spatial_parameters, scales=torch.tensor([0.05]), 
        angles=torch.tensor([0, torch.pi / 4, torch.pi / 2, 3 * torch.pi / 4]),
        ratios=torch.tensor([1.4, 1.0]), include_replicas=False)
    rf = rff(derivatives=derivatives)
    fields = norse.torch.functional.receptive_field.spatial_receptive_fields_with_derivatives(rf, size=size)
    fields = fields / fields.max()
    fl = list(fields)[:5]
    fl.reverse()
    return torch.concat([fl[0], fl[-1], *fl[1:-1]], dim=1).squeeze().detach()
fields0 = gen_rf(derivatives=[(0, 0)])
fields1 = gen_rf(derivatives=[(0, 1)])
fields2 = gen_rf(derivatives=[(1, 0)])
fields = torch.concat([fields0, fields1, fields2], dim=0)


f, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)

n_fields = 5
angles = [0]
angles.extend(np.linspace(0, 3 * np.pi / 4, n_fields - 1))
ratios = [1] + [1.4] * n_fields
xs = [0.5 * size, 1.5 * size, 2.5 * size, 3.5 * size, 4.5 * size]
# Ylabel
ax.set_ylabel("Receptive field kernel")
ax.set_yticks(xs[:3], ["$g$", "$g_{x1}$", "$g_{x2}$"])
ax.set_ylim(3 * size, 0)
# Xlabel
xlabels = [f"$\\phi={a:.2f}$\n$\\xi={r:.2f}$" for a, r in zip(angles, ratios)]
ax.set_xlim(0, 5 * size)
ax.set_xticks(xs, xlabels, multialignment="center")
ax.set_xlabel(r"Rotation ($\phi$) and ratio ($\xi$)")

norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-fields.max(), vmax=fields.max())
cax = ax.imshow(fields, cmap="bwr", norm=norm)
cb = f.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="bwr"), ax=ax, shrink=0.7, use_gridspec=True)
cb.ax.yaxis.set_tick_params(pad=5)

# Grid
ax.plot([0, 204], [0, 0], c="lightgray", linewidth=0.3)
ax.plot([0, 204], [40, 40], c="lightgray", linewidth=0.3)
ax.plot([0, 204], [81, 81], c="lightgray", linewidth=0.3)
for x in range(40, 210, 41):
    ax.plot([x, x], [0, 122], c="lightgray", linewidth=0.3)

f.tight_layout()