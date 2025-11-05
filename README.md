# Covariant spatio-temporal receptive fields for spiking neural networks

[![Nature Communications Paper](https://zenodo.org/badge/DOI/10.1038/s41467-025-63493-0.svg)](https://doi.org/10.1038/s41467-025-63493-0)
![GitHub Repo stars](https://img.shields.io/github/stars/jegp/nrf)

This repository contains the code for the paper "Covariant spatio-temporal receptive fields for spiking neural networks".

## Video summary (watch on YouTube)
[![Paper summary video](https://img.youtube.com/vi/qtHkrx4tYfI/0.jpg)](https://www.youtube.com/watch?v=qtHkrx4tYfI)


## Introduction
[Neuromorphic computing](https://en.wikipedia.org/wiki/Neuromorphic) exploits the laws of physics to perform computations, similar to the human brain.
If we can "lower" the computation into physics, we achieve **extreme energy gains**, [up to 27-35 orders of magnitude](https://ieeexplore.ieee.org/document/10363573).
So, why aren't we doing that? Presently, we *lack theories to guide efficient implementations*.
We can build the circuits, but we don't know how to combine them to achieve what we want.
[Current neuromorphic models cannot compete with deep learning](https://www.nature.com/articles/s43588-021-00184-y).


Here, **we provide a principled computational model for neuromorphic systems** based on tried-and-tested spatio-temporal covariance properties.
We demonstrate superior performance for simple neuromorphic primitives in an [event-based vision](https://en.wikipedia.org/wiki/Event_camera) task compared to naïve artificial neural networks.
The direction is exciting to us because

1. We define mathematically coherent covariance properties, which are required to correctly handle signals in space and time,
2. Use neuromorphic primitives (leaky integrator and leaky integrate-and-fire models) to outcompete a non-neuromorphic neural network, and
3. Our results have immediate relevance in signal processing and event-based vision, with the possibility to extend to other tasks over space and time, such as memory and control.

## Abstract
Biological nervous systems constitute important sources of inspiration towards computers that are faster, cheaper, and more energy efficient.
Neuromorphic disciplines view the brain as a coevolved system, simultaneously optimizing the hardware and the algorithms running on it.
There are clear efficiency gains when bringing the computations into a physical substrate, but we presently lack theories to guide efficient implementations.
Here, we present a principled computational model for neuromorphic systems in terms of spatio-temporal receptive fields, based on affine Gaussian kernels over space and leaky-integrator and leaky integrate-and-fire models over time.
Our theory is provably covariant to spatial affine and temporal scaling transformations, and with close similarities to the visual processing in mammalian brains.
We use these spatio-temporal receptive fields as a prior in an event-based vision task, and show that this improves the training of spiking networks, which otherwise is known as problematic for event-based vision.
This work combines efforts within scale-space theory and computational neuroscience to identify theoretically well-founded ways to process spatio-temporal signals in neuromorphic systems.
Our contributions are immediately relevant for signal processing and event-based vision, and can be extended to other processing tasks over space and time, such as memory and control.

**Read more at [jepedersen.dk](https://jepedersen.dk/posts/202510_nrf/)**

## Dataset
The data is generated using the [Geometric Event-Response Data Generation](https://github.com/ncskth/gerd) repository (seen in the movie above, panel **d**).
The exact data used in this publication can be recreated using the script `data_generation.sh` or retrieved by reaching out to the author of the paper below.
We use the PyTorch dataset class in `dataset.py` to load the data.
Additional information about the data generation method is available in the [GERD repository](https://github.com/ncskth/gerd) and [the preprint](https://arxiv.org/abs/2412.03259).

## Usage

Before our code can be run, the following dependencies must be installed:
```
torch>=2.2
norse>=1.1.0
pytorch-lightning==1.9.4
tensorboard==2.17.1
```

The main entry-point is the `learn_shapes.py` file, which takes two main arguments (path to the training data and path to the log directory) and a host of hyperparameters.

```
python3 learn_shapes.py <path-to-data> <path-to-log>
```

A description of the parameters can be found using `python3 learn_shapes.py --help`.

### Reproducing the results in the paper

The paper uses specific sets of parameters that are available in the `config_ann.txt` and `config_snn.txt` files. The data in the paper was procured from the [Swedish Alvis cluster](https://www.c3se.chalmers.se/about/Alvis/).
The cluster-specific code is available upon request, but generally, the code can be run with the following script:

```bash
# Set basic flags common for all models
DATADIR=<path-to-dataset>
LOGDIR=<path-to-logs>
FLAGS="${DATADIR} ${LOGDIR} --n_spatial_scales=4 --n_temporal_scales=4 --max_epochs=50 --devices=1 --accelerator=gpu --strategy=dp"

# Use the flags in the config files to set the following variables:
MODEL_FLAGS="$FLAGS --net=$net --coordinate=$coordinate --stack_frames=$stack_frames --init_scheme_spatial=$init_scheme_spatial \
 --init_scheme_temporal=$init_scheme_temporal --batch_size=$batch_size --weight_sharing=$weight_sharing --dropout=$dropout \
 --n_angles=$n_angles --n_ratios=$n_ratios --batch_normalization=$batch_normalization"

# Run the model
python3 learn_shapes.py $MODEL_FLAGS
```

## Acknowledgements
All simulations, neuron models, and the spatio-temporal receptive fields rely on [the Norse library](https://github.com/norse/norse).
The implementation of affine directional derivatives is based on the [affscsp module in the pyscsp package](https://github.com/tonylindeberg/pyscsp) and parts of the temporal smoothing operations are based on the [pytempscsp package](https://github.com/tonylindeberg/pytempscsp).

The authors gratefully acknowledge support from the EC Horizon 2020 Framework
Programme under Grant Agreements 785907 and 945539 (HBP), the Swedish Research Council under contracts 2022-02969 and 2022-06725, and the Danish National Research Foundation grant number P1.

## Citation
Our work is freely available at [https://www.nature.com/articles/s41467-025-63493-0](https://www.nature.com/articles/s41467-025-63493-0) and can be cited as follows
```bibtex
@misc{pedersen2025covariant,
    title={Covariant spatio-temporal receptive fields for spiking neural networks}, 
    author={Pedersen, J. E. and Conradt, J. and Lindeberg, T.},
    year={2025},
    pages={8231},
    volume={16},
    DOI={10.1038/s41467-025-63493-0},
    number={1},
    journal={Nature Communications},
}
```

