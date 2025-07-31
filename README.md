# Covariant spatio-temporal receptive fields for spiking neural networks

This repository contains the code for the paper "Covariant spatio-temporal receptive fields for spiking neural networks".

A preprint is available at: [arXiv:2405.00318](https://arxiv.org/abs/2405.00318)

Further explanations can be found at [jegp.github.io/nrf](https://jegp.github.io/nrf)

## Abstract
Biological nervous systems constitute important sources of inspiration towards computers that are faster, cheaper, and more energy efficient.
Neuromorphic disciplines view the brain as a coevolved system, simultaneously optimizing the hardware and the algorithms running on it.
There are clear efficiency gains when bringing the computations into a physical substrate, but we presently lack theories to guide efficient implementations.
Here, we present a principled computational model for neuromorphic systems in terms of spatio-temporal receptive fields, based on affine Gaussian kernels over space and leaky-integrator and leaky integrate-and-fire models over time.
Our theory is provably covariant to spatial affine and temporal scaling transformations, and with close similarities to the visual processing in mammalian brains.
We use these spatio-temporal receptive fields as a prior in an event-based vision task, and show that this improves the training of spiking networks, which otherwise is known as problematic for event-based vision.
This work combines efforts within scale-space theory and computational neuroscience to identify theoretically well-founded ways to process spatio-temporal signals in neuromorphic systems.
Our contributions are immediately relevant for signal processing and event-based vision, and can be extended to other processing tasks over space and time, such as memory and control.

## Spatio-temporal receptive fields
**Read more at [jegp.github.io/nrf](https://jegp.github.io/nrf)**

[Event camera data](https://en.wikipedia.org/wiki/Event_camera) is sparse and discrete, which means that any computational model is faced with several challenges: the signal must somehow be integrated over time to capture the **temporal** characteristics, the **spatial** structure of the events needs to be kept over both space *and* time, and the sparsity should be retained to exploit the low energy consumption of neuromorphic technologies.

<video src="https://github.com/Jegp/nrf/assets/1064317/d86b6f79-281b-4642-84a6-146c864a100a" muted autoplay loop style="margin: 0 auto;"></video>

We achieve all of the above by combining *spatial receptive fields* (**a**, **b**, and **d**) and *temporal receptive fields* (**c**, **e**) on both dense image data from the UCF-101 dataset (**b** and **c**) and event-based data from [our event-based dataset generator](https://github.com/ncskth/event-generator) (**d** and **e**).
The spatial receptive fields are Gaussian derivatives parameterizing certain spatial covariance properties.
The temporal receptive fields are truncated exponential kernels parameterizing temporal scaling properties.
Taken together, we provably achieve covariance for spatial affine transformations, Galilean transformations, and temporal scaling transformations in the image domain.
Put differently, we exploit the symmetries in natural image transformations to correctly code for physical movements of objects in space and time.

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
All simulations, neuron models, and the spatio-temporal receptive fields rely on [the Norse library](https://github.com/norse/norse).
The implementation of affine directional derivatives is based on the [affscsp module in the pyscsp package](https://github.com/tonylindeberg/pyscsp) and parts of the temporal smoothing operations are based on the [pytempscsp package](https://github.com/tonylindeberg/pytempscsp).

The authors gratefully acknowledge support from the EC Horizon 2020 Framework
Programme under Grant Agreements 785907 and 945539 (HBP), the Swedish Research Council under contracts 2022-02969 and 2022-06725, and the Danish National Research Foundation grant number P1.

## Citation
Please cite our work as follows
```bibtex
@misc{pedersen2024covariant,
      title={Covariant spatio-temporal receptive fields for spiking neural networks},
      author={Jens Egholm Pedersen and Jörg Conradt and Tony Lindeberg},
      year={2024},
      eprint={2405.00318},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```
