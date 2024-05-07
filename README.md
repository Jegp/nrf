# Covariant spatio-temporal receptive fields for neuromorphic computing

This repository contains the code for the paper "Covariant spatio-temporal receptive fields for neuromorphic computing".

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

We achieve all of the above by combining *spatial receptive fields* (**a**, **b**, and **d**) and *temporal receptive fields* (**c**, **e**) on both dense image data from the UCF-101 dataset (**b** and **c**) and event-based data from [our event-based dataset generator](https://github.com/ncskth/event-generator) (**d** and **e**).
The spatial receptive fields are Gaussian derivatives parameterizing certain spatial covariance properties.
The temporal receptive fields are truncated exponential kernels parameterizing temporal scaling properties.
Taken together, we provably achieve covariance for spatial affine transformations, Galilean transformations, and temporal scaling transformations in the image domain.
Put differently, we exploit the symmetries in natural image transformations to correctly code for physical movements of objects in space and time.

<video src="https://github.com/Jegp/nrf/assets/1064317/d86b6f79-281b-4642-84a6-146c864a100a" muted autoplay loop style="margin: 0 auto;"></video>

## Dataset
The data is generated using the [event-generator](https://github.com/ncskth/event-generator) repository (seen in the Figure above, panel **d**).
We use the PyTorch dataset class in `dataset.py` to load the data.
Please refer to the event-generator repository for more information how to generate data to reproduce our results.

## Acknowledgements
The authors gratefully acknowledge support from the EC Horizon 2020 Framework
Programme under Grant Agreements 785907 and 945539 (HBP), the Swedish Research Council under contracts 2022-02969 and 2022-06725, and the Danish National Research Foundation grant number P1.

## Citation
Please cite our work as follows
```bibtex
@misc{pedersen2024covariant,
      title={Covariant spatio-temporal receptive fields for neuromorphic computing}, 
      author={Jens Egholm Pedersen and Jörg Conradt and Tony Lindeberg},
      year={2024},
      eprint={2405.00318},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```
