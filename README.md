# Covariant spatio-temporal receptive fields for neuromorphic computing

This repository contains the code for the paper "Covariant spatio-temporal receptive fields for neuromorphic computing".

## Abstract
Biological nervous systems constitute important sources of inspiration towards computers that are faster, cheaper, and more energy efficient.
Neuromorphic disciplines view the brain as a coevolved system, simultaneously optimizing the hardware and the algorithms running on it.
There are clear efficiency gains when bringing the computations into a physical substrate, but we presently lack theories to guide efficient implementations.
Here, we present a principled computational model for neuromorphic systems in terms of spatio-temporal receptive fields, based on affine Gaussian kernels over space and leaky-integrator and leaky integrate-and-fire models over time.
Our theory is provably covariant to spatial affine and temporal scaling transformations, and with close similarities to the visual processing in mammalian brains.
We use these spatio-temporal receptive fields as a prior in an event-based vision task, and show that this improves the training of spiking networks, which otherwise is known as problematic for event-based vision. 
This work combines efforts within scale-space theory and computational neuroscience to identify theoretically well-founded ways to process spatio-temporal signals in neuromorphic systems.
Our contributions are immediately relevant for signal processing and event-based vision, and can be extended to other processing tasks over space and time, such as memory and control.

## Dataset
The data is generated using the [event-generator](https://github.com/ncskth/event-generator) repository.
We use the PyTorch dataset class in `dataset.py` to load the data.
Please refer to the event-generator repository for more information how to generate data to reproduce our results.

## Acknowledgements
The authors gratefully acknowledge support from the EC Horizon 2020 Framework
Programme under Grant Agreements 785907 and 945539 (HBP), the Swedish Research Council under contracts 2022-02969 and 2022-06725, and the Danish National Research Foundation grant number P1.
