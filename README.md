# ptycho-workstation
This repository provides an open-source implementation of a batched stochastic gradient descent algorithm for 2D ptychographic reconstruction. The code is part of a custom MATLAB pipeline designed to efficiently process large 4D-STEM/ptychographic datasets using GPU-accelerated computation.

## Tested Environment / System Requirements

This software has been validated to run successfully in the following configuration.
Other operating systems, GPUs, drivers, or MATLAB versions have not been tested and may require adjustments.
- OS: CentOS Linux 8
- MATLAB: R2023b
- GPU: NVIDIA L40
- Driver: 535.161.08
- CUDA: 12.2

Although the algorithm is platform-independent in principle, reproducibility outside this tested environment is not guaranteed.

## Required MATLAB Toolboxes

The following MATLAB toolboxes must be installed:
- Optimization Toolbox
- Parallel Computing Toolbox
- Image Processing Toolbox

The GPU implementation also requires a compatible CUDA-enabled device.

## Compile CUDA Kernels Before Running

The CUDA kernels must be compiled before running any reconstruction scripts.
Execute the following MATLAB script:

```./+myCUDA/compileAllCu.m```

This step generates all MEX/CUDA binaries required for GPU-accelerated forward and adjoint operators.

## Configuration Example & How to Run

An example configuration script is provided:

```cfg_ptychoRecon_example.m```

To perform a reconstruction:
- Modify the parameters inside the config file according to your dataset.
- Simply run the config script, and the reconstruction pipeline will start automatically.

If you encounter GPU memory limitations, you can reduce memory usage by adjusting either:
- the number of CBED patterns processed per batch (nSpotsParallel), or
- the number of sub-batches used within each batch before updating the object function (nSubSpotsParallelM).

## Code References

This project partially draws inspiration from the following codebases.
We sincerely acknowledge their contributions:
- fold_slice: https://github.com/yijiang1/fold_slice
- py4DSTEM: https://github.com/py4dstem/py4DSTEM
