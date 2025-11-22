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

## Output Files

For each reconstruction, the program writes all results to the specified output directory:

```path_saveOutputs/```

The following files will be generated:

(1) Log File
- log.txt
Records all console output, iteration progress, parameter settings, and runtime information.

(2) Reconstructed Object Function

Saved every specified number of iterations (XXX denotes the iteration index):
- objAbsXXX.mrcs – amplitude of the reconstructed object
- objAngleXXX.mrcs – phase of the reconstructed object

(3) Reconstructed Probe Function

Also saved per iteration (XXX denotes the iteration index):
- probAbsXXX.mrcs – probe amplitude
- probAngleXXX.mrcs – probe phase

(4) Probe Initialization Aperture (Optional)

If aperture-based initialization is enabled, the following file will be saved:
- vP.mrc – the input aperture used to initialize the probe function

(5) Initial Values (Iteration 0)

The program also stores the starting point of the reconstruction:
- objAbs000.mrcs – initial object amplitude
- objAngle000.mrcs – initial object phase
- probAbs.mrcs – initial probe amplitude
- probAngle.mrcs – initial probe phase
- spot000.mat – distances of all scan positions from the FOV center (in meters)

## Code References

This project partially draws inspiration from the following codebases.
We sincerely acknowledge their contributions:
- fold_slice: https://github.com/yijiang1/fold_slice
- py4DSTEM: https://github.com/py4dstem/py4DSTEM
