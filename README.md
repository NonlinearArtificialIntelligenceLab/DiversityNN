[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# DiversityNN
Code Repository for **Neural networks embrace diversity** paper

- [DiversityNN](#diversitynn)
  - [Usage](#usage)
    - [Metalearning](#metalearning)
    - [Validation training](#validation-training)
    - [Hessian Analysis](#hessian-analysis)
  - [Authors](#authors)
  - [Link to paper](#link-to-paper)
  - [Key Results](#key-results)
  - [Code References](#code-references)


## Usage

### Metalearning
```python Source/Driver.py --activation=<base activation to be modified> --n_activations=<number of activation functions to learn>```

    optional arguments:
    -h, --help            show this help message and exit
    --use_gpu             To GPU or not to GPU
    --gpu_no GPU_NO       Which GPU to use ?
    --data_path DATA_PATH
                            Path to data
    --output_folder OUTPUT_FOLDER
                            Path to output folder
    --inner_batch_size INNER_BATCH_SIZE
                            Batch size for classifier
    --outer_batch_size OUTER_BATCH_SIZE
                            Batch size for activation function
    --epochs EPOCHS       Number of epochs in inner loop
    --steps STEPS         Number of steps in outer loop
    --inner_lr INNER_LR   Learning rate for classifier
    --outer_lr OUTER_LR   Learning rate for activation function
    --inner_mu INNER_MU   SGD momentum for classifier
    --outer_mu OUTER_MU   RMSprop momentum for activation function
    --inner_hidden_size INNER_HIDDEN_SIZE
                            Hidden Layer size for classifier
    --outer_hidden_size OUTER_HIDDEN_SIZE
                            Hidden Layer size for activation function
    --verbose             Verbose output
    --seed SEED           fixed seed for reproducibility
    --runs RUNS           No. of runs to repeat the experiment
    --activation ACTIVATION
                            Select base activation function to be modified. Choose from elu, sin, tanh
    --n_activations N_ACTIVATIONS
                            No. of activation functions to be used in classifier

### Validation training
```python Source/Driver.py --act_path_=<path to learned activation text file> --validation_mode=<activation function(s) to use>```

    optional arguments:
    -h, --help            show this help message and exit
    --use_gpu             To GPU or not to GPU
    --gpu_no GPU_NO       Which GPU to use ?
    --data_path DATA_PATH
                            Path to data
    --verbose             Verbose output
    --seed SEED           fixed seed for reproducibility
    --runs RUNS           No. of runs to repeat the experiment
    --inner_batch_size INNER_BATCH_SIZE
                            Batch size for classifier
    --epochs EPOCHS       Number of epochs

    --dataset DATASET     Select dataset from MNIST1D, MNIST, CIFAR
    --validation_mode VALIDATION_MODE
                            select validation mode as activation number in file or false for all

    --act_path ACT_PATH   Give path to activation function for validation
    
    --save_curve          saves model training curve
    --save_model          saves model state dict

### Hessian Analysis
```python Hessian_analysis/Hessian.py ---model_path=<path to pytorch model> --act_path<path to activation function file for interpolant activation>```

    optional arguments:
    -h, --help            show this help message and exit
    --model_path MODEL_PATH
                            path to model for collecting hessian
    --act_path ACT_PATH   path to activation array for interpolation

***
## Authors
Anshul Choudhary, Anil Radhakrishnan, John F. Lindner, Sudeshna Sinha, and William L. Ditto

## Link to paper
* [arXiv](https://arxiv.org/abs/2204.04348)

## Key Results
* We construct neural networks with learnable activation functions and sere that they quickly diversify from each other under training. 
* These activations subsequently outperform their _pure_ counterparts on classification tasks.
* The neuronal sub-networks instantiate the neurons and meta-learning adjusts their weights and biases to find efficient spanning sets of nonlinear activations.
* These improved neural networks provide quantitative examples of the emergence of diversity and insight into its advantages.

## Code References
* [MNIST1D](https://github.com/greydanus/mnist1d)
* [PyHessian](https://github.com/amirgholami/PyHessian)
* [Torch Interpolation](https://github.com/sbarratt/torch_interpolations)

