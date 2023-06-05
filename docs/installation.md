# Installation 
The dependencies required to run PTArcade can be installed following either one of the following procedures

## conda installation
1) Install `(mini)conda`, an environment management system for python, from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html])
2) Exectue 
```
conda env create -f environment.yml
conda activate non-bhb-search
```

## pip installation

## singularity environment
A singularity environment with all the necessary dependencies already installed can be downloaded by typing 
```
singularity pull oras://ghcr.io/andrea-mitridate/non-bhb-search:latest
```

