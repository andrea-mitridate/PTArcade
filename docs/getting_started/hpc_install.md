## Prerequisites
`conda` likes to install its own version of MPI, which frequently causes issues on HPCs. To get around that, first load your HPC's MPI module. Save the path to the `mpicc` executable in an environment variable.
```
export MPICC=$(which mpicc)
```
## Install the code
Now you can `pip` install as show [here](../local_install/#with-pip)

## Singularity container
Singularity is widely supported on HPC systems. Assuming you have a Singularity module available to you, you can follow the instructions [here](../local_install/#with-singularity)
