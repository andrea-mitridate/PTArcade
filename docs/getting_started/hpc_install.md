## Prerequisites
`conda` likes to install its own version of MPI, which frequently causes issues on HPCs. To get around that, first, load your HPC's MPI module. Save the path to the `mpicc` executable in an environment variable.
```
export MPICC=$(which mpicc)
```
## Install the Code
Now you can `pip` install as shown [here](../local_install/#with-pip)

## singularity Container
singularity is widely supported on HPC systems. Assuming you have a singularity module available to you, you can follow the instructions [here](../local_install/#with-singularity)
