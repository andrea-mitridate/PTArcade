## `mpi4py` Fails to Install
You most likely don't have an MPI implementation installed. Choose one that works on your system, probably **OpenMPI** or **MPICH**.

## `libstempo` Fails to Install
If you are `pip` installing, check that you have installed `tempo2` as instructed [here](../local_install/#with-pip).

## `scikit-sparse` Fails to Install
If you are `pip` installing, check that you have installed `suitesparse` as instructed [here](../local_install/#with-pip).

## `RuntimeError` involving autocorrelation time
If you get an error like 
```python
RuntimeError: The autocorrelation time is too long relative to the variance in dimension 632846881.
```
a temporary solution is to increase [`N_samples`](../inputs/config.md#+config.N_samples).
We are actively working to upgrade our sampler and fix this issue.

