## `mpi4py` Fails to Install
You most likely don't have an MPI implementation installed. Choose one that works on your system, probably **OpenMPI** or **MPICH**.

## `libstempo` Fails to Install
If you are `pip` installing, check that you have installed `tempo2` as instructed [here](../local_install/#with-pip).
If the errors involve `cython`, see [here](#cython-errors-when-installing-libstempo)

## `scikit-sparse` Fails to Install
If you are `pip` installing, check that you have installed `suitesparse` as instructed [here](../local_install/#with-pip).

## `RuntimeError` involving autocorrelation time
If you get an error like 
```python
RuntimeError: The autocorrelation time is too long relative to the variance in dimension 632846881.
```
a temporary solution is to increase [`N_samples`](../inputs/config.md#+config.N_samples).
We are actively working to upgrade our sampler and fix this issue.

## `cython` errors when installing `libstempo`
If you are installing `ptarcade` with `pip`, you may run into an installation error involving `libstempo`.
The recent `cython` 3.0 update included breaking changes that have not yet been accounted for in all dependencies. `libstempo`
is one such dependency. The error presented should look something like this:
```
Collecting libstempo>=2.4.0 (from enterprise-extensions<3.0.0,>=2.4.2->ptarcade==0.1.5)
  Downloading libstempo-2.4.5.tar.gz (885 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 885.3/885.3 kB 53.5 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [217 lines of output]
      
      Error compiling Cython file:
      ------------------------------------------------------------
      ...
      cdef class tempopulsar:
          """tempopulsar(parfile, timfile=None, warnings=False, fixprefiterrors=True,
                         dofit=False, maxobs=None, units=False, ephem=None, t2cmethod=None,
                         toas=None, toaerrs=None, observatory=None, obsfreq=1400)"""
      
          cpdef public object parfile
                ^
```
There is an open pull request for `libstempo` to fix this issue, but it has not been merged yet. We've tested this 
pull request, and it fixes the `cython` issues while remaining backwards compatible with old `cython` versions.

The temporary fix for this problem is to first install `libstempo` from the merged pull request as follows:
```sh
pip install git+https://github.com/vallis/libstempo@refs/pull/54/merge
```
Now that the patched version of `libstempo` is installed, you can install `ptarcade` as usual.

## Can't install on Mac
Are you on an Apple silicon Mac? See instructions [here](../local_install#on-apple-silicon).

## My problem is not described here
If you don't see a solution to your problem on this page, please open an issue on our [GitHub](https://github.com/andrea-mitridate/PTArcade/issues) repository or email us at [ptarcade.dev@gmail.com](mailto:ptarcade.dev@gmail.com).
