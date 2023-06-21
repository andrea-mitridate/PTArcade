## pip and conda 
If you have installed PTArcade with  `pip` or `conda`, open a terminal and 
run PTArcade with 
``` sh
ptarcade -m ./model.py 
```
The argument passed to the `-m` input flag is the path to a 
[model file]. In addition to a [model file], the following 
optional arguments can be passed to `ptarcade`:

* A *[configuration file]* can be passed via the input flag `-c`.
The configuration file allows to control several parameters of 
the run, including the dataset to be analyzed, the number of MC
trials, etc. More details on the model and
configuration files can be found in the [inputs] section. 

* A string to append to the output folder. By default, the
chains will be saved in `./chains/np_model/chain_0`. Each of the 
three elements of this path can be controlled by the user. `./chains`
can be changed by using the [`out_dir`][out] parameter in the configuration 
file, `np_model` can be changed via the [`name`][name] parameter in the 
model file, and `chain_0` can be changed via the `-n` input flag passed 
to `ptarcade`. The passed argument will be appended to `chain_`, so, if you want 
to save the chains in a folder named `chain_42`, just pass the argument `-n 42`. 
This can be useful if you are running multiple chains for the same model and 
you want to save them in the same root folder. 


## Using a docker Container
Just prepend `ptarcade` with `docker run`
``` sh
docker run ptarcade -m model.py 
```

## Using a singularity Container
Just prepend `ptarcade` with `singularity run`
``` sh
singularity run ptarcade -m model.py 
```
  
  [model file]: ../inputs/model.md
  [configuration file]: ../inputs/config.md
  [inputs]: ../inputs/index.md
  [out]: ../inputs/config.md#+config.out_dir
  [name]: ../inputs/model.md#+model.name
