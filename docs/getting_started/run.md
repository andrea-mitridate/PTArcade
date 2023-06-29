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
The commands of the previous section must be slightly modified to run within a Docker container.
Docker does not mount any directories into the container by default. 
You must pass directories to mount inside the container using the syntax `-v <source>:<destination>`. 
In the example below, we assume that the only directories you will pass to the command line options of PTArcade
are accessible from your current working directory.
``` sh
docker run -v $(pwd):$(pwd) -w $(pwd) -i -t ptarcade -m ./model.py
```

* `-v` tells Docker what to mount from the host computer and where to mount it in the container. Here, we mount the current working directory of the host into the container using its full path.
* `-w` sets the working directory of the container. In this case, it sets it to the current working directory that was just mounted.
* `-i -t` keeps `STDIN` open and allocates a pseudo-TTY 

The PYArcade in the `docker run` command refers to the name of the Docker image.
If you would like to run something else inside the container, then replace the PTArcade options with the program to run. For example, to run an interactive Bash shell

```sh
docker run -v $(pwd):$(pwd) -w $(pwd) -i -t ptarcade bash
```

## Using a singularity Container
As with Docker, the commands to run PTArcade must be slightly modified to run using Singularity.
However, the commands are much simpler because Singularity will automatically mount your home directory inside the container. Using the `ptarcade.sif` file you created during the singularity installation, type into a terminal
``` sh
singularity run ptarcade.sif -m ./model.py
```
You can also pass another command to run. For example, to start a Jupyter notebook type

```sh
singularity run ptarcade.sif jupyter notebook
```

If you want an interactive shell, run the following command

```sh
singularity shell ptarcade.sif
```

  
  [model file]: ../inputs/model.md
  [configuration file]: ../inputs/config.md
  [inputs]: ../inputs/index.md
  [out]: ../inputs/config.md#+config.out_dir
  [name]: ../inputs/model.md#+model.name
