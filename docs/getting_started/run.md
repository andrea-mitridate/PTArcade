## pip and conda 
If you have install PTArcade with  `pip` or `conda`, open a terminal and 
run PTArcade with 
``` sh
ptarcade -m ./model.py 
```
Where the argument passed to the `-m` input flag is the path to a 
[model file]. In addition to a [model file], the following 
optional arguments can be passed to `ptarcade`:

* A *configuration file* can be passed via the input flag `-c`.
The configuration file allow ... More details on the model and
configuration files can be found in the [inputs] section. 

* A name for the ouptut chain can be specified via the input flag
`-n`. Explicitly naming the output chain can be useful if ...


## using a docker container
Just prepend `ptarcade` with `docker run`
``` sh
docker run ptarcade -m model.py 
```

## using a singularity container
Just prepend `ptarcade` with `singularity run`
``` sh
singularity run ptarcade -m model.py -c config.py 
```
  
  [model file]: ../inputs/model.md
  [inputs]: ../inputs/.md