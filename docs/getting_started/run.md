## pip and conda 
If you have install PTArcade with  `pip` or `conda`, open a terminal and 
run PTArcade with 
``` sh
ptarcade -m model.py -c config.py # (1)!
```

1. The optional flag `-n` can be used to specify the chain number

## using a docker container
Just prepend `ptarcade` with `docker run`
``` sh
docker run ptarcade -m model.py -c config.py
```
## using a singularity container
Just prepend `ptarcade` with `singularity run`
``` sh
singularity run ptarcade -m model.py -c config.py 
```
