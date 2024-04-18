If you're familiar with Python, you
can install PTArcade with [`pip`][pip] or [`conda`][conda], the Python package manager.
If not, we recommend using a [`docker`][docker] or [`singularity`][singularity] virtual environment.

### With conda <small>(recommended)</small> { #with-pip data-toc-label="with conda" }
PTArcade is now available on [conda-forge](https://anaconda.org/conda-forge/ptarcade)! you can install PTArcade using conda by typing in a terminal (1)
{ .annotate }

1. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for conda installation info.

``` sh
conda install -c conda-forge ptarcade # (1)!
```

1. If you want to install PTArcade in a new environment, run
```sh
conda create -n ptarcade -c conda-forge python=3.10 ptarcade
```

This will install PTArcade and all the required dependencies in a conda environment named `ptarcade`. PTArcade will download and cache the following PTA datasets at runtime:
[NANOGrav 12.5-year][NG12], [NANOGrav 15-year][NG12], and [IPTA DR2][IPTA2].

### With pip 
PTArcade is also published as a [PyPI package](https://pypi.org/project/PTArcade/) and can be installed with
`pip`, ideally by using a [virtual environment](https://docs.python.org/3/library/venv.html). Open up a terminal
 and install PTArcade with:
``` sh
pip install ptarcade # (1)!
```

1. We suggest to install PTArcade in a virtual environment. You can do
    so by running
    ```bash
    python3 -m venv <path/to/env>
    source <path/to/env>/bin/activate
    python3 -m pip install ptarcade
    ```

This will automatically install compatible versions of all **Python** dependencies and, as 
for the conda installation, download the following PTA datasets at runtime:
[NANOGrav 12.5-year][NG12], [NANOGrav 15-year][NG12], and [IPTA DR2][IPTA2].

!!! danger "Non-Python Dependencies"

    If you choose to install from PyPI, you'll need to get the non-Python dependencies yourself.

    - `libstempo` needs [tempo2](https://github.com/vallis/libstempo#pip-install). You can install
    it by typing in a terminal
    ```
    curl -sSL https://raw.githubusercontent.com/vallis/libstempo/master/install_tempo2.sh | sh
    ```
    - `sckit-sparse` needs [suitesparse](https://github.com/scikit-sparse/scikit-sparse#with-pip). 
    You can install it by typing in a terminal 

        === "Mac"
            ```
            brew install suite-sparse
            ```

        === "Debian"
            ```
            sudo apt-get install libsuitesparse-dev
            ```
    - `mpi4py` needs an MPI implementation. You can install it by typing
    in a terminal 

        === "Mac"
            ```
            brew install open-mpi
            ```

        === "Debian"
            ```
            sudo apt install libopenmpi-dev openmpi-bin
            ```


### With docker 
The official [Docker image][docker] is a great way to get up and running in a few
minutes, as it comes with all dependencies pre-installed. Open up a terminal
and pull the image with:
```sh
docker pull ngnewphy/ptarcade:latest
```


### With singularity 
A singularity environment with all the necessary dependencies already installed can be downloaded by typing 
```sh
singularity pull ptarcade.sif docker://ngnewphy/ptarcade:latest
```
This will create a Singularity image and save it as `ptarcade.sif` in the current working directory.

  [pip]: #with-pip
  [conda]: #with-conda
  [docker]: #with-docker
  [singularity]: #with-singularity
  [Python package]: https://pypi.org/project/PTArcade/
  [conda_env]: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
  [NG12]: https://nanograv.org/science/data/125-year-pulsar-timing-array-data-release
  [NG15]: https://nanograv.org/science/data/125-year-pulsar-timing-array-data-release
  [IPTA2]: https://gitlab.com/IPTA/DR2/tree/master/release
  [semantic versioning]: https://semver.org/
  [upgrade to the next major version]: upgrade.md
  [Markdown]: https://python-markdown.github.io/
  [Pygments]: https://pygments.org/
  [Python Markdown Extensions]: https://facelessuser.github.io/pymdown-extensions/
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/
  [env]: ../assets/downloads/ptarcade.yml
  [docker]: https://hub.docker.com/r/ngnewphy/ptarcade
