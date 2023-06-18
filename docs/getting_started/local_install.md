If you're familiar with Python, you
can install PTArcade with [`pip`][pip] or [`conda`][conda], the Python package manager.
If not, we recommend using a [`docker`][docker] or [`singularity`][singularity] virtual environment.

  [pip]: #with-pip
  [conda]: #with-conda
  [docker]: #with-docker
  [singularity]: #with-singularity

### with pip 
PTArcade is published as a [PyPI package](https://pypi.org/project/PTArcade/) and can be installed with
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


This will automatically install compatible versions of all **Python** dependencies,
as well as download the following PTA datasets: [NANOGrav 12.5-year][NG12], [NANOGrav 15-year][NG12], and [IPTA DR2][IPTA2].

!!! danger "Non-Python Dependencies"

    If you choose to install from PyPI, you'll need to get the Non-Python dependencies yourself.
    They are

    - `libstempo` needs [tempo2](https://github.com/vallis/libstempo#pip-install)
    - `sckit-sparse` needs [suitesparse](https://github.com/scikit-sparse/scikit-sparse#with-pip)

  [Python package]: https://pypi.org/project/PTArcade/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
  [NG12]: https://nanograv.org/science/data/125-year-pulsar-timing-array-data-release
  [NG15]: https://nanograv.org/science/data/125-year-pulsar-timing-array-data-release
  [IPTA2]: https://gitlab.com/IPTA/DR2/tree/master/release

### with conda 
PTArcade is also published as a [Conda package] that can be installed with
`conda`, ideally by using a [virtual environment]. Open up a terminal
 and install PTArcade with:
``` sh
conda install ptarcade # (1)!
```

1. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for conda installation info.

Conda will also install the non-Python dependencies.

  [Python package]: https://pypi.org/project/PTArcade/
  [virtual environment]: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
  [NG12]: https://nanograv.org/science/data/125-year-pulsar-timing-array-data-release
  [NG15]: https://nanograv.org/science/data/125-year-pulsar-timing-array-data-release
  [IPTA2]: https://gitlab.com/IPTA/DR2/tree/master/release

### with docker 
The official [Docker image] is a great way to get up and running in a few
minutes, as it comes with all dependencies pre-installed. Open up a terminal
and pull the image with:
```
    docker pull <path to remote> # TODO
```


### with singularity 
A singularity environment with all the necessary dependencies already installed can be downloaded by typing 
```
singularity pull oras://ghcr.io/andrea-mitridate/non-bhb-search:latest
```

  
  
  [semantic versioning]: https://semver.org/
  [upgrade to the next major version]: upgrade.md
  [Markdown]: https://python-markdown.github.io/
  [Pygments]: https://pygments.org/
  [Python Markdown Extensions]: https://facelessuser.github.io/pymdown-extensions/
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/
