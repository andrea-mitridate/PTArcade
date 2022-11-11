Bootstrap: docker
From: ubuntu:22.04
# We build off of the base ubuntu image hosted on docker

# Here we specify files to be copied in to the container
%files
    environment.yml

# Run the following commands during the build
%post
    # Update Ubuntu
    apt-get update -y && apt-get install -y build-essential bc wget git unzip

    # Remove cache to reduce image size
    rm -rf /var/lib/apt/lists/*


    # Set vars for conda install
    CONDA_INSTALL_PATH="/opt/miniconda"
    CONDA_ENV_NAME="$(head -n 1 environment.yml | cut -f 2 -d ' ')"

    # Download miniconda3 and install
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $CONDA_INSTALL_PATH

    rm -r miniconda.sh

    # Run conda setup
    . $CONDA_INSTALL_PATH/etc/profile.d/conda.sh

    # Create the environment and activate.
    # It is not necessary to activate, but it is a good test because the build
    # will fail if the environment is not found.
    conda env create -f environment.yml
    conda activate $CONDA_ENV_NAME
    conda clean --all

    # Setup a .bashrc specifically for the container.
    # There are other ways to get conda working, but this is the most straightforward
    echo ". $CONDA_INSTALL_PATH/etc/profile.d/conda.sh" >> /.singularity_bashrc
    echo "conda activate $CONDA_ENV_NAME" >> /.singularity_bashrc

# Set these environment variables
%environment
    CONDA_BIN_PATH="/opt/miniconda/bin"

    # if we run `singularity shell`, use our new .singularity_bashrc
    action="${0##*/}"
    if [ "$action" == "shell" ]; then
        if [ "${SINGULARITY_SHELL:-}" == "/bin/bash" ]; then
            set -- --noprofile --init-file /.singularity_bashrc
        elif test -z "${SINGULARITY_SHELL:-}"; then
            export SINGULARITY_SHELL=/bin/bash
            set -- --noprofile --init-file /.singularity_bashrc
        fi
    fi
    export PATH="$CONDA_BIN_PATH:$PATH"

# Run this on `singularity run` . It uses the non-bhb-search conda env
%runscript
    exec /bin/bash -c "source /.singularity_bashrc && IFS=' ' && $*"

%labels
    Author = David Wright <davecwright3>
    Version = v0.0.1

%help
    This container houses all the runtime dependencies for the NANOGrav 15yr non-bhb search.
    Intended usage:
    $ singularity run non-bhb-search.sif python sampler.py -m (model info file).py -n (numeric info file).py -c (chain number)
