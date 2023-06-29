FROM python:3.10-bookworm
LABEL maintainer="David C. Wright <david.wright@nanograv.org>"
RUN apt update && apt install -y --no-install-recommends libsuitesparse-dev gfortran openmpi-bin libopenmpi-dev
RUN curl -sSL https://raw.githubusercontent.com/vallis/libstempo/master/install_tempo2.sh | sh -s /usr/local


ENV TEMPO2=/usr/local/share/tempo2
RUN pip install ptarcade jupyterlab notebook

RUN ldconfig

COPY ./docker-entrypoint.sh /docker-entrypoint.sh 
RUN chmod +x /docker-entrypoint.sh 
ENTRYPOINT ["/docker-entrypoint.sh"]
