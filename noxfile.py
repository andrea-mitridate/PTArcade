#!/usr/bin/env python3
from nox_poetry import session

@session(python=["3.10"])
def ruff(session):
    session.install('ruff')
    session.run('ruff', 'check', '.')

@session(python=["3.10"])
def black(session):
    session.install('ruff')
    session.run('black', '.')
