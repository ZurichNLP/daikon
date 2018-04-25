#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='daikon',
      version='0.1',
      description='A simple encoder-decoder machine translation model',
      url='https://github.com/bricksdont/daikon',
      author='Samuel Läubli, Mathias Müller',
      author_email='laeubli@cl.uzh.ch, mmueller@cl.uzh.ch',
      license='LGPL',
      packages=['daikon'],
      scripts=['bin/daikon'],
      install_requires=[
        'numpy',
        'tensorflow-gpu'
    ])
