import os

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            return line.split("'")[1]

    raise RuntimeError('Unable to find version string.')


with open('requirements.txt', 'r') as requirements:
    setup(name='llm_attacks',
          version=get_version('llm_attacks/__init__.py'),
          install_requires=list(requirements.read().splitlines()),
          packages=find_packages(),
          description='library for creating adversarial prompts for language models',
          python_requires='>=3.10',
          author='Zijun Wang, Jieru Mei, Haoqin Tu, Siwei Yang, Bingchen Zhao',
          author_email='zijun_wang@zju.edu.cn',
          classifiers=[
              'Programming Language :: Python :: 3',
              'Operating System :: OS Independent'
          ],
          long_description=long_description,
          long_description_content_type='text/markdown')