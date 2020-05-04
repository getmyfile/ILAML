import os
import re

import setuptools

directory = os.path.dirname(os.path.abspath(__file__))

# Extract version information
path = os.path.join(directory, 'ilaml', '__init__.py')
with open(path) as read_file:
    text = read_file.read()
pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.MULTILINE)
version = pattern.search(text).group(1)

# Extract long_description
path = os.path.join(directory, 'README.md')
with open(path) as read_file:
    long_description = read_file.read()


#extract requirements
requirementPath = os.path.join(directory, 'requirements.txt')
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name='goggles',
    version=version,
    url='https://github.com/getmyfile/ILAML',
    description='',
    long_description_content_type='text/markdow11n',
    long_description=long_description,
    license='',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires = install_requires,
    keywords='',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],

    project_urls={  # Optional
        'Homepage': 'https://github.com/getmyfile/ILAML',
        'Source': 'https://github.com/getmyfile/ILAML',
        'Bug Reports': 'https://github.com/getmyfile/ILAML/issues',
    },
)
