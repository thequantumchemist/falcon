from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='falcon_md',
    version='1.3.0',
    description='The python distribution for the FALCON on-the-fly Machine Learning ab initio Molecular Dynamics code',
    long_description = long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/thequantumchemist/falcon',
    author='Noah Felis and  Wilke Dononelli',
    author_email='wido@uni-bremen.de',
    license='GPL-3.0',
    packages=['falcon_md','falcon_md/utils', 'falcon_md/models', 'falcon_md/structures', 'falcon_md/utils/analysis'],
    package_data={
        'falcon_md/structures': ['*.xyz', '*.traj'],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=['agox',
                      'numpy',
                      'ase',
                      'pytest'
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',

        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
