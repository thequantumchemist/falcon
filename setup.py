from setuptools import setup

setup(
    name='falcon_md',
    version='1.0.0',
    description='The python distribution for the FALCON on-the-fly Machine Learning ab initio Molceular Dynamics code',
    url='https://github.com/thequantumchemist/falcon',
    author='Noah Felis and  Wilke Dononelli',
    author_email='wido@uni-bremen.de',
    license='GPL-3.0',
    packages=['falcon_md','falcon_md/utils', 'falcon_md/models', 'falcon_md/structures', 'falcon_md/utils/analysis'],
    package_data={
        'falcon_md/structures': ['*.xyz', '*.traj'],
    },
    include_package_data=True,
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
