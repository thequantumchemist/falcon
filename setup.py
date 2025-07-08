from setuptools import setup

setup(
    name='falcon-md',
    version='1.0.0',    
    description='The python distribution for the FALCON on-the-fly Machine Learning ab initio Molceular Dynamcis code',
    url='https://github.com/thequantumchemist/falcon',
    author='Noah Felis and  Wilke Dononelli',
    author_email='wido@uni-bremen.de',
    license='GPL-3.0',
    packages=['falcon','falcon/utils'],
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
