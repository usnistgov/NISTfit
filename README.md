# NISTfit

A heavily abstracted C++-based kernel for fitting correlations

[![Build Status](https://travis-ci.org/usnistgov/NISTfit.svg?branch=master)](https://travis-ci.org/usnistgov/NISTfit)

## Why NISTfit?

  - In our work, we develop correlations between input properties and output properties.  These correlations are sometimes complex multiparameter equations of state, and other times, simple polynomials.  In any case, it was desired to have a generalized framework for carrying out these types of procedures in a generalized, fast(!), and flexible framework.  Being open-source and cross-platform, C++ is an ideal language for all three goals.

## Information

  - Please direct any questions to Ian Bell (ian.bell@nist.gov)

## Usage

    python wrapper can be built and installed with:

    ```
    python setup.py install
    ``` 
    
    Requirements:
    * Cmake
    * C++11 compliant compiler (MSVC 2015+ on windows, most recent versions of g++ or clang will work fine)
    * python (anaconda package is one good option, also includes plotting libraries (matplotlib) needed to run the timing tests)
    
    **Notes**: to specify that you want the Intel compiler, (on linux) you can do:

    ```
    CXX=icc python setup.py install
    ```
    
    Alternatively, you can checkout and install in one fell swoop with:
    ```
    pip install git+git://github.com/usnistgov/NISTfit.git
    ```
  
## License

  - Public Domain.  See LICENSE.txt

## Credits

  - Much of the parallel code is adapted from the StackOverflow response of Andy Prowl (http://stackoverflow.com/a/15257055/1360263)
  - Thanks also to Matthias Kunick