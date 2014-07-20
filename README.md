Chartbeats's: Predictive Web Analytics Challenge
================================================

Source code for the ECML/PKDD 2014 Discovery Challenge. Here we present a simple ideia which
was first place in two of the three tasks of the ECML/PKDD 2014 discovery challenge. See the
website below for details on the challenge:

https://sites.google.com/site/predictivechallenge2014/home


Dependencies
------------

    * Numpy >= 1.8.1 - http://scipy.org
    * Pandas >= 0.13.1 - http://pandas.pydata.org
    * Scikit-Learn >= 0.14.1 - http://scikit-learn.org
    * Scipy >= 0.14.1 - http://scipy.org

As an optional package (for some of the ideias exploited in the paper but not used on the
challenge), you will required pyksc:

    * pyksc - http://github.com/flaviovdf/pyksc (head branch)

For faster training times we used the packages available with Anaconda 
(http://http://www.continuum.io). Anaconda make's use of MKL for
multi-threaded matrix operations on the packages above. However, the code is 
pretty fast and should not take more than a few minutes even without MKL bindings.


How to parse the data
---------------------

Edit the `config.py` file with the approriate locations of the `train/` and `test/` folder
(provided by the competition organizers).
