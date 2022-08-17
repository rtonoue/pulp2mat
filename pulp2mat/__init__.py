import os, glob

__copyright__ = "Copyright (C) 2018 Your Name"
__version__ = "1.0.0"
__license__ = "MIT"
__author__ = "Ryota Tonoue"
__author_email__ = "rtonoue625@gmail.com"
__url__ = "https://github.com/rtonoue/pulp2mat"
__all__ = [
    os.path.split(os.path.splitext(file)[0])[1]
    for file in glob.glob(os.path.join(os.path.dirname(__file__), "[a-zA-Z0-9]*.py"))
]

from pulp2mat.lp2mat import *
