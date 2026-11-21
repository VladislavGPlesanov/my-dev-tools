import sys
import glob
import tables as tb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse as ap

from MyPlotter import myPlotter

parser = ap.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default='', help="threshold file to plot", required=True)
parser.add_argument('-n', "--name", type=str, default='EBALA', help="suffix of the plots")
#parser.add_argument("--descr", nargs="+", type=str,default='', help="Description of the runs in option -d (exact same order)")
parser.add_argument("--compare", type=str, default=None, help='file to compare with')

args = parser.parse_args()

thrfile = args.file
picname = args.name
compfile = args.compare

#thrfile = sys.argv[1]
#picname = sys.argv[2]

matrix = np.loadtxt(thrfile, dtype=int)

MP = myPlotter()

MP.plotMatrix(
    matrix,
    f"TPX-Trimmers-{picname}",
    labels=["THS matrix", "x, [pix]", "y, [pix]"],
    figsize=(9,8),
    cmap='jet',
    cbarname='[DAC]',
    fDebug=False    
)

if(compfile is not None):

    comp_matrix = np.loadtxt(compfile, dtype=int)

    MP.plotMatrix(
        comp_matrix,
        f"TPX-Trimmers-compared-{picname}",
        labels=["THS matrix (to compare with)", "x, [pix]", "y, [pix]"],
        figsize=(9,8),
        cmap='jet',
        cbarname='[DAC]',
        fDebug=False    
    )

    diff_matrix = comp_matrix - matrix

    MP.plotMatrix(
        diff_matrix,
        f"TPX-DIfferenceMatrix-{picname}",
        labels=["THS matrix (Difference)", "x, [pix]", "y, [pix]"],
        figsize=(9,8),
        cmap='coolwarm',
        cbarname='[DAC]',
        fDebug=False    
    )



