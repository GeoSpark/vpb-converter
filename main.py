import sys
import glob

from vpb.vpb import *

if __name__ == '__main__':
    # file_names = glob.glob('*.vpb')
    # dump_unknowns(file_names)
    # convert_whatever_to_vpb(sys.argv[1], sys.argv[2], description='This is Lena (or Lenna), one of the most iconic '
    #                                                               'images used in image compression research.\rSee: '
    #                                                               'https://www.cs.cmu.edu/~chuck/lennapg/lenna.shtml')
    convert_vpb_to_whatever(sys.argv[1], sys.argv[2])
