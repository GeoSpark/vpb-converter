# Copyright 2018 John Donovan <john@geospark.co.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Paintbox Converter v0.1

Input and output can be any image type supported by Python PIL(Pillow) or a
Quantel Picture.
Currently supports PAL images, and probably NTSC images too. Input images must
be the correct size (720 x 576 for PAL, 720 x 488 for NTSC). Scaling/cropping
may be added in future versions.
Only Quantel VPB Picture files are supported for now. Stencils and Cutouts may
be supported in future versions.
Comments and categories are lost during conversion, but in future versions
they may get added for image formats that support such features, e.g. PNG,
JPEG, TIFF.

Usage:
    pb_conv -i FILE -o FILE [--verbose] [--datehack]
    pb_conv (-h | --help)
    pb_conv --version

Options:
    -i --input FILE    Input image file name. Can be a Quantel file, or any
                   standard image file.
    -o --output FILE   Output image file name. Can be a Quantel file, or any
                   standard image file.
    --verbose      Prints metadata from the Quantel image file.
    --datehack     If set, treats the year as 2005 to get around the feature
                   on the Paintbox where it barfs with more recent years.
    -h --help      Show this screen.
    --version      Show version.
"""

import os

from vpb import vpb
from docopt import docopt

if __name__ == '__main__':
    # file_names = glob.glob('*.vpb')
    # dump_unknowns(file_names)
    # convert_whatever_to_vpb(sys.argv[1], sys.argv[2], description='This is Lena (or Lenna), one of the most iconic '
    #                                                               'images used in image compression research.\rSee: '
    #                                                               'https://www.cs.cmu.edu/~chuck/lennapg/lenna.shtml')

    arguments = docopt(__doc__, version='Paintbox Converter v0.1')

    input_extension = os.path.splitext(arguments['--input'])[1].lower()
    output_extension = os.path.splitext(arguments['--output'])[1].lower()

    if input_extension in vpb.supported_image_types and output_extension in vpb.supported_image_types:
        print('Both files are Quantel images, so there\'s nothing for me to do.')
    elif input_extension not in vpb.supported_image_types and output_extension not in vpb.supported_image_types:
        print('Neither file is a Quantel image, so there\'s nothing for me to do.')
    elif input_extension in vpb.supported_image_types:
        vpb.convert_vpb_to_whatever(arguments['--input'], arguments['--output'], arguments['--verbose'])
    elif output_extension in vpb.supported_image_types:
        vpb.convert_whatever_to_vpb(arguments['--input'], arguments['--output'], verbose=arguments['--verbose'],
                                    date_hack=arguments['--datehack'])
    else:
        print('I have no idea what\'s happening!')
