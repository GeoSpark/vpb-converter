Some investigatory code into Quantel's VPB image file format.

The images so far seem to be in YCbCr 4:2:2 format, interleaved as Cb Y Cr Y Cb Y Cr Y... with one byte per channel.
The colourspace used is probably the ITU-R BT.601 standard. See: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
and http://www.equasys.de/colorconversion.html

The header consists of a number of mandatory and optional records, and is in big endian format. The original software
was almost certainly written in Pascal because all strings are Pascal-style, with a 16-bit length at the start of the
string, rather than null-terminated or fixed length.

Strings are assumed to be ASCII encoded.

It's possible that there is RGB <-> YCbCr conversion parameters in the header to allow for different RGB colourspace
variants to be used. But this is speculation at this point.
