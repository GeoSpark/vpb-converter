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

import binascii
from datetime import datetime
from os import path

import construct as cs
import numpy as np
from PIL import Image

supported_image_types = ['.vpb']

header_record_file_metadata = cs.Struct(
    'record_count' / cs.Int8ub,  # Possibly.
    'record_size' / cs.Int8ub,  # Size of structure after this field.
    # Always seems to be \x00\x00
    'unknown1' / cs.Bytes(2),
    'day' / cs.Int16ub,
    'month' / cs.Int16ub,
    # Years since 1980.
    'year' / cs.Int16ub,
    'hours' / cs.Int16ub,
    'minutes' / cs.Int16ub,
    'seconds' / cs.Int16ub,
    # Always seems to be \x00\x00. Maybe milliseconds?
    'unknown2' / cs.Bytes(2)
)

# Without extension.
header_record_filename = cs.Struct(
    'file_name' / cs.PascalString(cs.Int16ub, encoding='ascii')
)

# Always seems to be "Picture".
header_record_filetype = cs.Struct(
    'file_type' / cs.PascalString(cs.Int16ub, encoding='ascii')
)

# Optional. This is the Quantel user who created the file.
header_record_owner = cs.Struct(
    'owner' / cs.PascalString(cs.Int16ub, encoding='ascii')
)

# Optional. Space separated list.
header_record_category = cs.Struct(
    'category' / cs.PascalString(cs.Int16ub, encoding='ascii')
)

# Optional. This is placed after the header, but always at 1024 bytes from the top of the file.
header_description = cs.Struct(
    'description' / cs.PascalString(cs.Int16ub, encoding='ascii')
)

# Some of these unknowns are going to be bitfields or enums for image format and so on.
# Have found 41984 extra image bytes in welcome.vpb.  831488 + 41984 = 873472
header_record_image_metadata = cs.Struct(
    'record_count' / cs.Int8ub,  # Possibly.
    'record_size' / cs.Int8ub,  # Size of structure after this field.
    'bits_per_pixel' / cs.Int16ub,  # Possibly.
    # Always seems to be \x00\x00
    'unknown1' / cs.Bytes(2),
    'width' / cs.Int16ub,
    'height' / cs.Int16ub,
    # Always PAL: \x08\x37 NTSC: \x08\x55
    'unknown2' / cs.Bytes(2),
    # Always PAL: \x08\xFC NTSC: \x07\x94
    'unknown3' / cs.Bytes(2),
    'description_ptr' / cs.Int32ub,  # Possibly.
    # 'description' / cs.Pointer(cs.this.description_ptr, cs.PascalString(cs.Int16ub, encoding='ascii')),
    'image_size_bytes' / cs.Int32ub,
    'unknown4' / cs.Bytes(lambda ctx: ctx.record_size - 20)
)

header_record_unknown2 = cs.Struct(
    'record_count' / cs.Int8ub,  # Possibly.
    'record_size' / cs.Int8ub,  # Size of structure after this field.
    # \x00\x0d or \x00\x0c
    'unknown1' / cs.Bytes(2),
    # \x50\x00 or \xac\x00
    'unknown2' / cs.Bytes(2)
)

header_record_unknown1 = cs.Struct(
    'record_count' / cs.Int8ub,  # Possibly.
    'record_size' / cs.Int8ub,  # Size of structure after this field.
    # Always seems to be \x00\x00\x00\x00
    'unknown1' / cs.Bytes(4)
)

header = cs.Struct(
    'type' / cs.Enum(cs.Int8ub,
                     filename=0x02,
                     file_type=0x03,
                     unknown2=0x04,
                     owner=0x0A,
                     category=0x0B,
                     unknown1=0x0D,
                     image_metadata=0x10,
                     file_metadata=0x98,
                     end=0xFF),

    'record' / cs.Switch(cs.this.type, {'file_metadata': header_record_file_metadata,
                                        'filename': header_record_filename,
                                        'file_type': header_record_filetype,
                                        'owner': header_record_owner,
                                        'category': header_record_category,
                                        'unknown1': header_record_unknown1,
                                        'unknown2': header_record_unknown2,
                                        'end': cs.Pass,
                                        'image_metadata': header_record_image_metadata}),
)


# Using the ITU-R BT.601 standard. See: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion and
# http://www.equasys.de/colorconversion.html
# noinspection PyPep8Naming
def RGB_to_YCbCr(RGB):
    mat = np.array([[0.257, -0.148, 0.439], [0.504, -0.291, -0.368], [0.098, 0.439, -0.071]])
    val = [16.0, 128.0, 128.0] + np.dot(RGB, mat)

    return np.clip(np.fix(val), 0.0, 255.0).astype('uint8')


# noinspection PyPep8Naming
def YCbCr_to_RGB(YCbCr):
    mat = np.array([[1.164, 1.164, 1.164], [0.0, -0.392, 2.017], [1.596, -0.813, 0.0]])
    val = (YCbCr - [16.0, 128.0, 128.0])

    val = np.dot(val, mat)

    return np.clip(np.fix(val), 0.0, 255.0).astype('uint8')


def dump_unknowns(file_names):
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            h = f.read(2048)
            records = cs.RepeatUntil(lambda obj, lst, ctx: obj.type == 'end', header).parse(h)
            print('{0: <24}: {1}'.format(file_name,
                                         [binascii.hexlify(x) for x in records.search_all('unknown') if x is not None]))


def convert_vpb_to_whatever(vpb_file_name, output_file_name, verbose=False):
    """
    Uses PIL to output the image, the format is automatically chosen by the file name extension.
    """
    buf, records = _parse_header(vpb_file_name)

    if verbose:
        print(records)

    # Image is Y'CbCr 4:2:2
    image_data = np.frombuffer(buf, dtype='uint8').reshape((records.search('height'), records.search('width') * 2))

    Y = (np.copy(image_data[:, 1::2]))
    Cb = np.repeat(np.copy(image_data[:, ::4]), 2, axis=1)
    Cr = np.repeat(np.copy(image_data[:, 2::4]), 2, axis=1)
    image = np.dstack((Y, Cb, Cr))
    bar = YCbCr_to_RGB(image)

    output = Image.fromarray(bar)

    output.save(output_file_name)


def _parse_header(vpb_file_name):
    with open(vpb_file_name, 'rb') as f:
        h = f.read(2048)
        records = cs.RepeatUntil(lambda obj, lst, ctx: obj.type == 'end', header).parse(h)
        buf = f.read(records.search('image_size_bytes'))
    return buf, records


def convert_whatever_to_vpb(input_file_name, vpb_file_name, verbose=False, date_hack=False, owner='DEFAULT',
                            description='', categories=None):
    input_image = Image.open(input_file_name)

    if input_image.width != 720 or not (input_image.height == 576 or input_image.height == 488):
        # TODO: Scale and fit the incoming image to a TV size.
        raise ValueError

    # Convert to YCbCr colour space, ignoring alpha.
    image_channels = input_image.split()
    R = np.frombuffer(image_channels[0].tobytes(), dtype='uint8').reshape((input_image.height, input_image.width))
    G = np.frombuffer(image_channels[1].tobytes(), dtype='uint8').reshape((input_image.height, input_image.width))
    B = np.frombuffer(image_channels[2].tobytes(), dtype='uint8').reshape((input_image.height, input_image.width))
    YCbCr = RGB_to_YCbCr(np.dstack([R, G, B]))

    # Compress the Cb and Cr channels horizontally by doing a simple averaging of pixel pairs. More complex
    # convolutions can be achieved by changing the kernel. May not work properly on odd widths, but that shouldn't be
    # much of an issue in this case.
    kernel = [0.5, 0.5]
    Cb = np.apply_along_axis(lambda i: np.convolve(i, kernel, 'valid'), axis=1, arr=YCbCr[:, :, 1])[:, ::2]
    Cr = np.apply_along_axis(lambda i: np.convolve(i, kernel, 'valid'), axis=1, arr=YCbCr[:, :, 2])[:, ::2]

    # Interleave the three channels as CbYCrYCbYCr...
    YCbCr422 = np.empty((input_image.height, input_image.width * 2), dtype='uint8')
    YCbCr422[:, 0::4] = Cb
    YCbCr422[:, 1::2] = YCbCr[:, :, 0]
    YCbCr422[:, 2::4] = Cr

    with open(vpb_file_name, 'wb') as f:
        # Build the image header.
        f.write(b'\x03')
        f.write(header_record_filetype.build(dict(file_type=u'Picture')))

        dt = datetime.now()
        if date_hack:
            year = 2005
        else:
            year = dt.year

        f.write(b'\x98')
        f.write(header_record_file_metadata.build(dict(record_count=1, record_size=16, unknown1=0,
                                                       day=dt.day, month=dt.month, year=year - 1980,
                                                       hours=dt.hour, minutes=dt.minute, seconds=dt.second,
                                                       unknown2=0)))

        f.write(b'\x02')
        f.write(header_record_filename.build(dict(file_name=path.splitext(path.basename(vpb_file_name))[0])))

        if categories is not None:
            f.write(b'\x0b')
            f.write(header_record_category.build(dict(category=categories)))

        if owner is not None:
            f.write(b'\x0a')
            f.write(header_record_owner.build(dict(owner=owner)))

        f.write(b'\x10')
        # These seem consistent with PAL-sized images.
        uk2_unknown1 = b'\x00\x0d'
        uk2_unknown2 = b'\x50\x00'

        im_unknown2 = b'\x08\x37'
        im_unknown3 = b'\x08\xfc'

        if input_image.height == 488:
            # These seem consistent with NTSC-sized images.
            uk2_unknown1 = b'\x00\x0c'
            uk2_unknown2 = b'\xac\x00'

            im_unknown2 = b'\x08\x55'
            im_unknown3 = b'\x07\x94'

        f.write(header_record_image_metadata.build(dict(record_count=1, record_size=20,
                                                        bits_per_pixel=16,
                                                        unknown1=b'\x00\x00',
                                                        width=input_image.width, height=input_image.height,
                                                        unknown2=im_unknown2, unknown3=im_unknown3,
                                                        description_ptr=1024,
                                                        # description=description,
                                                        image_size_bytes=input_image.width * input_image.height * 2,
                                                        unknown4=b'')))

        f.write(b'\x04')
        f.write(header_record_unknown2.build(dict(record_count=1, record_size=4, unknown1=uk2_unknown1,
                                                  unknown2=uk2_unknown2)))

        f.write(b'\x0d')
        f.write(header_record_unknown1.build(dict(record_count=1, record_size=4, unknown1=b'\x00\x00\x00\x00')))

        # Pad the header.
        f.write(b'\xff' * (1024 - f.tell()))

        # Write the description.
        f.write(cs.Padded(1024, header_description, b'\xe5').build(dict(description=description)))

        # Write the image bytes.
        f.write(YCbCr422.tobytes())

    _, records = _parse_header(vpb_file_name)

    if verbose:
        print(records)
