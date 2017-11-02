import sys
import glob
import binascii
from datetime import datetime
from os import path

import construct as cs
import numpy as np
from PIL import Image


header_record_file_metadata = cs.Struct(
    # Seems a bit incongruous because no other record has this field, and this is a fixed-length record, too.
    # It is always set at 16, in which case it's the length of this record not including this field.
    'record_size' / cs.Int8ub,
    # Always seems to be \x00\x00
    'fm_unknown' / cs.Bytes(2),
    'day' / cs.Int16ub,
    'month' / cs.Int16ub,
    # Years since 1980.
    'year' / cs.Int16ub,
    'hours' / cs.Int16ub,
    'minutes' / cs.Int16ub,
    'seconds' / cs.Int16ub,
    # Always seems to be \x00\x00. Maybe milliseconds?
    'fm_unknown2' / cs.Bytes(2)
)

header_record_filename = cs.Struct(
    # Without extension.
    'file_name' / cs.PascalString(cs.Int16ub, encoding='ascii')
)

header_record_filetype = cs.Struct(
    # Maybe this is originating software or system. Always seems to be "Picture".
    'file_type' / cs.PascalString(cs.Int16ub, encoding='ascii'),
    # Always seems to be \x98 Version?
    'ft_unknown' / cs.Int8ub
)

header_record_user = cs.Struct(
    # I think this is the Quantel user who created the file. The only optional record type I have found so far. Files
    # that do have this record always have it set to "DEFAULT".
    'user' / cs.PascalString(cs.Int16ub, encoding='ascii')
)

# Some of these unknowns are going to be bitfields or enums for image format and so on. There may even be record types
# that I've not split up. There are a lot of 0x04s in this record, and that is the next number in the lower sequence
# of record types.
header_record_image_metadata = cs.Struct(
    # Always seems to be \x01\x14\x00\x10\x00\x00
    'im_unknown1' / cs.Bytes(6),
    'width' / cs.Int16ub,
    'height' / cs.Int16ub,
    # For 720*576 PAL images, always seems to be \x08\x37 (2013). For NTSC 720*488 \x08\x55 (2133)
    'im_unknown2' / cs.Bytes(2),
    # For 720*576 PAL images, always seems to be \x08\xfc (2300). For NTSC 720*488 \x07\x94 (1940)
    'im_unknown3' / cs.Bytes(2),
    # Always seems to be \x00\x00\x04\x00 (1024) Possibly file offset or size of the large block of \xe5 bytes.
    'im_unknown4' / cs.Bytes(4),
    'image_size_bytes' / cs.Int32ub,
    # Always seems to be \x04\x01\x04\x00
    'im_unknown5' / cs.Bytes(4),
    # \x0c\xac for (3244) PAL,
    # \x0a\xc0 (2752) for NTSC
    'im_unknown6' / cs.Bytes(2),
    # Always seems to be \x00\x0d\x01\x04\x00\x00\x00\x00
    'im_unknown7' / cs.Bytes(8)
)

header = cs.Struct(
    'type' / cs.Enum(cs.Int8ub,
                     file_metadata=0x01,
                     filename=0x02,
                     file_type=0x03,
                     user=0x0A,
                     image_metadata=0x10,
                     end=0xFF),

    'record' / cs.Switch(cs.this.type, {'file_metadata': header_record_file_metadata,
                                        'filename': header_record_filename,
                                        'file_type': header_record_filetype,
                                        'user': header_record_user,
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


def dump_image_metadata_unknowns():
    file_names = glob.glob('*.vpb')

    for file_name in file_names:
        with open(file_name, 'rb') as f:
            h = f.read(2048)
            records = cs.RepeatUntil(lambda obj, lst, ctx: obj.type == 'end', header).parse(h)
            print('{0: <24}: {1}'.format(file_name, [binascii.hexlify(x) for x in records.search_all('im_unknown')]))


def convert_vpb_to_whatever(vpb_file_name, output_file_name):
    """
    Uses PIL to output the image, the format is automatically chosen by the file name extension.
    """
    with open(vpb_file_name, 'rb') as f:
        h = f.read(2048)
        records = cs.RepeatUntil(lambda obj, lst, ctx: obj.type == 'end', header).parse(h)
        buf = f.read(records.search('image_size_bytes'))

    # Image is Y'CbCr 4:2:2
    image_data = np.frombuffer(buf, dtype='uint8').reshape((records.search('height'), records.search('width') * 2))

    Y = (np.copy(image_data[:, 1::2]))
    Cb = np.repeat(np.copy(image_data[:, ::4]), 2, axis=1)
    Cr = np.repeat(np.copy(image_data[:, 2::4]), 2, axis=1)
    image = np.dstack((Y, Cb, Cr))
    bar = YCbCr_to_RGB(image)

    output = Image.fromarray(bar)

    output.save(output_file_name)


def convert_whatever_to_vpb(input_file_name, vpb_file_name):
    input_image = Image.open(input_file_name)

    if input_image.width != 720 or not (input_image.height == 576 or input_image.height == 488):
        # TODO: Scale and fit the incoming image to a TV size.
        raise ValueError

    # Convert to YCbCr colourspace, ignoring alpha.
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
        f.write(header_record_filetype.build(dict(file_type=u'Picture', ft_unknown=0x98)))

        dt = datetime.now()
        f.write(b'\x01')
        f.write(header_record_file_metadata.build(dict(record_size=16, fm_unknown=0,
                                                       day=dt.day, month=dt.month, year=dt.year - 1980,
                                                       hours=dt.hour, minutes=dt.minute, seconds=dt.second,
                                                       fm_unknown2=0)))

        f.write(b'\x02')
        f.write(header_record_filename.build(dict(file_name=path.splitext(path.basename(vpb_file_name))[0])))

        f.write(b'\x0a')
        f.write(header_record_user.build(dict(user='DEFAULT')))

        f.write(b'\x10')
        # These seem consistent with PAL-sized images.
        im_unknown2 = b'\x08\x37'
        im_unknown3 = b'\x08\xfc'
        im_unknown6 = b'\x0c\xac'

        if input_image.height == 488:
            # These seem consistent with NTSC-sized images.
            im_unknown2 = b'\x08\x55'
            im_unknown3 = b'\x07\x94'
            im_unknown6 = b'\x0a\xc0'

        f.write(header_record_image_metadata.build(dict(im_unknown1=b'\x01\x14\x00\x10\x00\x00',
                                                        width=input_image.width, height=input_image.height,
                                                        im_unknown2=im_unknown2, im_unknown3=im_unknown3,
                                                        im_unknown4=b'\x00\x00\x04\x00',
                                                        image_size_bytes=input_image.width * input_image.height * 2,
                                                        im_unknown5=b'\x04\x01\x04\x00',
                                                        im_unknown6=im_unknown6,
                                                        im_unknown7=b'\x00\x0d\x01\x04\x00\x00\x00\x00')))

        # Pad the header.
        f.write(b'\xff' * (1024 - f.tell()))
        f.write(b'\x00\x00')
        f.write(b'\xe5' * 1022)

        # Write the image bytes.
        f.write(YCbCr422.tobytes())


if __name__ == '__main__':
    # dump_image_metadata_unknowns()
    # convert_vpb_to_whatever(sys.argv[1], sys.argv[2])
    convert_whatever_to_vpb(sys.argv[2], sys.argv[1])
