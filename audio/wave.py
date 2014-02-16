#!/usr/bin/env python
# coding: utf-8

"""
Waveform Audio File Format (WAVE) Reader/Writer

This module supports [canonical WAVE files][canonical]: 

  - linear PCM, 
  - 16 bit / sample, 
  - mono or stereo.

[canonical]: https://ccrma.stanford.edu/courses/422/projects/WaveFormat

WAVE format online information: 

  - <https://ccrma.stanford.edu/courses/422/projects/WaveFormat>,
  - "Audio Formats Reference": <http://tinyurl.com/langenberger>,
  - <http://netghost.narod.ru/gff/graphics/summary/micriff.htm>,
  - <http://www.sonicspot.com/guide/wavefiles.html>,
  - <http://www.tactilemedia.com/info/MCI_Control_Info.html>.
"""

#
# Dependencies
# ------------------------------------------------------------------------------
# 
#   - the standard Python 2.7 library,
#   - the [NumPy](http://scipy.org/) library,
#   - `logger`, `script`, and `bitstream` from the digital audio coding project,
#   - optionally, [`lsprofcalltree`][lsprofcalltree].
# 
# [lsprofcalltree]: http://people.gnome.org/~johan/lsprofcalltree.py
#

# Standard Python Library
import cProfile
import inspect
import os
import os.path
import sys
import time

# Third-Party Libraries
from numpy import *
import lsprofcalltree

# Digital Audio Coding
from bitstream import BitStream
import logger # TODO: convert to logfile, remove tagging, error hooks, etc.
              # and add the dependency in the setup.py
import script

# TODO:
# 
#     Get rid of the "info" hack for read and write. Instead, for write,
#     use extra keyword arguments, and for read an optional `output`
#     attribute that is either a string or a list of strings that
#     determines the values to return. The parameters would be `data`, `df`
#     (instead of sample rate) ... and that's all ? Allow `dt` ? 
#     Keep `num_channels` or don't keep it ? The motivation would
#     be to have faster reads when `data` is not in the list ...
#     UPDATE: do NOT allow these extra attributes BUT introduce an argument
#     that is the number of samples to read. To get the number of channels,
#     one can read 0 sample and query the shape of the returned array.
#     Same thing for the "bit depath", (if we decide to support 8bits), 
#     you can get it from the array data type if you read without scaling.

#
# Metadata
# ------------------------------------------------------------------------------
#

from .about_wave import *

#
# Wave Writer
# ------------------------------------------------------------------------------
#
@logger.tag("wave.write")
def write(data, output=None, info=None, scale=None):
    r"""Wave Audio File Format Writer

    Arguments
    ---------

      - `data`: the audio data.
    
        The data should be either a 1-dim. numpy array or a 2-dim numpy 
        array with a dimension of 1 (mono) or 2 (stereo) along the first 
        axis. 

      - `output`: a bitstream, filename, file object or `None` (default).

        The object where the data is written in the WAVE format.
        An empty bitstream is created if no output is specified.

      - `info`: a dictionary or `None` (default).

        If there is a value attached to the `"sample rate"` key, 
        it does override the default sample rate of `44100`.

      - `scale`: the scaling policy: `None`, `True` or `False`.

        This argument determines the linear transformation that scales `data`
        before it is rounded and clipped the to 16-bit integer range.
        The following table displays what value is mapped to the largest signed 
        16-bit integer given `scale` and the type of `data`.


        `scale`                                float                              integer
        -------   ----------------------------------   ----------------------------------
        `None`    1.0 $\to$ $2^{15}-1$                 $2^{15}-1$ $\to$ $2^{15}-1$
        `True`    `amax(abs(data))` $\to$ $2^{15}-1$   `amax(abs(data))` $\to$ $2^{15}-1$
        `False`   $2^{15}-1$ $\to$ $2^{15}-1$          $2^{15}-1$ $\to$ $2^{15}-1$ 

        
        Advanced scaling policies can be specified: 

          - if `scale` is a number, `data` is multiplied by this number before
            the conversion to 16-bit integers. For example, for an array of floats,
            the `scale = None` policy could be implemented by setting `scale = 2**15 - 1`.

          - if `scale` is a function, it is given the `data` argument and should
            return a scale number. 
            For example, the policy `scale = True` is equivalent to the selection
            of the scaling function defined by `scale(data) = 2**15 - 1 / amax(data)`.

    Returns
    -------

      - `stream`: an output stream if no output was specified, `None` otherwise.

    See Also
    --------

      - `wave.read`,
      - `bitstream.Bitstream`.  
"""
    if isinstance(output, str):
        file = open(output, "w")
        stream = BitStream()
    elif isinstance(output, BitStream):
        file = None
        stream = output
    elif hasattr(output, "write"):
        file = output
        stream = BitStream()
    elif output is None:
        file = None
        stream = BitStream()

    data = array(data, copy=False)        
    if len(shape(data)) == 1:
        data = reshape(data, (1, len(data)))

    if scale is None:
        if issubclass(data.dtype.type, floating):
            A = float(2**15 - 1)
        else:
            A = 1
    elif scale is False:
        A = 1
    elif scale is True:
        A = float(2**15 - 1) / amax(abs(data))
    elif isinstance(scale, (int, float)):
        A = scale
    elif callable(scale):
        A = scale(data)
    else:
        raise TypeError("invalid scale specifier {0}.".format(scale))

    data = A * data

    # clip what's not in the [-2**15, 2**15) range
    if data.dtype.type != int16:
        ones_ = ones(shape(data))
        low  = (-2**15    ) * ones_
        high = ( 2**15 - 1) * ones_
        data = clip(data, low, high)
        data = data.astype(int16)

    info = info or {}
 
    # TODO: log some info.
  
    num_channels, num_samples = shape(data)
    file_size = 44 + 2 * size(data) # size in bytes
    stream.write("RIFF")
    chunk_size = file_size - 8
    stream.write(uint32(chunk_size).newbyteorder())
    stream.write("WAVE")
    stream.write("fmt ")
    sub_chunk1_size = 16
    stream.write(uint32(sub_chunk1_size).newbyteorder())
    audio_format = 1
    stream.write(uint16(audio_format).newbyteorder())
    stream.write(uint16(num_channels).newbyteorder())
    sample_rate = info.get("sample rate") or 44100
    stream.write(uint32(sample_rate).newbyteorder())
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample / 8)
    stream.write(uint32(byte_rate).newbyteorder())
    block_align = num_channels * bits_per_sample / 8
    stream.write(uint16(block_align).newbyteorder())
    stream.write(uint16(bits_per_sample).newbyteorder())
    stream.write("data")
    sub_chunk2_size = num_samples * num_channels * (bits_per_sample / 8)
    stream.write(uint32(sub_chunk2_size).newbyteorder())

    data = ravel(transpose(data))
    stream.write(int16(data).newbyteorder())

    if file:
        file.write(stream.read(str))
        try:
            file.close()
        except Attributeerror:
            try:
               file.flush()
            except Attributeerror:
               pass
    elif output is None:
        return stream        

#
# Wave Reader
# ------------------------------------------------------------------------------
#

        
@logger.tag("wave.read")
def read(input, info=None, scale=None):
    r"""
    Wave Audio File Format Reader

    Arguments
    ---------

      - `input`: the source of the WAVE data: a filename, file or a bitstream.
        

      - `info`: a dictionary or `None` (default).

         The value `info["sample rate"]` is set to the data sample rate if 
         `info` is not `None`.

      - `scale`: the scaling policy: `None` (the default), `True` or `False`.

        This argument determines the linear transformation that scales
        the array of 16-bit signed integers stored in `input` before it is
        returned.
        The following table displays the scaling that corresponds to three
        standard policies:
 
        `scale`                          scaling 
        --------   ----------------------------- 
         `None`             $2^{15}-1$ $\to$ 1.0 
         `True`      `amax(abs(data))` $\to$ 1.0
        `False`    $2^{15}-1$ $\to$ $2^{15} - 1$


        Advanced scaling policies can be specified: 

          - if `scale` is a number, it is used as a multiplier on the 16-bit  
            integer data. For example, `scale = None` corresponds to 
            `scale = 1.0 / float(2**15 - 1)` and `scale = False` to `scale = 1`.

          - if `scale` is a function, it is given the `data` argument and should
            return a scale multiplier. For example, the setting `scale = True` 
            is a shortcut for the function defined by `scale(data) = 1.0 / amax(abs(data))`.

    Returns
    -------
  
      - `data`: the audio data, as a 2-dim numpy array with a dimension of 1 
         (mono) or 2 (stereo) along the first axe. Its data type depends on
         the scaling policy.

    See Also
    --------

      - `wave.write`,
      - `bitstream.BitStream`.
"""
    logger.debug("loading the input.")

    if isinstance(input, str):
        file = open(input, "r")
        stream = BitStream(file.read())
    elif isinstance(input, BitStream):
        file = None
        stream = input
    elif hasattr(input, "read"):
        file = input
        stream = BitStream(file.read())

    if info is None:
        info = {}

    _read_header_chunk(stream, info)
    _read_format_chunk(stream, info)
    # TODO: need to take care of possible "LIST" chunks.
    # More generally, there is a pattern of 4 str (magic) + size (uint32)
    # then the data. We should exploit this structure in the code ...

    # hack so that I don't have to change the code of _read_data_chunk.
    # TODO: log the data in LIST
    stream_copy = stream.copy() # WOW, OMG, EVERYTHING IS COPIED. Should allow
    # PARTIAL copy in bitstream !
    if stream_copy.read(str, 4) == "LIST":
        assert stream.read(str, 4) == "LIST"
        num_bytes = stream.read(uint32).newbyteorder()
        _ = stream.read(uint8, num_bytes)
    # Rk: not very general, will work only with a single LIST chunk ...
    data = _read_data_chunk(stream, info)

    if scale is None:
        A = 1.0 / float(2**15 - 1)
    elif scale is False:
        A = 1
    elif scale is True:
        A = 1.0 / amax(abs(data))
    elif isinstance(scale, (int, float)):
        A = scale
    elif callable(scale):
        A = scale(data)
    else:
        raise TypeError("invalid scale specifier {0}.".format(scale))

    data = data * A 
    try:
        if file:
            file.close()
    except AttributEerror:
        pass

    return data


@logger.tag("wave.read.header")
def _read_header_chunk(stream, info):
    logger.debug("start of the header processing.")
    assert (len(stream) % 8 == 0)
    file_size = len(stream) / 8
    if file_size < 1024:
        file_size_B = file_size
        logger.info("file size: {file_size_B} B")
    elif file_size < 1024 * 1024:
        file_size_KiB = round(file_size / 1024.0, 1)
        logger.info("file_size: {file_size_KiB:.1f} KiB")
    else:
        file_size_MiB = round(file_size / 1024.0 / 1024.0, 1)
        logger.info("file_size: {file_size_MiB:.1f} MiB")
    logger.debug("file size (exact): {file_size} bytes.")
    magic = stream.read(str, 4)
    if magic != "RIFF":
        logger.error("invalid magic number {magic!r} (only 'RIFF' is supported).")
    chunk_size = stream.read(uint32).newbyteorder()
    logger.debug("chunk size: {chunk_size} bytes.")

    if (chunk_size + 8) != file_size:
        logger.error("file size ({file_size} bytes) inconsistent with the "
              "chunk size {chunk_size} bytes).")
    format = stream.read(str, 4)
    if format != "WAVE":
        logger.error("the format {format!r} is not supported (only 'WAVE' is).")
    logger.debug("end of the header processing.")

@logger.tag("wave.read.format")
def _read_format_chunk(stream, info):
    logger.debug("start of the format chunk processing.")
    format = stream.read(str, 4)
    if format != "fmt ":
       logger.error("invalid format {format!r}, only 'fmt ' is supported.")
    size = stream.read(uint32).newbyteorder()
    if size != 16:
        logger.error("inconsistent format chunk size {size}, should be 16 bits.")
    audio_format = stream.read(uint16).newbyteorder()
    if audio_format != 1:
       logger.error("invalid audio format {audio_format}, only PCM (1) is supported.")
    num_channels = stream.read(uint16).newbyteorder()
    info["num channels"] = num_channels
    if num_channels not in (1, 2):
        logger.error("invalid number of channels {num_channels}, "
              "neither mono (1) nor stereo (2).")
    logger.info("number of channels: {0}".format(num_channels))
    sample_rate = stream.read(uint32).newbyteorder()
    info["sample rate"] = sample_rate
    #if sample_rate != 44100:
    #    logger.error("invalid sample rate {sample_rate}, only 44100 Hz is supported.")
    byte_rate = stream.read(uint32).newbyteorder()
    block_align = stream.read(uint16).newbyteorder()
    bits_per_sample = stream.read(uint16).newbyteorder()
    if not bits_per_sample == 16:
        logger.error("invalid bit depth {bits_per_sample}, "
              "only 16 bits/sample is supported.")
    expected_block_align = num_channels * (bits_per_sample / 8)
    if not block_align == expected_block_align:
        logger.error("inconsistent number of bits per sample {block_align}, "
              "should be {expected_block_align}.")
    expected_byte_rate = sample_rate * num_channels * (bits_per_sample / 8) 
    if not byte_rate == expected_byte_rate:
        logger.error("inconsistent byte rate {byte_rate}, "
              "should be {expected_byte_rate} byte/s.")
    logger.debug("end of the format chunk processing.")

@logger.tag("wave.read.data")
def _read_data_chunk(stream, info, min_chunk_size=1000000, wait=2.0):
    logger.debug("start of the data chunk processing.")
    num_channels = info.get("num channels")
    data_ID = stream.read(str, 4)
    if data_ID != "data":
       logger.error("invalid data ID {data_ID} (only 'data' is supported).")
    size = stream.read(uint32).newbyteorder()
    bits_per_sample = 16
    num_samples = int(size / (bits_per_sample / 8) / num_channels)
    logger.info("number of samples: {num_samples} (x{num_channels})")
    
    num_remain = num_samples * num_channels
    # TODO: factor out the timing estimate pattern
    next_num_chunk = 0
    data_chunks = [zeros((0,), dtype=int16)] 
    while num_remain > 0:
        num_chunk = max(min_chunk_size, next_num_chunk)
        if num_chunk > num_remain:
            num_chunk = num_remain
        start = time.time()
        data_chunks.append(stream.read(int16, num_chunk).newbyteorder())
        delta = time.time() - start
        logger.debug("chunk size: {num_chunk}")
        logger.debug("chunk read time: {delta}")
        time_left = delta * (num_remain - num_chunk) / num_chunk
        logger.info("time left (estimate): {time_left} s")
        num_remain = num_remain - num_chunk
        next_num_chunk = int(wait / delta * num_chunk)
        logger.debug("next chunk size: {next_num_chunk}")
    data = concatenate(data_chunks)
    data = transpose(reshape(data, (num_samples, num_channels)))
    logger.debug("end of the data chunk processing.")    

    return data

#
# Profiling
# ------------------------------------------------------------------------------
#
def profile(command):
   """
   Generate a kcachegrind profile during the execution of `command`.

   The argument `command` should be usable in an `exec` statement.
   The profile is stored in the file `wave.kcg`.
   """
   output_file = open("wave.kcg", "w")   
   profile = cProfile.Profile()
   profile.run(command)
   kcg_profile = lsprofcalltree.KCacheGrind(profile)
   kcg_profile.output(output_file)
   output_file.close()   
   
#
# Unit Tests
# ------------------------------------------------------------------------------
#
def test_empty():
    """
Test empty data read/write:

    >>> stream = write(data=[])
    >>> read(stream).size
    0
    >>> stream = write(data=[[]])
    >>> read(stream).size
    0
"""

def test_int16():
    """
Test write/read round trip consistency for integers.

    >>> def int16_write_read(data):
    ...     stream = write(data)
    ...     return read(stream, scale=False)

    >>> data = arange(-2**15, 2**15)
    >>> all(data == int16_write_read(data))
    True
    """

def test_float64():
    """
Test write/read round trip consistency for floating-point numbers.

    >>> from numpy.testing import assert_array_max_ulp

    >>> def float64_write_read(data):
    ...     stream = write(data)
    ...     return read(stream)

    >>> input = [-1.0, 0.0, 1.0]
    >>> output = ravel(float64_write_read(input))
    >>> _ = assert_array_max_ulp(input, output, maxulp=1)

    >>> input = arange(-2**15 + 1, 2**15) / float(2**15 - 1)
    >>> output = ravel(float64_write_read(input))
    >>> _= assert_array_max_ulp(input, output, maxulp=1)
    """

def test_write_scale():
    """
Test the standard scaling policies of `write`.

Integer input data:

    >>> input = [[-2**8, 0, 2**8]]
    >>> stream = write(input)
    >>> output = read(stream, scale=False)
    >>> all(output == input)
    True

    >>> stream = write(input, scale=False)
    >>> output = read(stream, scale=False)
    >>> all(output == input)
    True

    >>> stream = write(input, scale=True)
    >>> output = read(stream, scale=False)
    >>> all(output == [[-2**15 + 1, 0, 2**15 - 1]])
    True

Floating-point input data:

    >>> input = [[-0.5, 0.0, 0.5]]
    >>> from numpy.testing import assert_array_max_ulp
    >>> stream = write(input)
    >>> output = read(stream)
    >>> all(abs(output - input) <= 2**(-15))
    True

    >>> stream = write(input, scale=True)
    >>> output = read(stream)
    >>> _ = assert_array_max_ulp(output, [[-1.0, 0.0, 1.0]], maxulp=1)

    >>> input = arange(-2**15 + 1, 2**15).astype(float64)
    >>> stream = write(input, scale=False)
    >>> output = read(stream, scale=False)
    >>> all(output == input)
    True
    """

def test_read_scale():
    """
Test the standard scaling policies of `read`.

    >>> from numpy.testing import assert_array_max_ulp

    >>> input = [[0, 2**8]]
    >>> stream = write(input)
    >>> output = read(stream)
    >>> _ = assert_array_max_ulp(output, [[0.0, 2**8/float(2**15-1)]], maxulp=1)

    >>> input = [[0, 2**8]]
    >>> stream = write(input)
    >>> output = read(stream, scale=False)
    >>> all(output == input)
    True

    >>> input = [[0, 2**8]]
    >>> stream = write(input)
    >>> output = read(stream, scale=True)
    >>> _ = assert_array_max_ulp(output, [[0.0, 1.0]], maxulp=1)
"""


def test(verbose=False):
    """
    Run the unit tests
    """
    import doctest
    return doctest.testmod(verbose=verbose)

#
# Command-Line Interface
# ------------------------------------------------------------------------------
#

# TODO: introduce -r -w (read/write) or detection by extension (.wav or .npy)
#       or -i -e (import / export), maybe more explicit (numpy POV still).
#       document that .npy is the numpy save/load binary format.

def help():
    """
Return the following message:
"""
    message = \
"""
usage: 
    python {filename} [OPTIONS] FILENAME

options: -h, --help ............................ display help message and exit,
         -o OUTPUT, --output=OUTPUT ............ select output filename,
         -p, --profile ......................... generate kcachegrind data,
         -s, --silent .......................... silent mode (may be repeated),
         -t, --test ............................ run the module self tests,
         -v, --verbose ......................... verbose mode (may be repeated).
"""
    return message.format(filename=os.path.basename(__file__))

help.__doc__ = "\nReturn the following message:\n\n" + \
               "\n".join(4*" " + line for line in help().splitlines()) 

def main(args, options):
    """
    Command-line main entry point
    """
    first = script.first

    if options.help:
        print help()
        sys.exit(0)
    if options.test:
        verbose = bool(options.verbose)
        test_results = test(verbose=verbose)
        sys.exit(test_results.failed)
    if not args:
        print help()
        sys.exit(1)

    filename = first(args)

    # logger configuration
    verbosity = len(options.verbose) - len(options.silent)
    logger.config.level = verbosity
    def _format(channel, message, tag):
        tag = tag or ""
        return " {0!r:<9} | {1:<16} | {2}\n".format(channel, tag, message)
    logger.config.format = _format
    def _error_hook(message, ExceptionType=ValueError):
        raise ExceptionType(message)
    logger.error.set_hook(_error_hook)

    output = first(options.output)
    if output:
        output_file = open(output, "w")
    else:
        base, ext = os.path.splitext(filename) 
        output_file = open(base + ".npy", "w")

    data = read(filename)
    save(output_file, data)

if __name__ == "__main__":
    # TODO: implement 'check' (aka dry run)
    options_list = "check help output= silent verbose profile test"
    options, args = script.parse(options_list)
    
    if options.profile:
        profile("main(args, options)")
    else:
        main(args, options)

