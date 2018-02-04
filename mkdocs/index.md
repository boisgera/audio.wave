Overview
==================================================================================

    import audio.wave
    from numpy import *

    df = 44100
    dt = 1.0 / df
    f = 440.0
    T = 3.0
    t = r_[0.0:T:dt]
    sound = sin(2*pi*f*t)

    audio.wave.write(sound, "A4.wav")

<div align=center>
<audio src="https://github.com/boisgera/audio.wave/blob/master/sounds/A4.wav?raw=true" controls>
</audio>
</div>


## `audio.wave.write`

Wave Audio File Format Writer

### Arguments


  - `data`: the audio data.

    The data should be either a 1-dim. numpy array or a 2-dim numpy 
    array with a dimension of 1 (mono) or 2 (stereo) along the first 
    axis. 

  - `output`: a bitstream, filename, file object or `None` (default).

    The object where the data is written in the WAVE format.
    An empty bitstream is created if no output is specified.

  - `df`: an integer, the sample rate (default: `44100`).

  - `scale`: the scaling policy: `None`, `True` or `False`.

    This argument determines the linear transformation that scales `data`
    before it is rounded and clipped the to 16-bit integer range.
    The following table displays what value is mapped to `2**15` 
    given `scale` and the type of `data`.

      - `scale = None`: `2**15` is mapped to `1.0`

      - `scale = True`: `amax(abs(data))` is mapped to `1.0`

      - `scale = False`: `2**15` is mapped to `2**15`


    `scale` |  scaling (float)              |  scaling (integer)
    --------|-------------------------------|-------------------------------
    `None`  |  `1.0` to `2**15`             |  `2**15` to `2**15`
    `True`  |  `amax(abs(data))` to `2**15` |  `amax(abs(data))` to `2**15`
    `False` |  `2**15` to `2**15`           |  `2**15` to `2**15`

    
    Advanced scaling policies can be specified: 

      - if `scale` is a number, `data` is multiplied by this number before
        the conversion to 16-bit integers. For example, for an array of floats,
        the `scale = None` policy could be implemented by setting 
        `scale = 2**15`.

      - if `scale` is a function, it is given the `data` argument and should
        return a scale number. 
        For example, the policy `scale = True` is equivalent to the selection
        of the scaling function defined by `scale(data) = 2**15 / amax(data)`.

### Returns

  - `stream`: an output stream if no output was specified, `None` otherwise.

### See Also

  - `audio.wave.read`,
  - `bitstream.Bitstream`.  

## `audio.wave.read`

Wave Audio File Format Reader

### Arguments

  - `input`: the source of the WAVE data: a filename, file or a bitstream.
    
  - `scale`: the scaling policy: `None` (the default), `True` or `False`.

    This argument determines the linear transformation that scales
    the array of 16-bit signed integers stored in `input` before it is
    returned.
    The following table displays the scaling that corresponds to three
    standard policies:


    `scale` |  scaling      
    --------|------------------------------
    `None`  | `2**15` to `1.0`            
    `True`  | `amax(abs(data))` to `1.0` 
    `False` | `2**15` to `2**15`         

    Advanced scaling policies can be specified: 

      - if `scale` is a number, it is used as a multiplier on the 16-bit  
        integer data. For example, `scale = None` corresponds to 
        `scale = 1.0 / float(2**15)` and `scale = False` to `scale = 1`.

      - if `scale` is a function, it is given the `data` argument and should
        return a scale multiplier. For example, the setting `scale = True` 
        is a shortcut for the function defined by `scale(data) = 1.0 / amax(abs(data))`.

  - `returns`: a string of comma-separated variable names.
    When `returns` is a single variable name, without a trailing comma, the
    value with this name is returned ; otherwise the named value(s) is (are) 
    returned as a tuple.
    
### Returns

The set of returned values is selected by the `returns` argument among:

  - `data`: the audio data, as a 2-dim numpy array with a dimension of 1 
     (mono) or 2 (stereo) along the first axe. Its data type depends on
     the scaling policy.

  - `df`: the sampling rate, an integer.

### See Also


  - `audio.wave.write`,
  - `bitstream.BitStream`.

