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


<audio src="https://github.com/boisgera/audio.wave/blob/master/sounds/A4.wav?raw=true" controls>
</audio>


