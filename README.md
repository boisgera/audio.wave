audio.wave
================================================================================

    import audio.wave
    from numpy import *

    df = 44100
    dt = 1.0 / df
    f = 440.0
    T = 3.0
    t = r_[0.0:T:dt]
    sound = sin(2*pi*f*t)

    audio.wave.write(sound, "A4.wav")


<audio control>
<source src="sounds/A4.wav" type="audio/wav">
</audio>
