audio.wave
================================================================================

    from numpy import *
    import audio.wave as wave

    df = 44100
    dt = 1.0 / df
    f = 440.0
    T = 3.0
    t = r_[0.0:T:dt]

    sound = sin(2.0 * pi * f * t)
    wave.write(sound, "A4.wav")


