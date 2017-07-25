from __future__ import print_function

import pysdruhd as uhd
import matplotlib.pyplot as plt
import numpy as np

print("using pysdruhd located in", uhd.__file__)

center_freq = 100e6
samp_rate = 4e6

usrp = uhd.Usrp(type="b200", streams={"A:A": {'frequency':center_freq, 'gain':70}}, rate=samp_rate, gain=30.0)
usrp.send_stream_command({'now': True})
samples, metadata = usrp.recv()
plt.psd(samples[0], NFFT=512, Fs=samp_rate/1e6, Fc=center_freq/1e6)
plt.show()
