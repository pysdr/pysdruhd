
import pysdruhd as uhd
import matplotlib.pyplot as plt

streams_config = {
        'A:0': {'mode': 'RX', 'frequency':100e6, 'rate':100e6, 'gain':70.},
        'A:1': {'mode': 'RX', 'frequency':100e6, 'rate':100e6, 'gain':70.},
        'B:0': {'mode': 'RX', 'frequency':100e6, 'rate':100e6, 'gain':70.},
        'B:1': {'mode': 'RX', 'frequency':100e6, 'rate':100e6, 'gain':70.},
        }

x300 = uhd.Usrp(streams=streams_config)

for xx in xrange(5000000):
    samples = x300.recv()
print "got samples in python"
print samples.shape
print samples.dtype

plt.figure()
plt.psd(samples[0])
plt.figure()
plt.psd(samples[1])
plt.show()
