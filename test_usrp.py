
import pysdruhd as uhd

x300 = uhd.Usrp(address="192.168.40.2", rate=50e6)

sample_count = 0
for xx in xrange(100):
    samples = x300.recv()
    sample_count += samples.__len__()

print sample_count
