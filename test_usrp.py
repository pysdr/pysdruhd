
import pysdruhd as uhd

x300 = uhd.Usrp(type="b200", rate=12.5e6)

sample_count = 0
for xx in xrange(1000):
    samples = x300.recv()
    sample_count += samples.__len__()

print sample_count
