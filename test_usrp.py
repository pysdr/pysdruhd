
import pysdruhd as uhd
import matplotlib.pyplot as plt

x300 = uhd.Usrp(type="x300", streams={"A:0": {'frequency':900e6}}, rate=12.5e6)
sensors = x300.sensor_names()
for sensor in sensors:
    print sensor
    print x300.get_sensor(sensor)

print x300.set_master_clock_rate(200e6)

x300.send_stream_command({'now': True})

sample_count = 0
for xx in xrange(10):
    samples, metadata = x300.recv()
    print metadata
    sample_count += samples.__len__()

print sample_count

plt.psd(samples[0])
