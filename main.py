from pylsl import StreamInlet, resolve_stream
import time

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
inlet.open_stream()
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)

samples, timestamps = inlet.pull_chunk(1)
print(samples)

t_end = time.time() + 60 * 15

while time.time() < t_end:



# while True:
#     # get a new sample (you can also omit the timestamp part if you're not
#     # interested in it)
#     sample, timestamp = inlet.pull_sample()
#     print(timestamp, sample)