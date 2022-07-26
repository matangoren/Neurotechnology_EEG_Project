from pylsl import StreamInlet, resolve_stream
import time
import pandas as pd
import matplotlib.pyplot as plt

print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
inlet.open_stream()


samples, timestamps = inlet.pull_chunk(5)

# form dataframe from data
df = pd.DataFrame(samples, columns=["Gray", "Purple", "Blue", "Green"])

# plot multiple columns such as population and year from dataframe
df.plot(y=["Gray", "Purple", "Blue", "Green"],
        kind="line", figsize=(10, 10), use_index=True)

# display plot
plt.show()




# while True:
#     # get a new sample (you can also omit the timestamp part if you're not
#     # interested in it)
#     sample, timestamp = inlet.pull_sample()
#     print(timestamp, sample)