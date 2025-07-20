#%%
from revolution_api.bitalino import *

from stream_processor_bit import *

#%%
mac_address = "/dev/tty.BITalino-3C-C2" 
device = BITalino(mac_address)
#%%
device.battery(10)

streamer = BitaStreamer(device)

#%%
streamer.stream_raw()
for i, data in enumerate(streamer.stream_raw()):
    if i <=20:
        print(data)