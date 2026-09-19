import numpy as np
import pandas as pd
import time
df = pd.read_csv('../data/stress_csv.csv')
alt = list(df[' ALTITUDE'])
# start index
start = 0
# end index
end = 4
window_size = end - start + 1 # 5 Hz

# looping for time
for a in range(len(alt)):
    time.sleep(1)
    print(alt[start:end+1],"S: ",start,"E: ",end)
    start+=1
    end+=1















