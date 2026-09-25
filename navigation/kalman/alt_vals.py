import numpy as np
import pandas as pd
import time
from collections import deque
def emit(path, rate, window_seconds=1):
    df = pd.read_csv(path)
    df[' ALTITUDE'] = df[' ALTITUDE'] + np.random.normal(0, 10, len(df))
    alt = list(df[' ALTITUDE'])
    window_samples = int(rate * window_seconds)
    window = deque(maxlen=window_samples)
    current_avg = 0.0
    sampled_avg = []
    for i in range(len(alt)):
        if len(window) == window_samples:
            oldest = window[0]
            current_avg = current_avg + (alt[i] - oldest) / window_samples
        window.append(alt[i])
        if len(window) < window_samples:
            current_avg = np.mean(window)
        sampled_avg.append(round(float(current_avg), 1))
        print("Sampled avg: ",sampled_avg)
        #time.sleep(0.01)
    return [alt, sampled_avg]

