import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def lpf(alpha, rate):
    df = pd.read_csv(".../data/stress_csv.csv")
    df[" ALTITUDE"] = df[" ALTITUDE"] + np.random.normal(0, 10, len(df))
    alt = list(df[" ALTITUDE"])
    filtered = []
    y_prev = alt[0] # initialize filter with the first raw reading
    filtered.append(round(y_prev, 1))
    for i in range(len(df)-1):
        y_k = (alpha * y_prev) + ((1 - alpha) * alt[i])
        filtered.append(round(y_k, 1))
        y_prev = y_k
    plot(alt, filtered,rate, alpha)
    
def plot(raw_vals, filtered_vals,rate, alpha):
    t = [i / rate for i in range(len(raw_vals))]
    plt.figure(figsize=(9, 5))
    plt.plot(t, raw_vals, linestyle='-', color='tab:blue', alpha=0.6, label='Sensor readings (raw)')
    plt.plot(t, filtered_vals, linestyle='-', color='tab:red', linewidth=2, label=f'Low-pass filtered (alpha={alpha})')
    plt.xlabel('Sample index')
    plt.ylabel('Sensor value')
    plt.title(f'Sensor readings vs Low-pass filter (rate={rate} Hz, alpha={alpha})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'sensor_vs_avg_{rate} @ {alpha}.png', dpi=150)
    plt.show()
if __name__ == '__main__':
    lpf(0.7, 50)