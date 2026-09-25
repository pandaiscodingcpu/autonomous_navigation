'''
Simulates a sensor reading altitude data at 5 Hz by reading through a CSV file containing 2,000 altitude values.
Each sample is passed through a moving-average filter to smooth out noise. Once per second (every 5 samples), the current
averaged altitude value is written to a new output .txt file, producing a downsampled,
filtered log of altitude over time (~400 seconds of simulated data, given 2000 samples at 5 Hz).
'''
import matplotlib.pyplot as plt
from alt_vals import emit
PATH = "../data/stress_csv.csv"
RATE = 10

k , avg_values= emit(PATH,RATE)
t = [i / RATE for i in range(len(k))]
plt.figure(figsize=(9, 5))
plt.scatter(t, k,marker='o', color='tab:blue', label='Sensor readings (k)')
plt.plot(t, avg_values,linestyle='-', color='tab:red', label='Running average (Avg)')
plt.xlabel('Time (s)')
plt.ylabel('Sensor value')
plt.title(f'Sensor readings vs Running average (rate={RATE} Hz)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'sensor_vs_avg_{RATE}.png', dpi=150)
plt.show()