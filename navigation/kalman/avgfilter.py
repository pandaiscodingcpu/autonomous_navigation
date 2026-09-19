'''
Simulates a sensor reading altitude data at 5 Hz by reading through a CSV file containing 2,000 altitude values.
Each sample is passed through a moving-average filter to smooth out noise. Once per second (every 5 samples), the current
averaged altitude value is written to a new output .txt file, producing a downsampled,
filtered log of altitude over time (~400 seconds of simulated data, given 2000 samples at 5 Hz).
'''






