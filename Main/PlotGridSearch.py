
import numpy as np
import matplotlib.pyplot as plt

def create_ranges(data, num_ranges=100):
    # Find the min and max
    min_val = np.min(data)
    max_val = np.max(data)

    # Create 100 ranges 
    value_ranges = np.linspace(min_val, max_val, num_ranges + 1)

    # #data points
    counts, _ = np.histogram(data, bins=value_ranges)

    return counts

your_dataset = np.random.rand(1000) * 100  

counts_in_ranges = create_ranges(your_dataset)

print("Number of points in each range:", counts_in_ranges)


def create_ranges(data, num_ranges=100):
    min_val = np.min(data)
    max_val = np.max(data)
    value_ranges = np.linspace(min_val, max_val, num_ranges + 1)
    counts, _ = np.histogram(data, bins=value_ranges)
    return counts, value_ranges

your_dataset = np.random.rand(1000) * 100 
counts_in_ranges, value_ranges = create_ranges(your_dataset)

# Plotting
plt.plot(value_ranges[:-1], counts_in_ranges, marker='o', linestyle='-', color='b')
plt.title('Number of Points in Each Range')
plt.xlabel('AUROC')
plt.ylabel('Number of models')
plt.show()