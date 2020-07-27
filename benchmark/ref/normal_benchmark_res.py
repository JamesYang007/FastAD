import numpy as np
import matplotlib.pyplot as plt

sizes = np.array([10, 50, 100, 500, 1000, 2000, 3000, 4000 ])
tot_sizes = (sizes + 1)**2 - 1
times = np.array([
    2159,
    58682,
    57420,
    1222708,
    2720733,
    16687945,
    47899494,
    104822861
])

slope = (times[-1] - times[-2]) / (tot_sizes[-1] - tot_sizes[-2])
print(slope)

plt.plot(tot_sizes, tot_sizes, '-',
         color='blue', alpha=0.5)
plt.show();
