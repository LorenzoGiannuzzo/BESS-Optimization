import matplotlib.pyplot as plt
import numpy as np

def degradation(cycle_num):

    capacity_remaining = (
            -0.00000000000000000000000000000005613 * cycle_num ** 9 +
            0.000000000000000000000000003121 * cycle_num ** 8 -
            0.00000000000000000000006353 * cycle_num ** 7 +
            0.000000000000000000663 * cycle_num ** 6 -
            0.000000000000003987 * cycle_num ** 5 +
            0.00000000001435 * cycle_num ** 4 -
            0.0000000307 * cycle_num ** 3 +
            0.00003746 * cycle_num ** 2 -
            0.0277 * cycle_num + 100
    )

    return capacity_remaining

cycle_nums = np.arange(0, 7001, 1)

prova = degradation(np.arange(0,7000,1,dtype=float))

plt.plot(np.arange(0,7000,1,dtype=float), prova, label='Capacity Remaining')
plt.xlabel('Number of Cycles')
plt.ylabel('Capacity Remaining (%)')
plt.title('Battery Capacity Degradation Over Cycles')
plt.grid(True)
plt.legend()


