import numpy as np

arrays = np.array([1, 5, 10, 3, 4, 25, 30])

result = arrays[(arrays % 5 == 0) & (arrays % 2 == 1)]
print(result)
