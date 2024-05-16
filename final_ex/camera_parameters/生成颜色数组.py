import random

import numpy as np

colors = []
for i in range(20):
    colors.append([random.randint(0, 120), random.randint(0, 120), random.randint(0, 120)])

print(colors)