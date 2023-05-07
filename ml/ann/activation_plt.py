# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import activation

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_xticks([-10, -5, 0, 5, 10])
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_yticks([-1, -0.5, 0.5, 1])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
x = np.linspace(-10, 10)
y = activation.logistic(x)
plt.plot(x, y, label='logistic', color='red')
plt.legend()

ax = fig.add_subplot(122)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_xticks([-10, -5, 0, 5, 10])
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_yticks([-1, -0.5, 0.5, 1])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
x = np.linspace(-10, 10)
y = activation.tanh(x)
plt.plot(x, y, label='tanh', color='red')
plt.legend()
plt.show()