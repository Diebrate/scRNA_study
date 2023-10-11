import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
Z = 3 * (X ** 2) + (Y ** 2) - 2 * X * Y + X - Y
gradient_x, gradient_y = np.gradient(Z, x[1] - x[0], y[1] - y[0])
plt.contour(X, Y, Z, levels=20, colors='gray')
# plt.quiver(X, Y, -gradient_x, -gradient_y, scale=40, color='blue')
plt.title("Contour Plot and Gradients")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()