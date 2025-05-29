import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri


fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.subplots_adjust(left=0.05, right=0.93, bottom=0.02, top=0.9, wspace=0.05)

labels = ["Swimmer-a", "HalfCheetah-a", "Ant-a", "Walker2d-a", "Hopper-a", "Hopper-3obj-a", "Hopper-e", "Hopper-3obj-e"]

N = 8
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
x = np.append(np.sin(theta), 0)
y = np.append(np.cos(theta), 0)
triangles = [[N, i, (i + 1) % N] for i in range(N)]
triang_backgr = tri.Triangulation(x, y, triangles)

HV = np.array([[3.14,  5.72, 5.85, 4.91, 1.99, 2.85, 2.03, 3.33, 3.14, ],
               [3.15,  5.71, 5.88, 4.91, 1.99, 2.90, 2.04, 3.37, 3.15, ]])
SP = np.array([[11.56, 0.06, 0.62, 0.24, 0.37, 0.13, 0.30, 0.14, 11.56,],
               [11.58, 0.06, 0.92, 0.26, 0.26, 0.12, 0.25, 0.12, 11.58,]])
RM = np.array([[0.56,  0.46, 0.55, 1.02, 1.05, 2.26, 2.62, 2.62, 0.56, ],
               [0.55,  0.35, 0.53, 0.97, 1.33, 1.84, 2.42, 2.38, 0.55, ]])

for i in range(9):
    mx = max(HV[0, i], HV[1, i])
    HV[0, i] = int(HV[0, i] / mx * 5)/5.0
    HV[1, i] = int(HV[1, i] / mx * 5)/5.0

    mx = max(SP[0, i], SP[1, i])
    SP[0, i] = int(SP[0, i] / mx * 5)/5.0
    SP[1, i] = int(SP[1, i] / mx * 5)/5.0

    mx = max(RM[0, i], RM[1, i])
    RM[0, i] = int(RM[0, i] / mx * 5)/5.0
    RM[1, i] = int(RM[1, i] / mx * 5)/5.0

x = np.append(np.sin(theta), 0)
y = np.append(np.cos(theta), 1)

for idx, ax in enumerate(axs):
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.triplot(triang_backgr, color='lightgrey', lw=1)
    for i in range(5):
        ax.plot(x * i / 5.0, y * i / 5.0, color='lightgrey', lw=1)
    ax.axis('off')



triang_foregr = tri.Triangulation(x * HV[0, :], y * HV[0, :], triangles)
axs[0].plot(x * HV[0, :], y * HV[0, :],  color=(0.8392,0.1529,0.1569), lw=3, marker="o", label="w/o slider")
axs[0].fill(x * HV[0, :], y * HV[0, :],  color=(0.8392,0.1529,0.1569), alpha=0.2)

triang_foregr = tri.Triangulation(x * HV[1, :], y * HV[1, :], triangles)
axs[0].plot(x * HV[1, :], y * HV[1, :],  color="steelblue", lw=3, marker="o", label="w/ slider")
axs[0].fill(x * HV[1, :], y * HV[1, :],  color="steelblue", alpha=0.2)
axs[0].set_title("HV↑", fontsize=20)

triang_foregr = tri.Triangulation(x * RM[0, :], y * RM[0, :], triangles)
axs[1].plot(x * RM[0, :], y * RM[0, :],  color=(0.8392,0.1529,0.1569), lw=3, marker="o")
axs[1].fill(x * RM[0, :], y * RM[0, :],  color=(0.8392,0.1529,0.1569), alpha=0.2)

triang_foregr = tri.Triangulation(x * RM[1, :], y * RM[1, :], triangles)
axs[1].plot(x * RM[1, :], y * RM[1, :],  color="steelblue", lw=3, marker="o")
axs[1].fill(x * RM[1, :], y * RM[1, :],  color="steelblue", alpha=0.2)
axs[1].set_title("RD↓", fontsize=20)

triang_foregr = tri.Triangulation(x * SP[0, :], y * SP[0, :], triangles)
axs[2].plot(x * SP[0, :], y * SP[0, :],  color=(0.8392,0.1529,0.1569), lw=3, marker="o")
axs[2].fill(x * SP[0, :], y * SP[0, :],  color=(0.8392,0.1529,0.1569), alpha=0.2)

triang_foregr = tri.Triangulation(x * SP[1, :], y * SP[1, :], triangles)
axs[2].plot(x * SP[1, :], y * SP[1, :],  color="steelblue", lw=3, marker="o")
axs[2].fill(x * SP[1, :], y * SP[1, :],  color="steelblue", alpha=0.2)
axs[2].set_title("SP↓", fontsize=20)

fig.legend(fontsize=16)

for label, xi, yi in zip(labels,  x, y):
    axs[0].text(xi * 1.05, yi * 1.05, label,  # color=cmap(color),
             ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
             va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center',
             fontsize=12)
    
    
fig.savefig('plots/hexagon.pdf')