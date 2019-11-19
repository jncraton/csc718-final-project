import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import imageio
import csv
import os

size = 10000000
radius = 6.357E6

frames = len(os.listdir('results')) - 1

print(f"Rendering {frames} frames...")

def plot(i):
    print(f"Generating frame {i}...") 
    v = [[] for i in range(6)]

    for row in csv.reader(open(f'results/{i}.csv')):
        for i in range(6): v[i].append(float(row[i]))
    fig = plt.figure()

    ax = fig.gca(projection='3d')   
    ax.set_xlim3d(-size,size)
    ax.set_ylim3d(-size,size)
    ax.set_zlim3d(-size,size)

    ax.quiver(v[0],v[1],v[2],v[3],v[4],v[5], length=1000, normalize=False)

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius*np.cos(u)*np.sin(v)
    y = radius*np.sin(u)*np.sin(v)
    z = radius*np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", linewidth=1)

    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

imageio.mimsave('./animation.gif', [plot(i) for i in range(frames)], fps=5)