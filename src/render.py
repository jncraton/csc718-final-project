import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import imageio
import csv
import os
import sys

size = 20000000
radius = 6.357E6

version = sys.argv[1].split('/')[-1].replace('.gif','')
results_dir = f'dist/{version}-results/'
frames = len(os.listdir(results_dir)) - 1

print(f"Rendering {frames} frames for {results_dir}...")

def plot(i):
    print(f"Generating frame {i}...") 
    v = [[] for i in range(8)]

    time = 0

    for row in csv.reader(open(f'{results_dir}/{i}.csv')):
        for i in range(8): v[i].append(float(row[i]))
        time = row[8]

    fig = plt.figure()

    ax = fig.gca(projection='3d')   
    ax.set_xlim3d(v[0][0]-size,v[0][0]+size)
    ax.set_ylim3d(v[1][0]-size,v[1][0]+size)
    ax.set_zlim3d(v[2][0]-size,v[2][0]+size)

    colors = [[0.2,0.8,0.2,0.3] if i == 0 else [0.8,0.2,0.2,1.0] for i in range(len(v[0]))]

    ax.scatter(v[0],v[1],v[2],s=[3000 * i/radius for i in v[7]],c=colors,depthshade=False)

    orbital_mass = sum(v[6][1:])

    plt.title(f't={time} N={len(v[0])} Mass={orbital_mass:.3e}')
    
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

imageio.mimsave(sys.argv[1], [plot(i) for i in [0,0,0] + list(range(frames))], fps=2)
