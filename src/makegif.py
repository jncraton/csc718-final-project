import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import imageio
import csv

def plot(i):
    print(f"Generating frame {i}...") 
    v = [[] for i in range(6)]

    for row in csv.reader(open(f'results/{i}.csv')):
        for i in range(6): v[i].append(float(row[i]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')    
    
    ax.quiver(v[0],v[1],v[2],v[3],v[4],v[5], length=1000, normalize=False)

    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./animation.gif', [plot(i) for i in range(9)], fps=1)