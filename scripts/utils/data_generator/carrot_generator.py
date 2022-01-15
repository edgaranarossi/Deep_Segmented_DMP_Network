from data_generator.utils import smooth
import numpy as np
from numpy.random import rand, randint
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from os.path import join

def generate_carrot(length = None, tip = 'right', smooth = True, smoothing_degree = 5):
    """
    Generates a 2D carrot shape in the scale of 100
    """
    length = np.array(length).ravel()
    if length.shape[0] > 2: raise ValueError("'length' value more than 2 dim")

    carrot = initializeShape()
    carrot = randomizeSize(carrot, length)
    carrot = randomizeShape(carrot)
    carrot = rotateCarrot(carrot)
    if tip == 'left': carrot = flipCarrot(carrot)
    if smooth: carrot = smoothenCarrot(carrot, smoothing_degree)

    return carrot

def plot_carrot(carrot):
    """
    Plots given carrot image
    """
    TEX_ROOT_DIR = 'assets'
    im = np.asarray(Image.open(join(TEX_ROOT_DIR, "carrot_texture_long.jpg")).resize((randint(100, 300), randint(25, 50)), Image.LANCZOS))
    p = Polygon(np.array(carrot), closed = False)
    p.set_color('none')
    ax = plt.gca()
    ax.add_patch(p)
    ax.imshow(im, clip_path = p, clip_on=True)

    leftmost_carrot = np.min(carrot[:,0])
    rightmost_carrot = np.max(carrot[:,0])
    bottommost_carrot = np.min(carrot[:,1])
    topmost_carrot = np.max(carrot[:,1])
    pad_X = (100 - (rightmost_carrot - leftmost_carrot)) / 2
    pad_Y = (100 - (topmost_carrot - bottommost_carrot)) / 2
    x_lim_down = (leftmost_carrot - pad_X)
    x_lim_up = (rightmost_carrot + pad_X)
    y_lim_down = (bottommost_carrot - pad_Y)
    y_lim_up = (topmost_carrot + pad_Y)
    plt.xlim(x_lim_down, x_lim_up)
    plt.ylim(y_lim_down, y_lim_up)
    plt.tick_params(bottom=False,     
                    top=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False)
    ax.set_facecolor('black')

def initializeShape():
    base_shape = [[  0.00, 12.50],
                  [  2.50,  0.00],
                  [ 25.00,  2.00],
                  [ 50.00,  4.00],
                  [ 75.00,  6.25],
                  [ 92.50,  7.50],
                  [100.00, 12.50],
                  [ 92.50, 17.50],
                  [ 75.00, 18.75],
                  [ 50.00, 20.75],
                  [ 25.00, 23.00],
                  [  2.50, 25.00]]
    carrot = np.array(base_shape)
    return carrot

def flipCarrot(carrot):
    carrot[:, 0] *= -1
    carrot[:, 0] += np.abs(carrot.min(axis = 0)[0])
    return carrot

def randomizeSize(carrot, length):
    # Default length randomizer (length == None)
    if length[0] == None: 
        min_length_modifier = carrot.max(axis=0)[0] * 3 / 4
        max_length_modifier = carrot.max(axis=0)[0]
        multiplier = 10**(3-len(str(min_length_modifier).split('.')[0]))
        min_length_modifier *= multiplier
        max_length_modifier *= multiplier
        length[0] = randint(min_length_modifier, max_length_modifier) / multiplier

    if length.shape[0] == 1:
        carrot = carrot * length[0] / carrot.max(axis=0)[0]
    elif length.shape[0] == 2:
        carrot[:, 0] = carrot[:, 0] * length[0] / carrot.max(axis=0)[0]
        carrot[:, 1] = carrot[:, 1] * length[1] / carrot.max(axis=0)[1]
    return carrot

def randomizeShape(carrot):
    max_ridge_size = carrot.max(axis = 0)[1] * 0.1
    for i in range(carrot.shape[0]):
        modifier = (-0.5 * max_ridge_size) + rand() * max_ridge_size
        carrot[i, 1] += modifier
    carrot = np.append(carrot, [carrot[0]], axis = 0)
    return carrot

def rotateCarrot(carrot):
    rot = np.deg2rad( -5 )
    rotated_carrot = np.copy(carrot)
    rotated_carrot[:, 0] = carrot[:, 0] * np.cos(rot) + carrot[:, 1] * np.sin(rot)
    rotated_carrot[:, 1] = carrot[:, 0] * np.sin(rot) + carrot[:, 1] * np.cos(rot)
    carrot = rotated_carrot
    return carrot

def smoothenCarrot(carrot, smoothing_degree):
    carrot = np.array(smooth(carrot.tolist(), number_of_refinements = smoothing_degree))
    return carrot