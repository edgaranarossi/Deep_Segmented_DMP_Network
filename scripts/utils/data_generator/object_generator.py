import numpy as np
from numpy import array, zeros_like, clip, where
from numpy.random import rand, randint
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from os.path import join
import copy
from multiprocessing import Process
from .utils import smooth

class ObjectGenerator:
    def __init__(self, 
                 base_shape):
        self.base_shape = array(base_shape)

    def initialize(self):
        self.cur_shape = copy.deepcopy(self.base_shape)

    def generate(self, 
                 size_random_magnitude = None, 
                 shape_random_magnitude = None,
                 smoothing_magnitude = None,
                 plot_shape = False, 
                 plot_save_path = '',
                 plot_target_size = None):
        self.initialize()
        self.randomize_size(size_random_magnitude)
        self.randomize_shape(shape_random_magnitude, smoothing_magnitude)
        if plot_shape or bool(plot_save_path): 
            self.plot(plot_shape, plot_save_path, plot_target_size)
        return self.cur_shape
    
    def randomize_size(self, 
                       random_magnitude):
        random_magnitude = array(random_magnitude)
        if random_magnitude.sum() != None: 
            assert len(random_magnitude) == self.base_shape.shape[1]
        else:
            random_magnitude = zeros_like(self.base_shape.shape[1])
        rand_limit = (self.cur_shape.max(axis = 0) - self.cur_shape.min(axis = 0)) * random_magnitude
        modifier = ((-0.5 * rand_limit) + rand(self.cur_shape.shape[1]) * rand_limit)
        self.cur_shape = self.cur_shape + modifier
        self.cur_shape = clip(self.cur_shape, a_min = 0, a_max = None)
        min_shape = self.cur_shape.min(axis = 0)
        for i in range(self.cur_shape.shape[1]):
            self.cur_shape[:, i] = where(self.cur_shape[:, i] == min_shape[i], 0, self.cur_shape[:, i])

    def randomize_shape(self, 
                        random_magnitude,
                        smoothing_magnitude):
        random_magnitude = array(random_magnitude)
        if smoothing_magnitude != None:
            self.cur_shape_randomized = array(smooth(self.cur_shape.tolist(), smoothing_magnitude))
        else: 
            self.cur_shape_randomized = copy.deepcopy(self.cur_shape)
        if random_magnitude.sum() != None: 
            assert len(random_magnitude) == self.base_shape.shape[1]
        else:
            random_magnitude = zeros_like(self.base_shape.shape[1])
        rand_limit = (self.cur_shape_randomized.max(axis = 0) - self.cur_shape_randomized.min(axis = 0)) * random_magnitude
        modifier = ((-0.5 * rand_limit) + rand(*self.cur_shape_randomized.shape) * rand_limit)
        self.cur_shape_randomized = self.cur_shape_randomized + modifier
        self.cur_shape_randomized -= self.cur_shape_randomized.min(axis = 0)

    def plot(self, 
             show_plot = False,
             plot_save_path = None,
             target_size = None):

        if self.base_shape.shape[1] == 2:
            self.fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()
            ax.set_facecolor('black')

            p = Polygon(np.array(self.cur_shape_randomized), closed = False)
            # p.set_color('none')
            p.set_color([1, 1, 1])
            ax.add_patch(p)

            axis_length = (self.base_shape.max(axis = 0) - self.base_shape.min(axis = 0)).max()

            ax.set_xlim(-0.1, axis_length + 0.6)
            ax.set_ylim(-0.1, axis_length + 0.6)

            ax.tick_params(bottom=False,     
                           top=False,
                           left=False,
                           labelbottom=False,
                           labelleft=False)

        elif self.base_shape.shape[1] == 3:
            raise NotImplementedError()

        if bool(plot_save_path):
            p = Process(target = self.save_plot, args = (show_plot,
                                                         plot_save_path,
                                                         target_size))
            p.start()
            p.join()

        if show_plot:
            plt.show()
            # pass
        else:
            plt.cla()
            plt.clf()
            plt.close('all')
            plt.close(self.fig)

    def save_plot(self, show_plot, plot_save_path, target_size = None):
        plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.0)
            
        if target_size != None:
            assert len(target_size) == 2
            img = Image.open(plot_save_path)
            img_resize = img.resize((target_size[0], target_size[1]))
            img_resize.save(plot_save_path)