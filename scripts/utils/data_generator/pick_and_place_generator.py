import numpy as np
from numpy.random import rand, randint
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from time import sleep
from PIL import Image, ImageOps
from os import makedirs
from os.path import join, isdir
from multiprocessing import Process
from datetime import datetime
import pickle as pkl
from pydmps import DMPs_discrete
from copy import copy, deepcopy
import random
import itertools
import math

class Block:
    def __init__(self, dim, color, initial_position):
        self.dim = dim
        self.x, self.y, self.z = dim
        self.x_pos, self.y_pos, self.z_pos = initial_position
        # self.center = (self.x_pos + (self.x / 2), self.y_pos + (self.y / 2))
        self.reCenter()
        self.color = color
        self.zorder = 0

    def plot(self, ax, z_max):
        if self.z_pos: z_max += 0.1
        size_modifier = (z_max - self.z_pos) / z_max
        new_x = self.x / size_modifier
        new_y = self.y / size_modifier
        offset_x = (new_x - self.x) / 2
        offset_y = (new_y - self.y) / 2
        box = Rectangle(xy = (self.x_pos - offset_x, self.y_pos - offset_y), 
                        width = new_x, 
                        height = new_y, 
                        zorder = self.z_pos)
        box.set_color(self.color)
        ax.add_patch(box)
        
    def moveCenter(self, new_center, new_z):
        self.x_pos = new_center[0] - (self.x / 2)
        self.y_pos = new_center[1] - (self.y / 2)
        self.z_pos = new_z
        self.reCenter()
        # self.center = (new_center[0], new_center[1])
        
    def pos(self):
        return [self.center[0], self.center[1], self.z_pos]
    
    def reCenter(self):
        self.center = (self.x_pos + (self.x / 2), self.y_pos + (self.y / 2))
        
class Gripper:
    def __init__(self, initial_position, gripper_x_size, color = 'black', gripper_max = 100):
        self.x_pos, self.y_pos, self.z_pos = initial_position
        self.default_pos = copy(initial_position)
        self.gripper_max = gripper_max
        self.distance = gripper_max
        self.gripper_x_size = gripper_x_size
        self.r = 0.75
        self.color = color
        
    def plot(self, ax, z_max):
        if self.z_pos: z_max += 0.1
        size_modifier = (z_max - self.z_pos) / z_max
        new_r = self.r / size_modifier
        l_pos = (self.x_pos - (((self.distance / self.gripper_max) * self.gripper_x_size) / 2) / size_modifier)
        r_pos = (self.x_pos + (((self.distance / self.gripper_max) * self.gripper_x_size) / 2) / size_modifier)
        l_gripper = Circle(xy = (l_pos, self.y_pos),
                           radius = new_r,
                           zorder = 9999)
        r_gripper = Circle(xy = (r_pos, self.y_pos),
                           radius = new_r,
                           zorder = 9999)
        l_gripper.set_color(self.color)
        ax.add_patch(l_gripper)
        r_gripper.set_color(self.color)
        ax.add_patch(r_gripper)
        
    def moveCenter(self, new_center, new_z):
        self.x_pos = new_center[0]
        self.y_pos = new_center[1]
        self.z_pos = new_z
        
    def setDistance(self, new_distance):
        self.distance = int(np.min([self.gripper_max, new_distance]))

class Map:
    def __init__(self, dim):
        self.dim = dim
        self.x_max, self.y_max, self.z_max = dim
        self.objects = []
        self.gripper = None
        self.gripped_obj = None
        self.gripper_history = []
        
    def addTarget(self, target, radius = 10, color = 'm'):
        self.target = target
        self.target_radius = radius
        self.target_color = color
        
    def addObject(self, obj):
        self.objects.append(obj)
        
    def addGripper(self, obj):
        self.gripper = obj
        self.noteGrippperHistory()
    
    def plot(self, hide_borders = False, show_gripper = False, show_trail = False, save_path = None, save_size = (100, 100), show_plot = True):
        self.fig = plt.figure(figsize = (12, 12))
        # plt.title('Top View')
        plt.xlim(0, self.x_max)
        plt.ylim(0, self.y_max)
        ax = plt.gca()
        
        if hide_borders:
            ax.tick_params(bottom=False,     
                            top=False,
                            left=False,
                            labelbottom=False,
                            labelleft=False)
        else:
            ax.set_xlabel('X-Axis')
            ax.set_ylabel('Y-Axis')
        
        target = Circle(xy = (self.target[0], self.target[1]),
                           radius = self.target_radius,
                           zorder = -1)
        target.set_color(self.target_color)
        ax.add_patch(target)
            
        for i, obj in enumerate(self.objects):
            obj.plot(ax, self.z_max)
            
        if show_gripper: self.gripper.plot(ax, self.z_max)
        
        if show_trail and len(self.gripper_history) > 1:
            plt.plot([self.gripper_history[-2][0], self.gripper_history[-1][0]], [self.gripper_history[-2][1], self.gripper_history[-1][1]], lw = 3, ls = '--', c = 'black')
        
        if save_path != None:
            p = Process(target = self.save_plot, args = (save_path,
                                                         save_size))
            p.start()
            p.join()
            
        if show_plot:
            plt.show()
        else:
            plt.cla()
            plt.clf()
            plt.close('all')
            plt.close(self.fig)
            
    def plot_side(self,):
        self.fig = plt.figure(figsize = (12, 12))
        plt.title('Side View')
        plt.xlim(0, self.x_max)
        plt.ylim(0, self.z_max)
        ax = plt.gca()
        
        for obj in self.objects:
            if type(obj) is Block:
                box = Rectangle(xy = (obj.x_pos, obj.z_pos), 
                                width = obj.x, 
                                height = obj.z, 
                                zorder = obj.z_pos)
                box.set_color(obj.color)
                ax.add_patch(box)
                
        plt.show()
            
    
    def save_plot(self, save_path, target_size = None):
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
            
        if target_size != None:
            assert len(target_size) == 2
            img = Image.open(save_path)
            img_resize = img.resize((target_size[0], target_size[1]))
            img_resize.save(save_path)
            
    def moveGripperXY(self, new_xy):
        self.gripper.moveCenter(new_xy, self.gripper.z_pos)
        if self.gripped_obj != None:
            self.objects[self.gripped_obj].moveCenter(new_xy, self.gripper.z_pos)
        self.noteGrippperHistory()
        
    def moveGripperXYZ(self, new_xy, new_z):
        self.gripper.moveCenter(new_xy, new_z)
        if self.gripped_obj != None:
            self.objects[self.gripped_obj].moveCenter(new_xy, new_z)
        self.noteGrippperHistory()
        
    def moveGripperUp(self):
        self.gripper.moveCenter((self.gripper.x_pos, self.gripper.y_pos), self.gripper.default_pos[-1])
        if self.gripped_obj != None:
            self.objects[self.gripped_obj].moveCenter((self.gripper.x_pos, self.gripper.y_pos), self.gripper.default_pos[-1])
        self.noteGrippperHistory()
        
    def moveGripperDown(self):
        self.gripper.moveCenter((self.gripper.x_pos, self.gripper.y_pos), 0)
        if self.gripped_obj != None:
            self.objects[self.gripped_obj].moveCenter((self.gripper.x_pos, self.gripper.y_pos), 0)
        self.noteGrippperHistory()
        
    def resetGripperPos(self):
        if self.gripper.distance != self.gripper.gripper_max:
            self.openGripper()
        if self.gripper.z_pos != self.gripper.default_pos[-1]:
            self.moveGripperUp()
        self.moveGripperXY(self.gripper.default_pos[:-1])
    
    def openGripper(self):
        self.gripper.distance = self.gripper.gripper_max
        self.checkObject()
        self.noteGrippperHistory()
        
    def closeGripper(self):
        self.gripper.distance = 0
        self.checkObject()
        self.noteGrippperHistory()
        
    def checkObject(self, threshold = 1):
        GRIP_THRESHOLD = threshold
        for i, obj in enumerate(self.objects):
            dist = float(np.sqrt((self.gripper.x_pos - obj.center[0])**2 + (self.gripper.y_pos - obj.center[1])**2))
            # print(self.gripped_obj, i, dist, GRIP_THRESHOLD)
            if self.gripped_obj == None and self.gripper.distance < self.gripper.gripper_max/2 and dist < GRIP_THRESHOLD:
                self.gripped_obj = i
            elif self.gripped_obj is not None and self.gripper.distance > self.gripper.gripper_max/2:
                self.gripped_obj = None
        return
            
    def moveObjectByGripper(self, obj_idx, target, plot_steps = False, delay = 0, show_gripper = False, show_trail = False, motion_noise_magnitude = None):
        obj = self.objects[obj_idx]
        self.motion_noise_magnitude = motion_noise_magnitude
        
        self.moveGripperXY(obj.center)
        if plot_steps: 
            self.plot(show_gripper = show_gripper, show_trail = show_trail)
            sleep(delay)
        self.moveGripperDown()
        if plot_steps: 
            self.plot(show_gripper = show_gripper, show_trail = show_trail)
            sleep(delay)
        self.closeGripper()
        if plot_steps: 
            self.plot(show_gripper = show_gripper, show_trail = show_trail)
            sleep(delay)
        self.moveGripperUp()
        if plot_steps: 
            self.plot(show_gripper = show_gripper, show_trail = show_trail)
            sleep(delay)
        self.moveGripperXY(target[:-1])
        if plot_steps: 
            self.plot(show_gripper = show_gripper, show_trail = show_trail)
            sleep(delay)
        self.moveGripperXYZ(target[:-1], target[-1])
        if plot_steps: 
            self.plot(show_gripper = show_gripper, show_trail = show_trail)
            sleep(delay)
        self.openGripper()
        if plot_steps: 
            self.plot(show_gripper = show_gripper, show_trail = show_trail)
            sleep(delay)
        self.moveGripperUp()
        if plot_steps: 
            self.plot(show_gripper = show_gripper, show_trail = show_trail)
            sleep(delay)
        # self.resetGripperPos()
    
    def addNoise(self, pos):
        pos = np.array(pos)
        noise = (rand(*pos.shape) - 0.5) * 2 * self.motion_noise_magnitude
        # print(noise)
        return pos + noise
    
    def getObjectPos(self, obj_idx):
        return copy(self.objects[obj_idx].pos())
            
    def moveGripper(self, 
                    new_x = None, 
                    new_y = None, 
                    new_z = None, 
                    new_gripper_dist = None, 
                    plot = False):
        target_x = new_x if new_x is not None else self.gripper.x_pos
        target_y = new_y if new_y is not None else self.gripper.y_pos
        target_z = new_z if new_z is not None else self.gripper.z_pos
        target_gripper_dist = new_gripper_dist if new_gripper_dist is not None else self.gripper.distance
        
        self.gripper.moveCenter((target_x, target_y), target_z)
        self.gripper.distance = target_gripper_dist
        self.checkObject(threshold = self.gripper_threshold)
        if self.gripped_obj != None:
            self.objects[self.gripped_obj].moveCenter((target_x, target_y), target_z)
        # print(target_x, target_y, target_z, target_gripper_dist)
        self.noteGrippperHistory()
        
        if plot: 
            self.plot(show_gripper = self.show_gripper, show_trail = self.show_trail)
            sleep(self.plot_delay)
    
    def moveObjectByGripper2(self, obj_idx, target, plot_steps = False, delay = 0.1, gripper_threshold = 1, show_gripper = True, show_trail = False, motion_noise_magnitude = None):
        self.motion_noise_magnitude = motion_noise_magnitude if motion_noise_magnitude is not None else 0
        self.show_gripper = show_gripper
        self.show_trail = show_trail
        self.plot_delay = delay
        self.gripper_threshold = gripper_threshold
        
        # Move to object pos
        cur_target_pos = self.getObjectPos(obj_idx)
        cur_target_pos[2] = copy(self.gripper.z_pos)
        cur_target_pos = self.addNoise(cur_target_pos)
        self.moveGripper(new_x = cur_target_pos[0],
                         new_y = cur_target_pos[1],
                         new_z = cur_target_pos[2],
                         plot = plot_steps)
        
        # Move down
        cur_target_pos = self.getObjectPos(obj_idx)
        cur_target_pos = self.addNoise(cur_target_pos)
        cur_target_pos[2] = 0
        self.moveGripper(new_x = cur_target_pos[0],
                         new_y = cur_target_pos[1],
                         new_z = cur_target_pos[2],
                         plot = plot_steps)
        
        # Close gripper
        self.moveGripper(new_gripper_dist = 0,
                         plot = plot_steps)
        
        # Move up
        cur_target_pos = self.getObjectPos(obj_idx)
        cur_target_pos[2] = copy(self.gripper.default_pos[2])
        cur_target_pos = self.addNoise(cur_target_pos)
        self.moveGripper(new_x = cur_target_pos[0],
                         new_y = cur_target_pos[1],
                         new_z = cur_target_pos[2],
                         plot = plot_steps)
        
        # Move to target pos
        cur_target_pos = list(copy(target))
        cur_target_pos[2] = copy(self.gripper.default_pos[2])
        cur_target_pos = self.addNoise(cur_target_pos)
        self.moveGripper(new_x = cur_target_pos[0],
                         new_y = cur_target_pos[1],
                         new_z = cur_target_pos[2],
                         plot = plot_steps)
        
        # Move down
        cur_target_pos = list(copy(target))
        cur_target_pos = self.addNoise(cur_target_pos)
        cur_target_pos[2] = target[2]
        self.moveGripper(new_x = cur_target_pos[0],
                         new_y = cur_target_pos[1],
                         new_z = cur_target_pos[2],
                         plot = plot_steps)
        
        # Open gripper
        self.moveGripper(new_gripper_dist = self.gripper.gripper_max, 
                         plot = plot_steps)
        
        # Move up
        cur_target_pos = list(copy(target))
        cur_target_pos[2] = copy(self.gripper.default_pos[2])
        cur_target_pos = self.addNoise(cur_target_pos)
        self.moveGripper(new_x = cur_target_pos[0],
                         new_y = cur_target_pos[1],
                         new_z = cur_target_pos[2],
                         plot = plot_steps)
        
    def plotTracks(self,):
        plt.xlim(0, self.x_max)
        plt.ylim(0, self.y_max)
        for i, hist in enumerate(self.gripper_history[:-1]):
            plt.plot([self.gripper_history[i]['pos'][0], self.gripper_history[i + 1]['pos'][0]], [self.gripper_history[i]['pos'][1], self.gripper_history[i + 1]['pos'][1]])
        
    def noteGrippperHistory(self):
        note = [self.gripper.x_pos, 
                self.gripper.y_pos, 
                self.gripper.z_pos, 
                self.gripper.distance]
        self.gripper_history.append(note)

class PickAndPlaceGenerator:
    def __init__(self, 
                 img_dir, 
                 pkl_dir, 
                 num_objects, 
                 map_dim = (100, 100, 100), 
                 block_dim = (6, 6, 6),
                 gripper_x_size = 6, 
                 gripper_threshold = 1,
                 border_padding = 10,
                 img_size = (100, 100),
                 randomize_block_pos = True,
                 permute_block_pos = False,
                 randomize_target_pos = True,
                 pos_noise_magnitude = None,
                 motion_noise_magnitude = None):
        self.num_objects = [num_objects] if type(num_objects) == int else num_objects
        self.num_data = None
        self.data_name = 'stacking_{}'.format(self.num_objects)            
        self.max_segments = (max(self.num_objects) * 8)
        self.img_dir = join(img_dir, self.data_name)
        self.pkl_path = join(pkl_dir, self.data_name)
        self.img_size = img_size
        self.map_dim = map_dim
        self.block_dim = block_dim
        self.border_padding = border_padding
        self.gripper_x_size = gripper_x_size
        self.gripper_threshold = gripper_threshold
        self.colors = ['r', 'y', 'b', 'g', 'm', 'c', 'k', 'gray']
        self.randomize_block_pos = randomize_block_pos
        self.permute_block_pos = permute_block_pos
        self.randomize_target_pos = randomize_target_pos
        self.pos_noise_magnitude = pos_noise_magnitude
        self.motion_noise_magnitude = motion_noise_magnitude
        
        np.random.seed(datetime.now().microsecond)
            
    def generatePermutations(self):
        if self.permuteCondition():
            self.obj_permutations = list(itertools.permutations(range(self.cur_num_object)))
            if self.num_data != None: self.permutation_portion = np.ceil(self.num_data / len(self.obj_permutations))
            self.permutation_idx = 0
        
    def permuteCondition(self):
        if self.cur_num_object > 1 and self.permute_block_pos:
            return True
        else:
            return False
      
    def initializeMap(self):
        self.map = Map(self.map_dim)
        self.generateInitialPositions(randomize_block_pos = self.randomize_block_pos)
        self.generateTargetPosition(randomize_target_pos = self.randomize_target_pos)
        for i in range(self.cur_num_object):
            self.map.addObject(Block(dim = self.block_dim, 
                                     color = self.colors[i], 
                                     initial_position = (self.init_pos[i, 0], self.init_pos[i, 1], 0)))
        self.map.addGripper(Gripper(initial_position = (self.map_dim[0] / 2, self.map_dim[1] / 2, self.map_dim[2] / 2), 
                                    gripper_x_size = self.gripper_x_size))
        
    def permuteBlocks(self):
        all_pos = [[i.x_pos, i.y_pos, i.z_pos] for i in self.map.objects]
        current_combination = self.obj_permutations[self.permutation_idx]

        for i in range(len(all_pos)):
            self.map.objects[i].x_pos = all_pos[current_combination[i]][0]
            self.map.objects[i].y_pos = all_pos[current_combination[i]][1]
            self.map.objects[i].z_pos = all_pos[current_combination[i]][2]
            self.map.objects[i].reCenter()
        
    def generate(self, num_data, base_bf = 2, base_dt = 0.001, print_every = 50):
        self.num_data = int(num_data)
        self.cur_num_object = max(self.num_objects)
        
        if self.permuteCondition():
            self.total_to_generate = 0
            for i in self.num_objects:
                self.total_to_generate += math.factorial(i) * self.num_data
        else:
            self.total_to_generate = len(self.num_objects) * self.num_data
        
        print('\nGenerating {} data\nNum objects = {}\nPermutation = {}\n'.format(self.total_to_generate, self.num_objects, self.permute_block_pos))
        
        self.img_dir += '[num-data-{}][block_permute-{}_random-pos-{}][target_random-pos-{}]'.format(self.total_to_generate, self.permute_block_pos, self.randomize_block_pos, self.randomize_target_pos)
        self.generation_time = datetime.now().strftime('[%Y-%m-%d_%H-%M-%S]')
        self.img_dir += self.generation_time
        if not isdir(self.img_dir): makedirs(self.img_dir)
        
        self.base_bf = base_bf
        self.base_dt = base_dt
        
        self.pkl_data = {'image': [],
                         'image_dim': (3, self.img_size[0], self.img_size[1]),
                         'original_trajectory': [],
                         'normal_dmp_seg_num': np.ones(self.total_to_generate).reshape(-1, 1),
                         'normal_dmp_dt': self.base_dt,
                         'normal_dmp_y0': [],
                         'normal_dmp_goal': [],
                         'normal_dmp_w': [],
                         'normal_dmp_tau': [],
                         'normal_dmp_bf': self.max_segments * self.base_bf,
                         'normal_dmp_ay': 25,
                         'normal_dmp_trajectory': [],
                         'normal_dmp_L_y0': [],
                         'normal_dmp_L_goal': [],
                         'normal_dmp_L_w': [],
                         'normal_dmp_L_tau': [],
                         'normal_dmp_L_bf': 1000,
                         'normal_dmp_L_ay': 200,
                         'normal_dmp_L_trajectory': [],
                         'segmented_dmp_max_observable_pos':max(self.num_objects) + 1,
                         'segmented_dmp_observable_pos': [],
                         'segmented_dmp_max_seg_num': self.max_segments,
                         'segmented_dmp_seg_num': [],
                         'segmented_dmp_y0': [],
                         'segmented_dmp_goal': [],
                         'segmented_dmp_w': [],
                         'segmented_dmp_tau': [],
                         'segmented_dmp_dt': self.base_dt * self.max_segments,
                         'segmented_dmp_bf': self.base_bf,
                         'segmented_dmp_ay': 3.4,
                         'segmented_dmp_trajectory': []
                       }
        
        for i, num_obj in enumerate(self.num_objects):
            self.cur_num_object = num_obj
            self.generatePermutations()
            if not self.permuteCondition():
                for j in range(self.num_data):
                    if (len(self.pkl_data['image']) + 1) % print_every == 0 or (len(self.pkl_data['image']) + 1) % self.total_to_generate == 0:
                        print("Generating {}/{}".format(len(self.pkl_data['image']) + 1, self.total_to_generate))
                        
                    self.img_path = join(self.img_dir, 'obj_{}_{}.jpg'.format(num_obj, '0'*(len(str(self.num_data)) - len(str(j + 1))) + str(j + 1)))
                    
                    self.initializeMap()
                    self.insertPos()
                    self.map.plot(hide_borders = True, 
                                  save_path = self.img_path, 
                                  save_size = self.img_size, 
                                  show_plot = False)
                    self.runMap()
                    self.insertData()
            else:
                for j in range(len(self.obj_permutations)):
                    self.permutation_idx = j
                    # print(self.obj_permutations[j])
                    for k in range(self.num_data):
                        if (len(self.pkl_data['image']) + 1) % print_every == 0 or (len(self.pkl_data['image']) + 1) % self.total_to_generate == 0:
                            print("Generating {}/{}".format(len(self.pkl_data['image']) + 1, self.total_to_generate))
                            
                        self.img_path = join(self.img_dir, 'obj_{}_{}_{}.jpg'.format(num_obj, j, '0'*(len(str(self.num_data)) - len(str(k + 1))) + str(k + 1)))
                        
                        self.initializeMap()
                        self.permuteBlocks()
                        self.insertPos()
                        self.map.plot(hide_borders = True, 
                                      save_path = self.img_path, 
                                      save_size = self.img_size, 
                                      show_plot = False)
                        self.runMap()
                        self.insertData()
                        
        self.generatePkl()
        
    def insertPos(self):
        observable_pos = []
        total_object = len(self.map.objects)
        for i in range(max(self.num_objects)):
            if i < total_object:
                observable_pos.append([self.map.objects[i].x_pos, self.map.objects[i].y_pos])
            else:
                observable_pos.append([0.0, 0.0])
        observable_pos.append([self.map.target[0], self.map.target[1]])
        self.pkl_data['segmented_dmp_observable_pos'].append(observable_pos)
        
    def generateTestData(self, num_object = None):
        if num_object is not None: 
            num_object = max(self.num_objects) if num_object > max(self.num_objects) else num_object
        if not isdir(self.img_dir): 
            makedirs(self.img_dir)
        self.img_path = join(self.img_dir, 'test.jpg')
        self.cur_num_object = random.choice(self.num_objects) if num_object == None else num_object
        self.initializeMap()
        if self.permuteCondition(): 
            self.generatePermutations()
            self.permutation_idx = randint(0, self.cur_num_object)
            self.permuteBlocks()
        self.map.plot(hide_borders = True,
                      save_path = self.img_path, 
                      save_size = self.img_size, 
                      show_plot = True)
        img = np.array(Image.open(self.img_path))
        
        old_map = deepcopy(self.map)
        self.runMap(plot = False)
        gripper_history = self.map.gripper_history
        ys = []
        for i in range(len(gripper_history) - 1):
            seg = gripper_history[i:i+2]
            seg = np.array(seg)
            dmp = DMPs_discrete(n_dmps = 4,
                                n_bfs = 2,
                                ay = np.ones(4) * 3.4,
                                dt = 0.024)
            dmp.imitate_path(seg.T)
            y, _, _ = dmp.rollout()
            ys.append(y)
        y_label = ys[0]
        for i in ys[1:]:
            y_label = np.append(y_label, i, axis = 0)
        self.map = old_map
        return img, y_label
    
    def insertData(self):
        self.map.gripper_history = np.array(self.map.gripper_history)
        traj = self.map.gripper_history
        img = np.array(Image.open(self.img_path))
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        self.pkl_data['image'].append(img)
        self.pkl_data['original_trajectory'].append(traj)
        
        normal_dmp = DMPs_discrete(n_dmps = traj.shape[1], 
                                   n_bfs = self.pkl_data['normal_dmp_bf'], 
                                   dt = self.pkl_data['normal_dmp_dt'],
                                   ay = np.ones(traj.shape[1]) * self.pkl_data['normal_dmp_ay'])
        normal_dmp.imitate_path(traj.T)
        normal_y, _, _ = normal_dmp.rollout()
        self.pkl_data['normal_dmp_y0'].append(normal_dmp.y0)
        self.pkl_data['normal_dmp_goal'].append(normal_dmp.goal)
        self.pkl_data['normal_dmp_w'].append(normal_dmp.w)
        self.pkl_data['normal_dmp_tau'].append(1)
        self.pkl_data['normal_dmp_trajectory'].append(normal_y)
        
        normal_dmp_L = DMPs_discrete(n_dmps = traj.shape[1], 
                                   n_bfs = self.pkl_data['normal_dmp_L_bf'], 
                                   dt = self.pkl_data['normal_dmp_dt'],
                                   ay = np.ones(traj.shape[1]) * self.pkl_data['normal_dmp_L_ay'])
        normal_dmp_L.imitate_path(traj.T)
        normal_L_y, _, _ = normal_dmp_L.rollout()
        self.pkl_data['normal_dmp_L_y0'].append(normal_dmp_L.y0)
        self.pkl_data['normal_dmp_L_goal'].append(normal_dmp_L.goal)
        self.pkl_data['normal_dmp_L_w'].append(normal_dmp_L.w)
        self.pkl_data['normal_dmp_L_tau'].append(1)
        self.pkl_data['normal_dmp_L_trajectory'].append(normal_L_y)
        
        segmented_dmp_y0s = []
        segmented_dmp_goals = []
        segmented_dmp_ws = []
        segmented_dmp_taus = []
        segmented_dmp_trajectories = []
        for i in range(traj.shape[0] - 1):
            sub_traj = traj[i:i + 2]
            segmented_dmp = DMPs_discrete(n_dmps = sub_traj.shape[1], 
                                          n_bfs = self.pkl_data['segmented_dmp_bf'], 
                                          dt = self.pkl_data['segmented_dmp_dt'],
                                          ay = np.ones(sub_traj.shape[1]) * self.pkl_data['segmented_dmp_ay'])
            segmented_dmp.imitate_path(sub_traj.T)
            segmented_y, _, _ = segmented_dmp.rollout()
            segmented_dmp_y0s.append(segmented_dmp.y0)
            segmented_dmp_goals.append(segmented_dmp.goal)
            segmented_dmp_ws.append(segmented_dmp.w)
            segmented_dmp_taus.append(1)
            segmented_dmp_trajectories.append(segmented_y)
        segmented_dmp_trajectory = segmented_dmp_trajectories[0]
        for i in segmented_dmp_trajectories[1:]:
            segmented_dmp_trajectory = np.append(segmented_dmp_trajectory, i, axis = 0)
        self.pkl_data['segmented_dmp_seg_num'].append(len(segmented_dmp_trajectories))
        self.pkl_data['segmented_dmp_y0'].append(segmented_dmp_y0s)
        self.pkl_data['segmented_dmp_goal'].append(segmented_dmp_goals)
        self.pkl_data['segmented_dmp_w'].append(segmented_dmp_ws)
        self.pkl_data['segmented_dmp_tau'].append(segmented_dmp_taus)
        self.pkl_data['segmented_dmp_trajectory'].append(segmented_dmp_trajectory)
        
    def padData(self):
        to_process = deepcopy(self.pkl_data)

        unique_lengths = []
        for i in to_process['segmented_dmp_w']:
            if len(i) not in unique_lengths:
                unique_lengths.append(len(i))
        unique_lengths = sorted(unique_lengths)
        unique_lengths = [i for i in range(1, unique_lengths[0])] + unique_lengths

        idx_segments = {'y0': [[] for i in range(unique_lengths[-1])],
                        'goal': [[] for i in range(unique_lengths[-1])],
                        'w': [[] for i in range(unique_lengths[-1])],
                        'tau': [[] for i in range(unique_lengths[-1])]}

        for i in range(len(to_process['segmented_dmp_y0'])):
            for seg in range(len(to_process['segmented_dmp_y0'][i])):
                idx_segments['y0'][seg].append(to_process['segmented_dmp_y0'][i][seg])
                idx_segments['goal'][seg].append(to_process['segmented_dmp_goal'][i][seg])
                idx_segments['w'][seg].append(to_process['segmented_dmp_w'][i][seg])
                idx_segments['tau'][seg].append(to_process['segmented_dmp_tau'][i][seg])
        
        idx_segments['y0'] = [np.array(i) for i in idx_segments['y0']]
        idx_segments['goal'] = [np.array(i) for i in idx_segments['goal']]
        idx_segments['w'] = [np.array(i) for i in idx_segments['w']]
        idx_segments['tau'] = [np.array(i) for i in idx_segments['tau']]


        pads = idx_segments

        for i in range(len(to_process['segmented_dmp_y0'])):
            if len(to_process['segmented_dmp_y0'][i]) < unique_lengths[-1]:
                while len(to_process['segmented_dmp_y0'][i]) < unique_lengths[-1]:
                    to_process['segmented_dmp_y0'][i].append(pads['y0'][len(to_process['segmented_dmp_y0'][i])].mean(axis = 0))
                    to_process['segmented_dmp_goal'][i].append(pads['goal'][len(to_process['segmented_dmp_goal'][i])].mean(axis = 0))
                    to_process['segmented_dmp_w'][i].append(pads['w'][len(to_process['segmented_dmp_w'][i])].mean(axis = 0))
                    to_process['segmented_dmp_tau'][i].append(pads['tau'][len(to_process['segmented_dmp_tau'][i])].mean(axis = 0))
        self.pkl_data = to_process
        
        data_len = len(self.pkl_data['image'])
        dof = self.pkl_data['original_trajectory'][0].shape[1]

        self.pkl_data['image']                   = np.array(self.pkl_data['image'])
        self.pkl_data['normal_dmp_y0']           = np.array(self.pkl_data['normal_dmp_y0']).reshape(data_len, dof)
        self.pkl_data['normal_dmp_goal']         = np.array(self.pkl_data['normal_dmp_goal']).reshape(data_len, dof)
        self.pkl_data['normal_dmp_w']            = np.array(self.pkl_data['normal_dmp_w']).reshape(data_len, dof, self.pkl_data['normal_dmp_bf'])
        self.pkl_data['normal_dmp_tau']          = np.array(self.pkl_data['normal_dmp_tau']).reshape(-1, 1)
        self.pkl_data['normal_dmp_trajectory']   = np.array(self.pkl_data['normal_dmp_trajectory'])
        self.pkl_data['normal_dmp_L_y0']           = np.array(self.pkl_data['normal_dmp_L_y0']).reshape(data_len, dof)
        self.pkl_data['normal_dmp_L_goal']         = np.array(self.pkl_data['normal_dmp_L_goal']).reshape(data_len, dof)
        self.pkl_data['normal_dmp_L_w']            = np.array(self.pkl_data['normal_dmp_L_w']).reshape(data_len, dof, self.pkl_data['normal_dmp_L_bf'])
        self.pkl_data['normal_dmp_L_tau']          = np.array(self.pkl_data['normal_dmp_L_tau']).reshape(-1, 1)
        self.pkl_data['normal_dmp_L_trajectory']   = np.array(self.pkl_data['normal_dmp_L_trajectory'])
        self.pkl_data['segmented_dmp_observable_pos'] = np.array(self.pkl_data['segmented_dmp_observable_pos'])
        self.pkl_data['segmented_dmp_seg_num']   = np.array(self.pkl_data['segmented_dmp_seg_num']).reshape(-1, 1)
        self.pkl_data['segmented_dmp_goal']      = np.array(self.pkl_data['segmented_dmp_goal'])
        self.pkl_data['segmented_dmp_tau']       = np.array(self.pkl_data['segmented_dmp_tau'])
        self.pkl_data['segmented_dmp_w']         = np.array(self.pkl_data['segmented_dmp_w'])
        self.pkl_data['segmented_dmp_y0']        = np.array(self.pkl_data['segmented_dmp_y0'])
        if 'normal_dmp_target_trajectory' in self.pkl_data: self.pkl_data['normal_dmp_target_trajectory']   = np.array(self.pkl_data['normal_dmp_target_trajectory'])
        if 'segmented_dmp_target_trajectory' in self.pkl_data:  self.pkl_data['segmented_dmp_target_trajectory']        = np.array(self.pkl_data['segmented_dmp_target_trajectory'])
        if 'rotation_degrees' in self.pkl_data:  self.pkl_data['rotation_degrees'] = np.array(self.pkl_data['rotation_degrees'])
    
    def generatePkl(self):
        self.padData()
        
        PKL_NAME = self.pkl_path
        PKL_NAME += '[num-data-{}][max-seg-{}]'.format(self.total_to_generate,
                                                       self.pkl_data['segmented_dmp_max_seg_num'])
        PKL_NAME += '[normal-dmp_bf-{}_ay-{}_dt-{}]'.format(self.pkl_data['normal_dmp_bf'],
                                                            self.pkl_data['normal_dmp_ay'],
                                                            self.pkl_data['normal_dmp_dt'])
        PKL_NAME += '[seg-dmp_bf-{}_ay-{}_dt-{}]'.format(self.pkl_data['segmented_dmp_bf'],
                                                         self.pkl_data['segmented_dmp_ay'],
                                                         self.pkl_data['segmented_dmp_dt'])
        PKL_NAME += '[block_permute-{}_random-pos-{}][target_random-pos-{}]'.format(self.permute_block_pos, self.randomize_block_pos, self.randomize_target_pos)
        PKL_NAME += self.generation_time
        PKL_NAME += '.pkl'
        
        pkl.dump(self.pkl_data, open(PKL_NAME, 'wb'))
        print('\nImage directory:\n{}\n'.format(self.img_dir))
        print('pkl directory:\n{}\n'.format('/'.join(self.pkl_path.split('/')[:-1])))
        print('pkl name:\n{}'.format(PKL_NAME.split('/')[-1]))
    
    def runMap(self, plot = False):
        cur_height = 0
        for i in range(self.cur_num_object):
            self.map.moveObjectByGripper2(obj_idx = i,
                                    target = (self.map.target[0], self.map.target[1], cur_height),
                                    gripper_threshold = self.gripper_threshold,
                                    motion_noise_magnitude = self.motion_noise_magnitude,
                                    plot_steps = plot)
            cur_height += self.block_dim[2]
    
    def generateInitialPositions(self, randomize_block_pos = True, max_loop = 100):
        self.init_pos = []
        if randomize_block_pos:
            loop = 0
            while len(self.init_pos) < self.cur_num_object:
                pos_x = self.border_padding + (rand() * ((self.map_dim[0] / 2) - np.array([self.block_dim[0], self.gripper_x_size]).min()))
                pos_y = rand() * (self.map_dim[1] - self.block_dim[1])
                if len(self.init_pos) == 0:
                    # print([pos_x, pos_y])
                    self.init_pos.append([pos_x, pos_y])
                else:
                    hit = False
                    for pos in self.init_pos:
                        # print([pos_x, pos_y], [pos[0], pos[1]])
                        if not ((pos[0] > (pos_x + self.block_dim[0] + self.border_padding) or pos_x > (pos[0] + self.block_dim[0] + self.border_padding)) or \
                           (pos[1] > (pos_y + self.block_dim[1] + self.border_padding) or pos_y > (pos[1] + self.block_dim[1] + self.border_padding))):
                            hit = True
                    if not hit: self.init_pos.append([pos_x, pos_y])
                loop += 1
                if loop >= max_loop: 
                    self.init_pos = []
                    loop = 0
        else:
            pos_y = np.linspace(0, self.map_dim[1], max(self.num_objects) + 2)[1:-1]
            for i in range(self.cur_num_object):
                self.init_pos.append([self.map_dim[0] * 0.2, pos_y[i] - (self.block_dim[1] / 2)])
        self.init_pos = np.array(self.init_pos)
        if self.pos_noise_magnitude != None:
            self.init_pos = self.init_pos + ((rand(*self.init_pos.shape) - 0.5) * 2 * self.pos_noise_magnitude)
    
    def generateTargetPosition(self, randomize_target_pos = True):
        if randomize_target_pos:
            target = rand(2)
            target[0] = (self.map_dim[0] / 2) + self.border_padding + (((self.map_dim[0] / 2) - self.block_dim[0] - self.border_padding) * target[0])
            target[1] = (self.block_dim[1] / 2) + ((self.map_dim[1] - self.block_dim[1]) * target[1])
        else:
            target = np.array([0.8 * self.map_dim[0], 0.5 * self.map_dim[1]])
        if self.pos_noise_magnitude != None:
            target = target + ((rand(*target.shape) - 0.5) * 2 * self.pos_noise_magnitude)
        self.map.addTarget(target, radius = 5)
"""#%%
if __name__=='__main__':
    plot_steps = 1
    plot_delay = 0.2
    show_gripper = 1
    show_trail = 1
    
    map = Map((100, 100, 100))
    
    box_w = 6
    box_l = 6
    box_h = 6
    gripper_x_size = 6
    border = 3
    
    init_pos = []
    while len(init_pos) < 3:
        pos_x = border + (rand() * (50 - np.array([box_w, gripper_x_size]).min()))
        pos_y = rand() * (100 - 15)
        if len(init_pos) == 0:
            # print([pos_x, pos_y])
            init_pos.append([pos_x, pos_y])
        else:
            hit = False
            for pos in init_pos:
                # print([pos_x, pos_y], [pos[0], pos[1]])
                if not ((pos[0] > (pos_x + box_w + border) or pos_x > (pos[0] + box_w + border)) or \
                    (pos[1] > (pos_y + box_l + border) or pos_y > (pos[1] + box_l + border))):
                    hit = True
            if not hit: init_pos.append([pos_x, pos_y])

    init_pos = np.array(init_pos)
    # init_pos = rand(3, 2)
    # init_pos[:, 0] *= (50 - box_w) 
    # init_pos[:, 1] *= (100 - box_l)
    map.addObject(Block((box_w, box_l, box_h), 'r', (init_pos[0, 0], init_pos[0, 1], 0)))
    map.addObject(Block((box_w, box_l, box_h), 'y', (init_pos[1, 0], init_pos[1, 1], 0)))
    map.addObject(Block((box_w, box_l, box_h), 'b', (init_pos[2, 0], init_pos[2, 1], 0)))
    map.addGripper(Gripper((50, 50, 50), gripper_x_size))
    
    target = rand(2)
    target[0] = 50 + border + ((50 - box_w - border) * target[0])
    target[1] = (box_l / 2) + ((100 - box_l) * target[1])
    map.addTarget(target, radius = 2)
    
    map.plot(show_gripper = show_gripper, show_trail = show_trail)
    sleep(plot_delay)
    
    map.moveObjectByGripper(0, (target[0], target[1], 0), plot_steps = plot_steps, delay = plot_delay, show_gripper = show_gripper, show_trail = show_trail)
    map.moveObjectByGripper(1, (target[0], target[1], 6), plot_steps = plot_steps, delay = plot_delay, show_gripper = show_gripper, show_trail = show_trail)
    map.moveObjectByGripper(2, (target[0], target[1], 12), plot_steps = plot_steps, delay = plot_delay, show_gripper = show_gripper, show_trail = show_trail)
    gripper_history = map.gripper_history
    if not plot_steps: map.plot(show_gripper = show_gripper, show_trail = show_trail)
    
    map.plot_side()"""
    
#%%
if __name__=='__main__':
    num_objects = [3]
    # block_permute = True
    block_permute = False
    # block_random_pos = True
    block_random_pos = False
    # target_random_pos = True
    target_random_pos = False
    gripper_threshold = 3
    DATA_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data'
    IMG_DIR = join(DATA_DIR, 'images/stacking/generate_test')
    PKL_DIR = join(DATA_DIR, 'pkl/stacking/generate_test')
    
    
    generator = PickAndPlaceGenerator(img_dir = IMG_DIR,
                                      pkl_dir = PKL_DIR,
                                      num_objects = num_objects,
                                      img_size = (150, 150),
                                      gripper_threshold = gripper_threshold,
                                      randomize_block_pos = block_random_pos,
                                      permute_block_pos = block_permute,
                                      randomize_target_pos = target_random_pos,
                                      # pos_noise_magnitude = 3,
                                      pos_noise_magnitude = 0,
                                      # motion_noise_magnitude = 2,
                                      motion_noise_magnitude = 0,
                                      )
    generator.generateTestData()
    generator.map.plot(show_gripper = True)
    generator.runMap(plot = True)
    generator.map.plot_side()
#%%
if __name__=='__main__':
    num_data = 500
    num_objects = [1, 2, 3]
    # block_permute = True
    block_permute = False
    block_random_pos = True
    # block_random_pos = False
    target_random_pos = True
    # target_random_pos = False
    IMG_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/images/stacking'
    PKL_DIR = '/home/edgar/rllab/scripts/dmp/SegmentedDeepDMPs/data/pkl/stacking'
    
    generator = PickAndPlaceGenerator(img_dir = IMG_DIR,
                                      pkl_dir = PKL_DIR,
                                      num_objects = num_objects,
                                      img_size = (150, 150),
                                      gripper_threshold = 3,
                                      randomize_block_pos = block_random_pos,
                                      permute_block_pos = block_permute,
                                      randomize_target_pos = target_random_pos,
                                      # pos_noise_magnitude = 3,
                                      pos_noise_magnitude = 0,
                                      # motion_noise_magnitude = 2,
                                      motion_noise_magnitude = 0,
                                      )
    generator.generate(num_data, print_every = 20)