from typing import List
from pydmps.dmp_discrete import DMPs_discrete
import numpy as np
from matplotlib import pyplot as plt
# from data_generator import subDivideTraj
from utils.data_generator.cutting_trajectory_generator import subDivideTraj
from os.path import join
from PIL import Image
from multiprocessing import Process

class SegmentTrajectoryGenerator:
    def __init__(self, shape_templates : List[List[List]] = None, segment_types : List[List] = None, dict_dmp_bf = 5, dict_dmp_ay = 4, dict_dmp_dt = 0.05, subdiv_traj_length = 500):
        self.generateDMPDictionary(dict_dmp_bf, dict_dmp_ay, dict_dmp_dt)
        
        if shape_templates != None and segment_types == None:
            segment_types = []
            for shape in shape_templates:
                shape_segment_types = []
                for i in range(len(shape)-1):
                    shape_segment_types.append(0)
                segment_types.append(shape_segment_types)
                
        self.shape_templates    = shape_templates
        self.segment_types      = segment_types
        
        self.segment_types_padded = []
        for segment_type in segment_types:
            segment_type_padded = segment_type[:]
            last = segment_type[-1]
            while len(segment_type_padded) < max([len(i) for i in self.segment_types]):
                segment_type_padded.append(last)
            self.segment_types_padded.append(segment_type_padded)
        self.segment_types_padded = np.array(self.segment_types_padded)
        
        self.dict_dmp_bf        = dict_dmp_bf
        self.dict_dmp_ay        = dict_dmp_ay
        self.dict_dmp_dt        = dict_dmp_dt
        self.subdiv_traj_length = subdiv_traj_length
        
        if self.shape_templates != None:
            self.base_shapes = []
            self.subdivided_base_shapes = []
            self.base_num_segments = []
            for shape in range(len(self.shape_templates)):
                shape, num_segments = self.parseTrajectory(self.shape_templates[shape], self.segment_types[shape])
                self.base_shapes.append(shape)
                self.base_num_segments.append(num_segments)
            self.padded_base_shapes = self.padTrajectory(self.base_shapes)
            for shape in self.base_shapes:
                self.subdivided_base_shapes.append(subDivideTraj(shape, self.subdiv_traj_length))
            self.subdivided_base_shapes = np.array(self.subdivided_base_shapes)
        # if self.shape_templates != None: self.parseTrajectory()
    
    def parseTrajectory(self, shape_points, shape_segments):
        num_segments = len(shape_segments)
        for point in range(len(shape_points) - 1):
            p_start = shape_points[point]
            p_end = shape_points[point + 1]
            mul_x = p_end[0] - p_start[0]
            mul_y = p_end[1] - p_start[1]
            segment = np.copy(self.traj_dict[shape_segments[point]]['dmp_traj'])
            segment[:,0] = segment[:,0] * (mul_x if shape_segments[point] != 4 else 1)
            segment[:,1] = segment[:,1] * (mul_y if shape_segments[point] != 3 else 1)
            segment[:,0] = segment[:,0] + p_start[0]
            segment[:,1] = segment[:,1] + p_start[1]
            
            if point == 0:
                traj = segment
            else:
                traj = np.append(traj, segment, axis = 0)
        return traj, num_segments
    
    def padTrajectory(self, trajs):
        max_length = max([i.shape[0] for i in trajs])
        padded_shapes = []
        for traj in trajs:
            pad = np.tile(traj[-1,:].reshape(1,-1), (max_length - traj.shape[0], 1))
            padded_shapes.append(np.append(traj, pad, axis = 0))
        return np.array(padded_shapes)

    def generateDMPDictionary(self, dict_dmp_bf, dict_dmp_ay, dict_dmp_dt):
        dict_trajectories = [
                             [[0.0, 0.0],
                              [1.0, 1.0]], # 0: Straight line
                             [[0.0, 0.0],
                              [1.0, 0.0],
                              [1.0, 1.0]], # 1: Diagonal curve bottom
                             [[0.0, 0.0],
                              [0.0, 1.0],
                              [1.0, 1.0]], # 2: Diagonal curve top
                             [[0.0, 0.0],
                              [0.0, 1.0],
                              [1.0, 1.0],
                              [1.0, 0.0]], # 3: Curve horizontal (Cannot move vertical)
                             [[0.0, 0.0],
                              [1.0, 0.0],
                              [1.0, 1.0],
                              [0.0, 1.0]], # 4: Curve vertical (Cannot move horizontal)
                            ]

        traj_dict = []
        for dict_trajectory in dict_trajectories:
            traj = np.array(dict_trajectory)
            dmp = DMPs_discrete(n_dmps = traj.shape[-1],
                                n_bfs=dict_dmp_bf,
                                ay = np.ones(traj.shape[-1]) * dict_dmp_ay,
                                dt = dict_dmp_dt)
            dmp.imitate_path(traj.T)
            y, _, _ = dmp.rollout()
            traj_dict.append((traj, y, dmp))
        
        self.segment_traj_length = int(1 / dict_dmp_dt)
        self.traj_dict = np.array(traj_dict, dtype = [('traj',object),('dmp_traj',object),('dmp',DMPs_discrete)])

    def plotDictionaryTrajectory(self, traj_type = 'dmp'):
        if traj_type == 'dmp':
            for i in range(len(self.traj_dict)):
                plt.plot(self.traj_dict[i]['dmp_traj'][:,0], self.traj_dict[i]['dmp_traj'][:,1], c='r')
                plt.scatter(self.traj_dict[i]['dmp_traj'][:,0], self.traj_dict[i]['dmp_traj'][:,1], c='r')
        elif traj_type == 'original':
            for i in range(len(self.traj_dict)):
                plt.plot(self.traj_dict[i]['traj'][:,0], self.traj_dict[i]['traj'][:,1], c='r')
                plt.scatter(self.traj_dict[i]['traj'][:,0], self.traj_dict[i]['traj'][:,1], c='r')

    def plotShapeTemplates(self, fig_size = 6):
        fig, axs = plt.subplots(1, len(self.base_shapes), figsize=(fig_size*len(self.base_shapes), fig_size))
        for i in range(len(self.base_shapes)):
            if len(self.base_shapes) > 1:
                axs[i].plot(self.base_shapes[i][:, 0], self.base_shapes[i][:, 1], color = 'blue')
                axs[i].scatter(self.base_shapes[i][:, 0], self.base_shapes[i][:, 1], color = 'blue')
                plt.title('Shape ' + str(i + 1))
                # axs[i].axis('equal')
            else:
                axs.plot(self.base_shapes[i][:, 0], self.base_shapes[i][:, 1], color = 'blue')
                axs.scatter(self.base_shapes[i][:, 0], self.base_shapes[i][:, 1], color = 'blue')
                plt.title('Shape ' + str(i + 1))
                # axs.axis('equal')
        plt.show()
        
    def plotTrajectories(self, shapes, fig_size = 6):
        fig, axs = plt.subplots(1, len(shapes), figsize=(fig_size*len(shapes), fig_size))
        for i in range(len(shapes)):
            if len(shapes) > 1:
                axs[i].plot(shapes[i][:, 0], shapes[i][:, 1], color = 'blue')
                axs[i].scatter(shapes[i][:, 0], shapes[i][:, 1], color = 'blue')
                axs[i].set_title('Shape ' + str(i + 1))
                # axs[i].axis('equal')
                axs[i].set_xlim(-2., 8.)
                axs[i].set_ylim(-2., 8.)
            else:
                axs.plot(shapes[i][:, 0], shapes[i][:, 1], color = 'blue')
                axs.scatter(shapes[i][:, 0], shapes[i][:, 1], color = 'blue')
                axs.set_title('Shape ' + str(i + 1))
                # axs.axis('equal')
                axs.set_xlim(-2., 8.)
                axs.set_ylim(-2., 8.)
        plt.show()
        
    def plotTrajectory(self, traj, linewidth, save_path = None, target_size = None):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.xlim(-2., 8.)
        plt.ylim(-2., 8.)
        ax.plot(traj[:,0], traj[:,1], linewidth = linewidth, color = 'w', zorder = 1)
        ax.scatter(traj[0,0], traj[0,1], linewidth = linewidth*2, color = 'g', zorder = 2)
        ax.scatter(traj[-1,0], traj[-1,1], linewidth = linewidth*2, color = 'r', zorder = 2)
        plt.tick_params(
            bottom=False,     
            top=False,
            left=False,
            labelbottom=False,
            labelleft=False)
        ax.set_facecolor('black')
        if save_path == None:
            plt.show()
        else:
            plt.savefig(save_path + '.jpg', bbox_inches='tight', pad_inches=0.0)
            
            if target_size != None:
                img = Image.open(save_path + '.jpg')
                img_resize = img.resize((target_size, target_size))
                img_resize.save(save_path + '.jpg')
        
            plt.cla()
            plt.clf()
            plt.close('all')
            plt.close(fig)

    def getDMPParameters(self):
        print('DMP Parameters:')
        print('- dt   =', self.dict_dmp_dt)
        print('- N BF =', self.dict_dmp_bf)
        print('- ay   =', self.dict_dmp_ay)
        
    def getDictInfo(self):
        self.getDMPParameters()
        self.plotDictionaryTrajectory()

    def generateRandomizedShape(self, magnitude = 2e-1, return_padded = False, return_interpolated = False, plot_save_path = None, plot_prefix = None, plot_target_size = None, plot_linewidth = None):
        data = {'points': None,
                'points_padded': None,
                'dmp_traj': None,
                'dmp_traj_padded': None,
                'dmp_traj_interpolated': None,
                'segment_types': self.segment_types,
                'segment_types_padded': self.segment_types_padded,
                'segment_num': self.base_num_segments}
        randomized_dmp_traj = []
        randomized_shapes = []
        for shape in range(len(self.shape_templates)):
            randomized_shape = np.array(self.shape_templates[shape])
            range_x = randomized_shape[:,0].max() - randomized_shape[:,0].min()
            range_y = randomized_shape[:,1].max() - randomized_shape[:,1].min()
            for point in range(len(randomized_shape)):
                randomized_shape[point][0] = randomized_shape[point][0] + (((np.random.rand() * 2) - 1) * range_x * magnitude)
                randomized_shape[point][1] = randomized_shape[point][1] + (((np.random.rand() * 2) - 1) * range_y * magnitude)
            shape, num_segments = self.parseTrajectory(randomized_shape, self.segment_types[shape])
            randomized_shapes.append(np.array(randomized_shape))
            randomized_dmp_traj.append(shape)
            
        data['points'] = randomized_shapes
        data['points_padded'] = self.padTrajectory(randomized_shapes)
        data['dmp_traj'] = randomized_dmp_traj
        
        if return_padded: data['dmp_traj_padded'] = self.padTrajectory(randomized_dmp_traj)
        if return_interpolated: 
            subdivided_shape = []
            for shape in randomized_dmp_traj:
                subdivided_shape.append(subDivideTraj(shape, self.subdiv_traj_length))
            data['dmp_traj_interpolated'] = np.array(subdivided_shape)
        
        if plot_save_path != None:
            file_names = []
            for i in range(len(data['points'])):
                p = Process(target = self.plotTrajectory, args = (data['points'][i],
                                                                  plot_linewidth,
                                                                  join(plot_save_path, plot_prefix + 'shape_' + str(i)),
                                                                  plot_target_size))
                p.start()
                p.join()
                
                file_names.append(plot_prefix + 'shape_' + str(i) + '.jpg')
            data['image_names'] = file_names
            
        return data
    
    def generateRandomTrajectory(self, max_segments = None):
        pass

if __name__=='__main__':
    shapes = [[[0.0, 5.0],
               [4.0, 0.0],
               [6.0, 2.5]],
              [[1.5, 1.0],
               [3.5, 5.0],
               [5.5, 1.0],
               [1.5, 1.0]],
              [[0.0, 6.0],
               [6.0, 6.0],
               [3.0, 4.0],
               [2.5, 0.5]],
              [[2.0, 2.0],
               [2.0, 4.0],
               [4.0, 4.0],
               [4.0, 2.0],
               [2.0, 2.0]],
              [[2.5, 5.0],
               [0.0, 1.0],
               [4.5, 1.5],
               [3.5, 3.0],
               [5.25, 0.5]],
              [[1.0, 0.0],
               [2.5, 5.0],
               [4.0, 0.0],
               [0.0, 3.0],
               [5.0, 3.0],
               [1.0, 0.0]],
              [[1.0, 0.0],
               [0.0, 3.0],
               [2.5, 5.0],
               [5.0, 3.0],
               [4.0, 0.0],
               [1.0, 0.0]],
              [[1.0, 0.0],
               [1.0, 5.0],
               [2.5, 5.0],
               [2.5, 0.0],
               [2.5, 5.0],
               [4.0, 5.0],
               [4.0, 0.0]]]
    segment_types = [[1, 2],
                     [1, 2, 1],
                     [1, 2, 1],
                     [2, 1, 1, 2],
                     [2, 1, 1, 2],
                     [2, 1, 2, 1, 1],
                     [2, 1, 2, 1, 1],
                     [2, 1, 2, 1, 2, 2]]
    test = SegmentTrajectoryGenerator(shape_templates = shapes,
                                      segment_types = None,
                                      dict_dmp_bf = 5,
                                      dict_dmp_ay = 4,
                                      dict_dmp_dt = 0.05,
                                      subdiv_traj_length = 300)
    # test.plotShapeTemplates()
    # rand_shapes = test.generateRandomizedShape(magnitude = 2e-1,
    #                                             return_padded = 1, 
    #                                             return_interpolated = 1,
    #                                             plot_save_path = '.',
    #                                             plot_prefix = 'iter_0-',
    #                                             plot_target_size = 150,
    #                                             plot_linewidth = 7.5)
    
    rand_shapes = test.generateRandomizedShape(magnitude = 2e-1,
                                                return_padded = 1, 
                                                return_interpolated = 1)
    test.plotTrajectories(rand_shapes['dmp_traj'])
    
    # for i in range(len(rand_shapes)):
    #     plt.scatter(rand_shapes[i,:,0], rand_shapes[i,:,1], c='b')
    #     plt.plot(rand_shapes[i,:,0], rand_shapes[i,:,1], c='b')