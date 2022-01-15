from carrot_generator import generate_carrot, plot_carrot
from cutting_trajectory_generator import generate_cutting_trajectory, plot_trajectory
from segment_splitter import split_traj_into_segment, get_index
from dmp_utils import generate_dmps, plot_dmp_trajectory, check_w_min_max, plot_dmp_segment
from matplotlib import pyplot as plt, figure

carrot = generate_carrot(tip = 'right')
traj = generate_cutting_trajectory( obj = carrot,
                                    dist_btw_cut = 5,
                                    lift_height = 10,
                                    orientation = 0,
                                    traj_length = 100,
                                    margin = 5
                                  )
#%
fig = figure.Figure()
plot_carrot(carrot)
plot_trajectory(traj)
segments = split_traj_into_segment(traj['direct'], 20)

dmp = generate_dmps(segments, 20, 100, 0.001, segmented = True)
plot_dmp_trajectory(dmp, segmented = True)
# dmp = generate_dmps(traj['subdivided'], 20, 100, 0.001, segmented = False)
# plot_dmp_trajectory(dmp, segmented = False)

# print(check_w_min_max(dmp, segmented = True))
# plot_dmp_segment(dmp, 2)
#%%
# dpi_size = 0.33149999999999996
# plt.savefig('carrot500px.jpg', dpi=dpi_size * 500, bbox_inches='tight', pad_inches=0.0)
#%%
import numpy as np

to_plot = np.copy(carrot)
to_plot_x = carrot[:,0]
to_plot_y = carrot[:,1]
roof = []

bottom_most = to_plot_y.min()
left_most = to_plot_x.min()
right_most = to_plot_x.max()
margin = 5

plt.scatter(to_plot_x, to_plot_y, c = 'y')

for i in range(to_plot.shape[0]):
    if to_plot_x[i] > left_most + (margin / 2) and \
       to_plot_x[i] < right_most - (margin / 2) and \
       to_plot_y[i] > bottom_most + margin :
        plt.scatter(to_plot_x[i], to_plot_y[i], c = 'r')
        roof.append([to_plot_x[i], to_plot_y[i]])
        
roof_np = np.array(roof)