if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import Main
from mod.utils import Timer

sbml_file = '/home/sandy/Documents/Code/my_sim/examples/schlogl_sbmlv2.xml'
settings_file = '/home/sandy/Documents/Code/my_sim/examples/schlogl_settings.xml'

sim_type = 'TL'
with Timer() as t:
    simR_TL = Main.main(['-m', sbml_file, '-s', settings_file, '-t', sim_type])
print "=> elapsed tauLeaping: %s s" % t.secs

sim_type = 'G'
simR_G = Main.main(['-m', sbml_file, '-s', settings_file, '-t', sim_type])


#####
# RESULTS FIGURES FOR SCHLOGL
#
import matplotlib.pyplot as plt
from scipy import stats
import numpy

# # Histogram at particular time
# species_idx = 0
# time_idx = 40
# num_bins = 200
# plt_range = [0, 700]
# n_TL, bins_TL, patches_TL = plt.hist(simR_TL[species_idx][time_idx], num_bins, range=plt_range, normed=1, facecolor='blue', alpha=0.5)
# n_G, bins_G, patches_G = plt.hist(simR_G[species_idx][time_idx], num_bins, range=plt_range, normed=1, facecolor='red', alpha=0.5)


# # Density plot at particular time
# species_idx = 0
# time_idx = 40
#
# density_TL = stats.gaussian_kde(simR_TL[species_idx][time_idx])
# # density_G = stats.gaussian_kde(simR_G[species_idx][time_idx])
#
# plt_range = numpy.arange(0, 700, 1)
# plt.figure(1)
# plt.plot(plt_range, density_TL(plt_range), 'r-')
# # plt.plot(plt_range, density_G(plt_range), 'b-')

# ## Plot all paths
# plt.figure(2)
# my_range = numpy.linspace(0, 10, num=101)
# plt.plot(my_range, simR_TL[0])
# plt.plot(my_range, simR_G[0])

from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
species_idx = 0
plt_range = numpy.arange(0, 700, 1)

fig = plt.figure(1)
data_TL = []
data_G = []
for time_idx in numpy.arange(10, 100, 10):
    density_TL = stats.gaussian_kde(simR_TL[species_idx][time_idx])
    density_G = stats.gaussian_kde(simR_G[species_idx][time_idx])
    # print density_TL(plt_range)
    data_TL.append(list(zip(plt_range, density_TL(plt_range))))
    data_G.append(list(zip(plt_range, density_G(plt_range))))
    # plt.plot(plt_range, density_TL(plt_range))

poly_TL = PolyCollection(data_TL)
poly_TL.set_alpha(0.5)
poly_TL.set_color('red')

poly_G = PolyCollection(data_G)
poly_G.set_alpha(0.5)
poly_G.set_color('blue')

# print data
ax = fig.gca(projection='3d')
ax.set_xlabel('species_0')
ax.set_ylabel('time')
ax.set_xlim3d(0, 700)
ax.set_ylim3d(0, 11)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 0.01)


# ax.plot(data, numpy.arange(10, 100, 10), zs=plt_range, color='b')
ax.add_collection3d(poly_G, zs=numpy.arange(1, 10, 1), zdir='y')
ax.add_collection3d(poly_TL, zs=numpy.arange(1, 10, 1), zdir='y')
plt.show()
#####