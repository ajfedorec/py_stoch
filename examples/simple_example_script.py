if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import Main

sbml_file = '/home/sandy/Documents/Code/my_sim/examples/simple_sbml.xml'
settings_file = '/home/sandy/Documents/Code/my_sim/examples/simple_settings.xml'

sim_type = 'TL'
simR_TL = Main.main(['-m', sbml_file, '-s', settings_file, '-t', sim_type])

sim_type = 'G'
simR_G = Main.main(['-m', sbml_file, '-s', settings_file, '-t', sim_type])

#####
#   RESULTS FOR SIMPLE MODEL
#
import matplotlib.pyplot as plt
from scipy import stats
import numpy


density_TL = stats.gaussian_kde(simR_TL[0][1])
density_G = stats.gaussian_kde(simR_G[0][1])

plt_range = numpy.arange(10, 90, 0.1)

plt.plot(plt_range, density_TL(plt_range), 'r-')
plt.plot(plt_range, density_G(plt_range), 'b-')

plt.xlabel('S3 Population')
plt.ylabel('Proportion')
plt.title('S3(0.1)')

plt.show()
