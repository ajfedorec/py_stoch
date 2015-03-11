from __future__ import division

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import Main
from mod.utils import Timer


sbml_file = '/home/sandy/Downloads/plasmid_stability.xml'
settings_file = '/home/sandy/Downloads/plasmid_stability_settings.xml'

sim_type = 'TL'
with Timer() as t:
    simR_TL = Main.main(['-m', sbml_file, '-s', settings_file, '-t', sim_type])
print "=> elapsed tauLeaping: %s s" % t.secs

# sim_type = 'G'
# with Timer() as t:
#     simR_G = Main.main(['-m', sbml_file, '-s', settings_file, '-t', sim_type])
# print "=> elapsed Gillespie: %s s" % t.secs


#####
# RESULTS FIGURES FOR PLASMID STABILITY
#
import matplotlib.pyplot as plt
from scipy import stats
import numpy

## Xm distribution at t = 10
# density_TL = stats.gaussian_kde(simR_TL[1][100])
# density_G = stats.gaussian_kde(simR_G[1][100])
#
# plt_range = numpy.arange(0, 1000, 10)
#
# plt.plot(plt_range, density_TL(plt_range), 'r-')
# plt.plot(plt_range, density_G(plt_range), 'b-')
#
# plt.xlabel('Xm Population')
# plt.ylabel('Proportion')
# plt.title('Xm(10)')


#
propXp_TL = simR_TL[0] / (simR_TL[0] + simR_TL[1])
# print propXp_TL
density_propXp_TL = stats.gaussian_kde(propXp_TL[100])

plt_range = numpy.arange(0, 1, 0.01)
plt.plot(plt_range, density_propXp_TL(plt_range), 'r-')


# x_range = numpy.linspace(0, 100, num=101)

# plt.plot(x_range, simR_TL[1])
# plt.plot(x_range, propXp_TL)

plt.show()


