import Main

sbml_file = '/home/sandy/Documents/Code/my_sim/examples/lotkaVoltera_sbml.xml'
settings_file = '/home/sandy/Documents/Code/my_sim/examples/lotkaVolterra_settings.xml'

sim_type = 'TL'
simR_TL = Main.main(['-m', sbml_file, '-s', settings_file, '-t', sim_type])

sim_type = 'G'
simR_G = Main.main(['-m', sbml_file, '-s', settings_file, '-t', sim_type])


#####
# RESULTS FIGURES FOR LOTKA VOLTERRA
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import numpy


my_range = numpy.linspace(0, 10, num=10001)

mean_TL_0 = numpy.mean(simR_TL[0], axis=1)
mean_TL_1 = numpy.mean(simR_TL[1], axis=1)

sd_TL_0 = numpy.std(simR_TL[0], axis=1)
sd_TL_1 = numpy.std(simR_TL[1], axis=1)

mean_G_0 = numpy.mean(simR_G[0], axis=1)
mean_G_1 = numpy.mean(simR_G[1], axis=1)

sd_G_0 = numpy.std(simR_G[0], axis=1)
sd_G_1 = numpy.std(simR_G[1], axis=1)

plt.figure(1)
plt.subplot(211)
plt.xlabel('Time')
plt.ylabel('Mean')
plt.title('Variation of mean')
plt.plot(my_range, mean_TL_0, 'b-')
plt.plot(my_range, mean_TL_1, 'g-')

plt.plot(my_range, mean_G_0, 'r-')
plt.plot(my_range, mean_G_1, 'y-')

plt.subplot(212)
plt.xlabel('Time')
plt.ylabel('Standard deviation')
plt.title('Variation of standard deviation')
plt.plot(my_range, sd_TL_0, 'b-')
plt.plot(my_range, sd_TL_1, 'g-')

plt.plot(my_range, sd_G_0, 'r-')
plt.plot(my_range, sd_G_1, 'y-')
plt.show()

fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.set_xlabel('species_0')
ax.set_ylabel('species_1')
ax.plot(mean_TL_0, mean_TL_1, zs=my_range, color='b')
ax.plot(mean_G_0, mean_G_1, zs=my_range, color='r')
plt.show()
#####
