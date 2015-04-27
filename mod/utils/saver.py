from datetime import datetime
import numpy
import os


def save_results(sim_result, arg_obj, sim_name, sim_type='TL'):
    # sim_result is len(Kappa) x len(Ita) x thread_num
    # kappa is likely to be the smallest, so we should split files based on the
    # value of kappa.

    cwd = os.getcwd()

    now = datetime.now()
    dt = '{0}-{1}-{2}-{3}-{4}'.format(str(now.year), str(now.month),
                                      str(now.day), str(now.hour),
                                      str(now.minute))

    x0 = 'x'
    for i in range(len(arg_obj.x_0)):
        x0 += '-' + str(arg_obj.x_0[i])

    c = '_c'
    for i in range(len(arg_obj.c)):
        c += '-' + str(arg_obj.c[i]).rstrip('.0')

    out_dir = cwd + '/' + sim_name + '/' + x0 + c + '/'

    try:
        os.makedirs(out_dir)
    except OSError, e:
        print "folder already exists"

    depth = sim_result.ndim

    if depth == 3:
        for species_idx in range(len(sim_result)):
            filename = out_dir + sim_type + '_species' + str(
                species_idx) + '_' + dt
            numpy.savetxt(filename, sim_result[species_idx], fmt='%1u')
    elif depth == 2:
        for species_idx in range(len(sim_result[0])):
            filename = out_dir + sim_type + '_species' + str(
                species_idx) + '_' + dt
            numpy.savetxt(filename, sim_result[:, species_idx], fmt='%1u')
