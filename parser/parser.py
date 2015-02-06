import numpy
import string

from jinja2 import Template


class Parser:
    @staticmethod
    def define_hazards(spn):
        hazards_mod = Template(
            """
            __device__ void UpdatePropensities(float* a, uint* x, float* c)
            {
            {{hazards}}
            }
            """)
        hazards_list_string = ''
        for haz_idx, haz in enumerate(spn.h):
            hazards_list_string = hazards_list_string + '    ' + 'a[' + str(
                haz_idx) + '] = ' + haz + ';\n'
        hazards_mod_code = hazards_mod.render(hazards=hazards_list_string)
        for i in range(len(spn.P), 0, -1):
            hazards_mod_code = string.replace(hazards_mod_code,
                                              'species_' + str(i - 1),
                                              'x[' + str(i - 1) + ']')
        for i in range(len(spn.c), 0, -1):
            hazards_mod_code = string.replace(hazards_mod_code,
                                              'parameter_' + str(i - 1),
                                              'c[' + str(i - 1) + ']')

        # mymod = SourceModule(hazards_mod_code)
        # print hazards_mod_code
        return hazards_mod_code

    @staticmethod
    def flatten_matrix(matrix):
        # each entry in the flattened matrix is an uchar4
        f_matrix = []

        count = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] != 0:
                    entry = numpy.array(
                        [numpy.byte(i), numpy.byte(j), numpy.byte(matrix[i][j]),
                         numpy.byte(0)])
                    f_matrix.append(entry)
                    count += 1
        return numpy.array(f_matrix), numpy.ubyte(count)
