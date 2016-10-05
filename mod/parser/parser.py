import numpy
import string
import xml.etree.ElementTree as ET
from jinja2 import Template


class Parser:
    @staticmethod
    def parse_settings(settings_file, args_obj):
        """
        THIS NEEDS SOME SANITY CHECKING FOR ROBUSTNESS
        :param settings_file: XML file containing user defined settings for the
         simulations
        :param args_obj: an object holding the parameters to be passed in to the
         simulator
        """
        tree = ET.parse(settings_file)
        root = tree.getroot()

        out_species = root.find('output_species')
        temp_E = []
        for species in out_species.findall('species'):
            temp_E.append(int(species.get('idx')))

        args_obj.E = numpy.array(temp_E).astype(numpy.int32)

        args_obj.t_max = float(root.find('duration').text)
        args_obj.ita = int(root.find('num_output_time_points').text)
        args_obj.U = int(root.find('num_simulations').text)

    @staticmethod
    def define_hazards(spn):
        hazards_mod = Template(
            """
            __device__ void UpdatePropensities(double* __restrict__ a, int* __restrict__ x)
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
                                              'd_c[' + str(i - 1) + ']')

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
