def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:s:", ["model=", "simulation="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-m", "--model"):
            # READ IN SBML
            # in:  path to sbml file
            # out: sbml_model object
            sbml_file = arg
            reader = libsbml.SBMLReader()
            document = reader.readSBML(sbml_file)
            # check the SBML for errors
            error_count = document.getNumErrors()
            if (error_count > 0):
                raise UserWarning(error_count + ' errors in SBML file: ' +
                                  open_file_.name)
            sbml_model = document.getModel()

        elif opt in ("-s", "--simulation"):
    # READ IN SIM_XML
    # in:  path to simulation xml file
    # out: sim_info object
    #TODO


    # PARSE SBML - DEPENDENT ON SIM_XML
    #   I.E. PARSING MAY BE DIFFERENT FOR TAU LEAPING, RANGES OF PARAMETERS ETC.
    #   in:  sbml_model object, "simulation setup info"?
    #   out: simulationArguments object
    if sim_type in "TL":
        sim_args = tl_parser.parse(sbml_model, sim_info)


    # INITIATE SIMULATION
    # in:  simulationArguments object, "simulation setup info"?
    #   out: simulation exit message
    if sim_type in "TL":
        sim_result = cuTauLeaping.run(sim_args)


    # CHECK SIMULATION RESULTS
    if sim_result == 0:
        sys.exit(0)
    else:
        # TODO
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
