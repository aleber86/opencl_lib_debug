import argparse as agp

def menu_cmd():
    Arg_parse = agp.ArgumentParser()

    Arg_parse.add_argument('-F','--full-test', type = bool,
                           action = 'store',
                           help = 'Executes every posible function.\n',
                           default = True)

    Arg_parse.add_argument('-sP','--simple-precision', type = float,
                           action = 'store',
                           help = 'Simple precision limit value for absolute error.\n',
                           default = 1.e-6)

    Arg_parse.add_argument('-dP','--double-precision', type = float,
                           action = 'store',
                           help = 'Double precision limit value for absolute error.\n',
                           default = 1.e-15)

    Arg_parse.add_argument('-wL','--warn-limit', type = int,
                           action = 'store',
                           help = 'Warning limit. Default: 1 order of magnitud. Set 0 for no threshold\n',
                           default = 1)

    Arg_parse.add_argument('-v', '--verbose', type = bool,
                           help = 'Prints (min,max) value for absolute error.\n',
                           action = 'store',
                           default = False)

    Arg_parse.add_argument('-l', '--log', type = bool,
                           help = 'Prints computation to a file.\n',
                           action = 'store',
                           default = False)

    command_values = Arg_parse.parse_args()
    return command_values

if __name__ == "__main__":
    arguments = menu_cmd()
    print(arguments.__dict__)
