import argparse

def print_args(arg_map):
    for key in arg_map.keys():
        print("key {} has value {}".format(key, arg_map[key]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("test program for argparse")

    # add the test argument
    parser.add_argument('-t', '--test', help='argument to pass in test', required=True)

    # add the test argument
    parser.add_argument('-b', '--boo', help='argument to pass in boo', required=False)

    # add the test argument
    parser.add_argument('-s', '--secret', help='argument to pass in secret', required=False)

    # get the args
    args = vars(parser.parse_args())

    # print the args
    print_args(args)



