import argparse

parser = argparse.ArgumentParser(description='Main Entry Point')
parser.add_argument('--record', 
                    help='Usage: ')

args = parser.parse_args()
# print(args['record'])
print(args.record)