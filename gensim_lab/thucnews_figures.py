"""Current THUCNews figures use the submission figure workflow."""
import argparse
from .submission_figures import build
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    args=parser.parse_args()
    build(export=args.export)
