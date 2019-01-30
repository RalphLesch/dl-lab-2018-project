# Plot results

import argparse
import sys
import os
from glob import glob
import matplotlib.pyplot as plt

def get_data(file):
    with open(file) as f:
        #data = [nun for n in (s.split(' ') for s in (l.rstrip('\n') for l in f) if s)]
        data = [[int(v[0]), float(v[1])] for v in (s.split(' ') for s in (l.rstrip('\n') for l in f) if s)]
    return zip(*data)


def plot_all_checkpoint_IoU(args):
	file_paths = glob(os.path.join(args.checkpoints, 'testIoU.txt'))
	names = [os.path.basename(os.path.dirname(f)) for f in file_paths]
	
	args.file = file_paths
	if not args.title:
		args.title = names
	plot(args)


def plot(args):
    if not args.file:
    	return
    if len(args.file) == 1:
        args.file = glob(args.file[0]) or args.file
        
    if args.title == []:
        args.title = (os.path.splitext(os.path.basename(f))[0] for f in args.file)
    elif not args.title:
        args.title = (False for f in args.file)
    
    if not args.labels:
        args.labels = (False for f in args.file)
    
    if args.color and args.merge:
        for x in range(args.color):
            plt.plot([], [])
    
    for i, (file, title, label) in enumerate(zip(args.file, args.title, args.labels)):
        data = get_data(file)
        
        extra = {'label': label} if label else {}
        if args.merge:
            print('adding {} as {}'.format(file, label))
            plt.plot(*data, **extra)
        else:
            plt.figure()
            plt.ylabel(args.ylabel)
            plt.xlabel(args.xlabel)
            if title:
                plt.title(title)
            
            if args.color:
                for _ in range(i + args.color):
                    plt.plot([], [])
            
            plt.plot(*data, **extra)
            if args.output or args.output == '':
                if args.output == '':
                    outputName = title or os.path.splitext(file)[0]
                elif os.path.isdir(args.output):
                    outputName = os.path.join(args.output, title or os.path.splitext(os.path.basename(file))[0])
                else:
                    outputName = os.path.splitext(args.output)[0]
                outputName = '{}.{}'.format(outputName, args.format)
                print('Saving ' + outputName)
                plt.savefig(outputName)
            if not args.noplot:
                print('Showing', title or file, sep=' ')
                plt.show()
    if args.merge:
        plt.legend(bbox_to_anchor=(0.95, 0.85))  # CUSTOM

        plt.ylabel(args.ylabel)
        plt.xlabel(args.xlabel)
        if args.output or args.output == '':
            outputName = '{}.{}'.format((args.output or title or os.path.splitext(args.file[0])[0] + '_merged'), args.format)
            print('Saving ' + outputName)
            plt.savefig(outputName)
        if not args.noplot:
            print('Showing', title or file, sep=' ')
            plt.show()
	

if __name__ == "__main__":
    default_checkpoint_dir = '../code/checkpoints/*'
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, nargs='+',
                        help='result json file(s)')
    group.add_argument('--checkpoints', type=str, nargs='*', metavar='CHECKPOINT_DIRS',
                        help=('testIoU.txt files under CHECKPOINT_DIRS, default: ' + default_checkpoint_dir))
    
    parser.add_argument('-t', '--title', type=str, nargs='*',
                        help='title of the plot, defaults to filename')
    parser.add_argument('-y', '--ylabel', type=str, nargs='?',
                        help='y label', default='error')
    parser.add_argument('-x', '--xlabel', type=str, nargs='?',
                        help='x label', default='epoch')
    parser.add_argument('-o', '--output', type=str, nargs='?',
                        help='save to output folder or file, defaults to title.format',
                        metavar='PATH', const='')
    parser.add_argument('-f', '--format', type=str, nargs='?',
                        help='output format, defaults to png', default='png')
    parser.add_argument('-n', '--noplot', action='store_true',
                        help='do not show plot')
    parser.add_argument('-m', '--merge', action='store_true',
                        help='merge plots into one')
    parser.add_argument('-l', '--labels', nargs='+',
                        help='labels for in a merge plot')
    parser.add_argument('-c', '--color', type=int, nargs='?',
                        help='preserve the color in single plots like in merged plot, starting with COLOR_INDEX', metavar='COLOR_INDEX', default=0)
    parser.add_argument('-b', '--backend', type=str, help='use different matplotlib backend')
    args = parser.parse_args()
    
    if args.backend:
    	plt.switch_backend(args.backend)
    
    if args.file is None:
    	if not args.checkpoints:
    		args.checkpoints = default_checkpoint_dir
    	plot_all_checkpoint_IoU(args)
    else:
    	plot(args)
            
