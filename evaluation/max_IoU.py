import os
import sys
from glob import glob

y_index = int(sys.argv[1]) if len(sys.argv) > 1 else 1

for fname in glob('../code/test/*/testIoU.txt'):
	with open(fname) as f:
		x, y = zip(*[[int(v[0]), float(v[y_index])] for v in (s.split(' ') for s in (l.rstrip('\n') for l in f) if s)])
		#x, y = [float(s.split(' ')[1]) for s in (l.rstrip('\n') for l in f) if s])
		ymax = max(y)
		x_ymax = x[y.index(ymax)]
		print("{:>20}: {} {}".format(os.path.basename(os.path.dirname(fname)), x_ymax, ymax))
