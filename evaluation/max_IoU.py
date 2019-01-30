import os
from glob import glob
for fname in glob('../code/checkpoints/*/testIoU.txt'):
    with open(fname) as f:
       x, y = zip(*[[int(v[0]), float(v[1])] for v in (s.split(' ') for s in (l.rstrip('\n') for l in f) if s)])
       #x, y = [float(s.split(' ')[1]) for s in (l.rstrip('\n') for l in f) if s])
       ymax = max(y)
       x_ymax = x[y.index(ymax)]
       print("{}: {} {}".format(os.path.basename(os.path.dirname(fname)), x_ymax, ymax))
