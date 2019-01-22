for i in range(1, 5):
    with open('code/checkpoints/ultraslimS_{}/testIoU.txt'.format(i)) as f:
        x, y = zip(*[[int(v[0]), float(v[1])] for v in (s.split(' ') for s in (l.rstrip('\n') for l in f) if s)])
        #x, y = [float(s.split(' ')[1]) for s in (l.rstrip('\n') for l in f) if s])
        ymax = max(y)
        x_ymax = x[y.index(ymax)]
        print(f.name, x_ymax, ymax)