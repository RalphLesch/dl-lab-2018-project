import sys
import subprocess
import re
import json

def s_to_hms(s):
	#min, s = divmod(int(sys.argv[1]), 60)
	min, s = divmod(int(float(s)), 60)
	h, min = divmod(min, 60)
	return '{:02}:{:02}:{:02}'.format(h,min,s)

output = subprocess.check_output("tail -n 2  */train.log | grep '=='", shell=True, stderr=subprocess.STDOUT)

if output:
	for m in re.finditer(r"==> ([^/]+)/[^<=]+<==\s*Global training time ==\s*([\d.]+)$", output.decode('ascii'), re.M):
		print("{:>10}: {}".format(m.group(1), s_to_hms(m.group(2))))
