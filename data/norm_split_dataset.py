import sys
import os
import numpy as np

root_dir = './istella_letor/'
train_raw_file = root_dir + './full/train.txt'
test_raw_file = root_dir + './full/test.txt'

out_dir = root_dir + 'split/'
train_file = out_dir + 'train.txt'
valid_file = out_dir + 'valid.txt'
test_file = out_dir + 'test.txt'

os.system('mkdir %s' % out_dir)
os.system('cp %s %s' % (test_raw_file, test_file))

all_query = set()
for line in open(train_raw_file):
    part = line.strip().split()
    label = part[0]
    q = part[1]
    all_query.add(q)

# Split Train set to train and valid
import random
all_query = list(all_query)
random.shuffle(all_query)

valid_count = len(all_query) // 8
valid_query = set(all_query[:valid_count])
train_query = set(all_query[valid_count:])

print(len(all_query))
print(len(train_query), len(valid_query))

ftrain = open(train_file, 'w')
fvalid = open(valid_file, 'w')

for line in open(train_raw_file):
    part = line.strip().split()
    label = part[0]
    q = part[1]
    if q in train_query:
        ftrain.write(line)
    else:
        fvalid.write(line)

ftrain.close()
fvalid.close()

# Normalize Features

from glob import glob
flist = glob(out_dir + '/*.txt')

import tqdm

def statisitc(flist):
    x = [[] for i in range(221)]
    for filename in flist:
        print('[Info] Processing', filename)
        for line in tqdm.tqdm(open(filename)):
            part = line.split()[2:]
            for p in part:
                id, val = p.split(':')
                id = int(id)
                val = float(val)
                if val == 1.79769313486e+308:
                    val = -1000
                x[id].append(val)
    state = []
    for i in range(1, 221):
        state.append([np.min(x[i]), np.max(x[i]), np.mean(x[i]), len(np.unique(x[i]))])
        #print('Feature', i)
        #print('Min:', state[-1][0], 'Max:', state[-1][1], 'Mean:', state[-1][2], 'Uniq:', state[-1][3])
    return state

state = statisitc(flist)

filtered_state = {}
filtered_id = 1
for id, s in enumerate(state):
    if s[3] > 1:
        filtered_state[id+1] = [filtered_id, s]
        filtered_id += 1
    else:
        filtered_state[id+1] = [-1, None]

def normalize(filename, filtered_state):
    fout = open(filename+'.norm', 'w')
    print(filename)
    for line in tqdm.tqdm(open(filename)):
        part = line.strip().split()
        fout.write('%s %s ' % (part[0], part[1]))
        for p in part[2:]:
            id, val = p.split(':')
            id = int(id)
            if filtered_state[id][0] != -1:
                s = filtered_state[id][1]
                new_id = filtered_state[id][0]
                val = float(val)
                if val == 1.79769313486e+308:
                    val = -1000
                val = (val - s[0]) / (s[1] - s[0])
                fout.write('%s:%s ' % (new_id, val))
        fout.write('\n')
    fout.close()

for f in flist:
    normalize(f, filtered_state)

