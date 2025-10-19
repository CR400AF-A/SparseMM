import matplotlib.pyplot as plt
import numpy as np
import json

with open('./visual_head/head_score/llava-v1.6.json') as f:
    head_list = json.load(f)

## use the average visual score and ranking
head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
visual_heads = [[l[0],  round(np.mean(l[1]), 6)] for l in head_score_list]

data = np.random.rand(32, 32) # for qwen, set to (28, 28)

for head in visual_heads:
    pos = head[0]
    data[pos[0], pos[1]] = head[1]

fig, ax = plt.subplots()

cax = ax.imshow(data, cmap='Greens', interpolation='nearest')

cbar = fig.colorbar(cax, ax=ax)
plt.savefig('viz/VizHead_llava-v1.6.png')
