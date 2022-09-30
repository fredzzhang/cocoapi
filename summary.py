"""
Print a summary of the generated dataset

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import json

sys.path.append("PythonAPI")
from pycocotools.coco import COCO

DIR = "./tmp"

with open(os.path.join(DIR, "dataset.json"), 'r') as f:
    labels = json.load(f)["labels"]

num = dict()
for x in labels:
    if x[1] in num:
        num[x[1]] += 1
    else:
        num[x[1]] = 1

coco = COCO("./annotations/instances_val2014.json")
for k, v in num.items():
    category = coco.loadCats(k)[0]
    print(category["name"], v)