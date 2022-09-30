"""
Preprocess the dataset to generate cropped images of objects

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import json
import click
import random

from tqdm import tqdm
from PIL import Image

sys.path.append("PythonAPI")
from pycocotools.coco import COCO

def crop(image, box, min_size=200, max_shift=.5):

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    bw = x2 - x1; bh = y2 - y1

    # Skip small enclosing boxes
    if max(bw, bh) < min_size:
        return None

    # Pad the bounding box to a square
    if bw < bh:
        x1 = cx - bh / 2
        x2 = cx + bh / 2
    else:
        y1 = cy - bw / 2
        y2 = cy + bw / 2
    
    # Check for overflow and underflow
    iw, ih = image.size
    if x1 < 0:
        # Discard the image if the shifted distance is larger
        # than a specified proportion of the image dimensions
        if -x1 >= max_shift * bw:
            return None
        x2 -= x1; x1 = 0
        if x2 > iw: return None
    if y1 < 0:
        if -y1 >= max_shift * bh:
            return None
        y2 -= y1; y1 = 0
        if y2 > ih: return None
    if x2 > iw:
        if (x2 - iw) >= max_shift * bw:
            return None
        x1 -= (x2 - iw); x2 = iw
        if x1 < 0:  return None
    if y2 > ih:
        if (y2 - ih) >= max_shift * bh:
            return None
        y1 -= (y2 - ih); y2 = ih
        if y1 < 0:  return None

    cropped = image.crop([round(x1), round(y1), round(x2), round(y2)])

    # Sanity check
    # iw, ih = cropped.size
    # if iw != ih:
    #     raise ValueError("The cropped image is not a square! {} != {}".format(iw, ih))

    return cropped

class SimpleDict(dict):
    def __init__(self, input_dict=None, **kwargs):
        data_dict = dict() if input_dict is None else input_dict
        data_dict.update(kwargs)
        super(SimpleDict, self).__init__(**data_dict)
    def __getattr__(self, name):
        if name in self:    return self[name]
        else:               raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value

@click.command()
@click.option('--data-root', type=str, default='./data', show_default=True)
@click.option('--partition', type=str, default='train2014', show_default=True)
@click.option('--min-size', type=float, default=200., show_default=True)
@click.option('--cache-dir', type=str, default='./tmp', show_default=True)
@click.option('--trial', type=bool, default=False, show_default=True)

def main(**kwargs):
    args = SimpleDict(kwargs)
    # Create directory for preprocessed images
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # Initialise datasets
    coco = COCO(os.path.join(args.data_root, "annotations/instances_{}.json".format(args.partition)))
    image_dir = os.path.join(args.data_root, args.partition)

    ids = list(coco.imgs.keys())
    # Select a small collection of images for sanity check
    if args.trial:
        ids = random.sample(ids, 100)

    labels = []
    print("Processing the dataset...")
    for i in tqdm(ids):
        meta = coco.loadImgs(i)[0]
        image = Image.open(os.path.join(image_dir, meta["file_name"]))
        target = coco.loadAnns(coco.getAnnIds(i))

        # Crop out each bounding box and save as an image
        for j, tgt in enumerate(target):
            image_ = crop(image, tgt["bbox"], min_size=args.min_size)
            if image_ is not None:
                category_id = int(tgt["category_id"])
                category = coco.loadCats(category_id)[0]["name"]
                name = "{}_{}_{}".format(category, j, meta["file_name"])
                image_.save(os.path.join(args.cache_dir, name))
                labels.append([name, category_id])

    # Save the labels
    label_fname = os.path.join(args.cache_dir, 'dataset.json')
    if os.path.exists(label_fname):
        with open(label_fname, 'r') as f:
            existing_labels = json.load(f)
        cache_labels = existing_labels['labels'] + labels
        with open(label_fname, 'w') as f:
            json.dump({'labels': cache_labels}, f)
    else:
        with open(label_fname, 'w') as f:
            json.dump({'labels': labels}, f)

if __name__ == "__main__":
    main()
