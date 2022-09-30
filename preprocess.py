"""
Preprocess the dataset to generate cropped images of objects

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import click
import pocket
import random

from tqdm import tqdm
from PIL import Image
from typing import List, Optional
from hicodet.hicodet import HICODet

def crop(image: Image.Image, bx_1: List[float], bx_2: List[float], min_size: float = 200) -> Optional[Image.Image]:

    # Find the smallest enclosing box
    x1 = min(bx_1[0], bx_2[0]); y1 = min(bx_1[1], bx_2[1])
    x2 = max(bx_1[2], bx_2[2]); y2 = max(bx_1[3], bx_2[3])
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
        x2 -= x1; x1 = 0
        if x2 > iw: return None
    if y1 < 0:
        y2 -= y1; y1 = 0
        if y2 > ih: return None
    if x2 > iw:
        x1 -= (x2 - iw); x2 = iw
        if x1 < 0:  return None
    if y2 > ih:
        y1 -= (y2 - ih); y2 = ih
        if y1 < 0:  return None

    cropped = image.crop([int(x1), int(y1), int(x2), int(y2)])

    # Sanity check
    iw, ih = cropped.size
    if iw != ih:
        raise ValueError(f"The cropped image is not a square! {iw} != {ih}")

    return cropped

@click.command()
@click.option('--data-root', type=str, default='./hicodet', show_default=True)
@click.option('--partition', type=str, default='train2015', show_default=True)
@click.option('--class-in', type=int, default=18, show_default=True)
@click.option('--class-out', type=int, default=0, show_default=True)
@click.option('--min-size', type=float, default=200., show_default=True)
@click.option('--cache-dir', type=str, default='./tmp', show_default=True)
@click.option('--trial', type=bool, default=False, show_default=True)
@click.option('--simplify', type=bool, default=False, show_default=True)

def main(**kwargs):
    args = pocket.data.DataDict(kwargs)
    # Create directory for preprocessed images
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # Initialise datasets
    dataset = HICODet(
        root=os.path.join(args.data_root, f'hico_20160224_det/images/{args.partition}'),
        anno_file=os.path.join(args.data_root, f'instances_{args.partition}.json')
    )

    # Traverse datasets to find images with the designated interaction
    index_train = [i for i, j in enumerate(dataset._idx) if args.class_in in dataset.annotations[j]['hoi']]
    # Select a small collection of images for sanity check
    if args.trial:
        index_train = random.sample(index_train, 20)

    labels = []
    print("Processing the dataset...")
    for i in tqdm(index_train):
        image, targets = dataset[i]
        # Find human-object pairs with the designated interaction
        keep_idx = [j for j, k in enumerate(targets['hoi']) if k == args.class_in]
        
        # Skip complex scenes (with more than one pairs)
        if args.simplify and len(keep_idx) > 1:
            continue

        for j in keep_idx:
            box_h = targets['boxes_h'][j]; box_o = targets['boxes_o'][j]
            image_ = crop(image, box_h, box_o, min_size=args.min_size)
            if image_ is not None:
                name = f'{args.partition}_{args.class_in}_{i:08d}_{j+1}.png'
                image_.save(os.path.join(args.cache_dir, name))
                labels.append([name, args.class_out])

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
