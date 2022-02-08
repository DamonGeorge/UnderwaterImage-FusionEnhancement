import argparse
import glob
import multiprocessing as mp
from pathlib import Path

import cv2
from tqdm import tqdm

from fusion_enhance import enhance

parser = argparse.ArgumentParser(description='apply fusion underwater enhancement to images')
parser.add_argument('image', metavar='F', type=str,
                    help='either an image or a directory containing images to enhance')
parser.add_argument('-o', '--output-folder', type=str, default='./enhanced',
                    help='output folder for enhanced images')
parser.add_argument('--processes', '-p', type=int, default=1,
                    help="max number of worker processes")
parser.add_argument('--level', '-l', type=int, default=3,
                    help="fusion enhancement level")
parser.add_argument('--balance-clip', '-b', type=int, default=3,
                    help="fusion color balance clip percentage")
parser.add_argument('--clahe-clip', '-c', type=float, default=4.0,
                    help="fusion clahe clip level")

args = parser.parse_args()
path = Path(args.image).resolve()
output_path = Path(args.output_folder).resolve()
output_path.mkdir(parents=True, exist_ok=True)

valid_suffixes = {'.jpeg', '.jpg', '.png', '.tiff', '.tif', '.webp'}

if path.is_file():
    if path.suffix in valid_suffixes:
        filenames = [(path, output_path / path.name)]  # input, output path
    else:
        print(f"Invalid filename extension")
        exit(-1)
elif path.is_dir():
    filenames = []
    ps = glob.glob(str(path / '**' / '*'), recursive=True)
    for p in ps:
        p = Path(p).resolve()
        if p.is_file() and p.suffix in valid_suffixes:
            filenames.append((p, output_path / p.relative_to(path)))
            filenames[-1][-1].parent.mkdir(parents=True, exist_ok=True)
else:
    print(f"Path doesn't exist!")
    exit(-1)


def enhance_helper(arg):  # our helper
    input_path, output_path = arg
    img = cv2.imread(str(input_path))
    enhanced = enhance(img,
                       level=args.level,
                       color_balance_clip_percentile=args.balance_clip,
                       clahe_clip_limit=args.clahe_clip)
    cv2.imwrite(str(output_path), enhanced * 255)
    # tqdm.write(f"Writing {str(output_path)}...")


if args.processes <= 1 or len(filenames) < args.processes:
    print(f"Enhancing {len(filenames)} files...")
    for f in tqdm(filenames,):
        enhance_helper(f)
else:
    print(f"Enhancing {len(filenames)} files with {args.processes} processes...")
    with mp.Pool(processes=args.processes) as pool:
        for _ in tqdm(pool.imap(enhance_helper, filenames, chunksize=10), total=len(filenames)):
            pass
