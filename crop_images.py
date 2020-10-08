import argparse

from crop_images.croppers import crop_images, YOLOCropper
from crop_images.io import save_images_as_original_tree, search_images_paths

parser = argparse.ArgumentParser(description='Crop images')

parser.add_argument('images_path', help='path of images')
parser.add_argument('destination_path', help='destination path of images')
parser.add_argument('--relation', help='destination path of images', default="2:3")

args = parser.parse_args()
images_paths = search_images_paths(args.images_path)
relation = [int(i) for i in args.relation.split(":")]
cropped_images = crop_images(images_paths, relation=relation)
save_images_as_original_tree(images_paths, cropped_images, args.destination_path)
