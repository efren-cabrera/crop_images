from typing import List
import cv2
import os


def save_images_as_original_tree(original_paths: List[str], images, destionation_folder) -> None:
    for original_path, image in zip(original_paths, images):
        _, original_path_route = os.path.splitdrive(original_path)
        if original_path_route.startswith("/"):
            original_path_route = original_path_route.replace("/", "", 1)
        if original_path_route.startswith("//"):
            original_path_route = original_path_route.replace("//", "", 1)
        final_file_route = os.path.join(destionation_folder, original_path_route)
        final_path, _ = os.path.split(final_file_route)
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        cv2.imwrite(final_file_route, image)
