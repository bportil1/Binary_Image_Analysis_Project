import numpy as np
from PIL import Image
import os

from binary_parser import *

def collect_dirs(dir_path):
    files = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            files.append(file_path)
    return files

def convert_binaries(files, out_dir):
    for file in files:
        with open(file, mode='rb') as data:
            bin_data = np.fromfile(data, dtype=np.uint8)
            base_name = os.path.splitext(os.path.basename(file))[0]
            out_file_path = os.path.join(out_dir, f"{base_name}.png")

            sections, entropy_values = parse_binary_file(file)
            #for start, end, section_type, entropy in sections:
            #    print(f"Bytes {start}-{end}: {section_type} (Entropy: {entropy:.2f})")

            PILimage = generate_visualization(bin_data, sections)

            #PILimage = Image.fromarray(bin_data)
            PILimage.save(out_file_path)

def test_driver():
    base_dir = '/bin'
    out_dir = './binary_images'
    os.makedirs(out_dir, exist_ok=True)

    files = collect_dirs(base_dir)

    convert_binaries(files, out_dir)

if __name__ == '__main__':
    test_driver()
