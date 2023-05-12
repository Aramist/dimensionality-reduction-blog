import argparse
from pathlib import Path
import re


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True, help='path to the video')
file_path = ap.parse_args().path

with open(file_path, 'r') as f:
    contents = f.read()


def change_folder(path):
    stem = Path(path.group()).stem
    return f'blog_contents_files/{stem}.mp4'


# Locate the wrong paths
pattern = re.compile(r'cloth_images/.*\.mp4')
matches = pattern.findall(contents)
new_contents = pattern.sub(change_folder, contents)
with open(file_path, 'w') as f:
    f.write(new_contents)

print(f"Number of changes made: {len(matches)}")