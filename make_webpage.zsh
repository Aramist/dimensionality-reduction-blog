source ~/.zshrc

# cleanup
rm -r blog_contents_files
rm index.html

conda activate blog
python -m nbconvert --to html --template classic dev/blog_contents.ipynb
python -m nbconvert --to markdown dev/blog_contents.ipynb
mv dev/blog_contents.html ./index.html
rm dev/blog_contents.md
mv dev/blog_contents_files/ .
cp dev/cloth_images/*.mp4 blog_contents_files/

python modify_vid_paths.py -p index.html
