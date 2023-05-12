source ~/.zshrc

rm -r blog_contents_files

conda activate blog
python -m nbconvert --to html --template classic dev/blog_contents.ipynb
python -m nbconvert --to markdown dev/blog_contents.ipynb
mv dev/blog_contents.html .
rm dev/blog_contents.md
mv dev/blog_contents_files/ .
cp dev/cloth_images/*.mp4 blog_contents_files/

python mod_fpaths.py blog_contents.html
