# open an image and insert another image into

from PIL import Image
import glob
import os
import numpy as np
import cv2

img_source_folder = 'c:/nn/dataset/mazzo_bicycle'
img_out_folder = 'c:/nn/dataset/dest'
num_immagini_da_generare = 200 # genera x immagini per ogni immagine presente in source
write_boxed_images = False # write image con box per test
xml_database_info = 'BurracoPoints'


file_list = os.listdir(img_source_folder)

images_list = []

for j, item in enumerate(file_list):
  
  # filtra solo le immagini png
  if item.lower().endswith(('.png')):
    img_path = img_source_folder + '/' + item
    images_list.append(img_path)

n_pic = 0
s_w = 0
s_h = 0

for j, img_path in enumerate(images_list):
 
    print('img_path:', img_path)
    img = Image.open(img_path, 'r')
    img_w, img_h = img.size
    s_h = s_h + img_h
    s_w = s_w + img_w
    print(img.size)
    n_pic = n_pic + 1
    

m_size = ((s_w) // n_pic, (s_h) // n_pic)  

print(m_size)


for j, img_path in enumerate(images_list):
    print('img_path:', img_path)
    fname, fextension = os.path.splitext(img_path)
    fpath, fname2 = os.path.split(fname)
    print(fpath, fname, fname2, fextension)
    out_file = fname + '.jpg'
    print(out_file)
    img = Image.open(img_path, 'r')
    imgr = img.resize((400,550), Image.ANTIALIAS)
    imgr = imgr.convert("RGB")
    imgr.save(out_file)

# Resize and save



'''
img = Image.open('C:/nn/dataset/tmp/10Q.jpg', 'r')
img_w, img_h = img.size
print(img.size)
basewidth = 200
hsize = 300
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img_w, img_h = img.size
print(img.size)

background = Image.open('C:/nn/sample_imgs/panno-verde-900x900.jpg', 'r')
bg_w, bg_h = background.size
print(background.size)
offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
print(offset)

background.paste(img, offset)
background.save('C:/nn/sample_imgs/out.jpg')
'''