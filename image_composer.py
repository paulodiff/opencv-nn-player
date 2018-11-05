# Inserisce una immagine in un background e salva jpg
# Inoltre apre il template xml VOC e ne genera una versione per l'immagine

from PIL import Image
import glob
import os
import numpy as np
import cv2

xml_voc_template  = 'C:/nn/sample_imgs/template1.xml'
img_source_folder = 'c:/nn/dataset/mazzo_bicycle'
img_out_folder = 'c:/nn/dataset/source_1_label'
num_immagini_da_generare = 200 # genera x immagini per ogni immagine presente in source
write_boxed_images = False # write image con box per test
xml_database_info = 'BurracoPoints'


file_list = os.listdir(img_source_folder)

images_list = []

for j, item in enumerate(file_list):
  
  # filtra solo le immagini jpg
  if item.lower().endswith(('.jpg')):
    img_path = img_source_folder + '/' + item
    images_list.append(img_path)

n_pic = 0
s_w = 0
s_h = 0

for j, img_path in enumerate(images_list):
 
  print('read:', img_path)
  img = Image.open(img_path, 'r')
  img_w, img_h = img.size
  print(img.size)

  fname, fextension = os.path.splitext(img_path)
  fpath, fname2 = os.path.split(fname)
  print(fpath, fname, fname2, fextension)


  background = Image.open('C:/nn/sample_imgs/panno-verde-900x900.jpg', 'r')
  bg_w, bg_h = background.size

  print(background.size)
  offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
  print(offset)

  background.paste(img, offset)
  out_file_jpg = img_out_folder + '/' + fname2 + '-verde-900x900.jpg'
  print ('write: ',  out_file_jpg)
  background.save(out_file_jpg)

  # open xml VOC
  with open(xml_voc_template, 'r') as myfile:
    data=myfile.read()
    data = data.replace('##FOLDER##', 'dest')
    data = data.replace('##FILENAME##', fname2)
    data = data.replace('##PATH##', out_file_jpg)
    data = data.replace('##DATABASE##', 'BurracoCards')
    data = data.replace('##LABEL##', fname2)
    out_file_voc = img_out_folder + '/' + fname2 + '-verde-900x900.xml'
    print(out_file_voc)
    with open(out_file_voc, "w") as text_file:
      text_file.write(data)


    


'''
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
'''
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