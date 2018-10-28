# Inserisce una immagine in un background e salva jpg
# Inoltre apre il template xml VOC e ne genera una versione per l'immagine

from PIL import Image
import glob
import os
import numpy as np
import cv2
import binascii

# xml_voc_template  = 'C:/nn/dataset/tmp/template.xml'
img_source_folder = 'c:/nn/dataset/tfrecord'
crc_file_out_folder = 'c:/nn/dataset/tfrecord'
#num_immagini_da_generare = 200 # genera x immagini per ogni immagine presente in source
#write_boxed_images = False # write image con box per test
#xml_database_info = 'BurracoPoints'

import binascii

def CRC32_from_file(filename):
    buf = open(filename,'rb').read()
    buf = (binascii.crc32(buf) & 0xFFFFFFFF)
    return "%08X" % buf

file_list = os.listdir(img_source_folder)

images_list = []
crc_list = []

for j, item in enumerate(file_list):
  
  # filtra solo i file tfrecord
  if item.lower().endswith(('.tfrecord')):
    img_path = img_source_folder + '/' + item
    images_list.append(img_path)

n_pic = 0
s_w = 0
s_h = 0

for j, img_path in enumerate(images_list):
 
    print('read:', img_path)
    #img = Image.open(img_path, 'r')
    #img_w, img_h = img.size
    #print(img.size)

    crc32 = CRC32_from_file(img_path)


    print(crc32)

    # write crc file

    fname, fextension = os.path.splitext(img_path)
    fpath, fname2 = os.path.split(fname)
    print(fpath, fname, fname2, fextension)
    crc_out_file = img_path + '.svf'
    print(fname2, fextension, crc32)
    print('write:', crc_out_file)
    crc_list.append(fname2 + fextension + ' ' + crc32)



with open(img_source_folder  + '/crc32.sfv', 'w') as f:
    for item in crc_list:
        f.write("%s\n" % item)
      


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


with open(filename, 'rb') as f:
    crc = 0
    while True:
        b = f.read(0x6400000)
        if len(b) == 0:
        break
        crc = binascii.crc32(b, crc)

print(crc)




'''