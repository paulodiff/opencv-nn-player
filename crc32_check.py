# verify a CRC32 code presnet in sfv file
# sfv file entry: FNAME CRC32CODE

from PIL import Image
import glob
import os
import numpy as np
import cv2
import binascii
import os.path
import binascii

img_source_folder = 'c:/nn/dataset/tfrecord'

def CRC32_from_file(filename):
    buf = open(filename,'rb').read()
    buf = (binascii.crc32(buf) & 0xFFFFFFFF)
    return "%08X" % buf

file_list = os.listdir(img_source_folder)

images_list = []
crc_list = []

for j, item in enumerate(file_list):
  
  # filtra solo i file sfv
  if item.lower().endswith(('.sfv')):
    img_path = img_source_folder + '/' + item
    images_list.append(img_path)

crc_ok = 0
crc_error = 0
file_not_found = 0
file_count = 0

for j, img_path in enumerate(images_list):
 
    print('read:', img_path)

    lines = [line.rstrip('\n').strip() for line in open(img_path)]
    curr_id = 0

    for l in lines:
        file_count = file_count + 1
        file_to_check = l.split(' ')[0]
        file_crc32 = l.split(' ')[1]

        #print(file_to_check, file_crc32)
        file_to_check_path = img_source_folder + '/' + file_to_check
        #print('check:', file_to_check_path)
        if os.path.exists(file_to_check_path):
            crc32 = CRC32_from_file(file_to_check_path)
            if crc32 == file_crc32:
                crc_ok = crc_ok + 1
                print('CRCOK          :', file_to_check_path, crc32, file_crc32)
            else:
                crc_error = crc_error + 1
                print('CRCERROR       :', file_to_check_path, crc32, file_crc32)
        else:
            print('FILE NOT FOUND :', file_to_check_path, file_crc32)
            file_not_found = file_not_found + 1

  

print('file count:', file_count, 'crc_ok:', crc_ok, 'crc_error:', crc_error, 'file not found:', file_not_found )

 