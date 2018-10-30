# data_set_builder with annotation
# aumenta un data set con annotazioni
# partendo da alcune immagini già annotate applicando trasformazioni
# geometriche e rigenerando i file di annotazione xml
# nella cartella source devono essere presenti le immagini con annotazioni xml
# accetta solo file .jpg 
# per salvare le immagini uso PIL per ovviare ad alcuni errori sulla lettura JPG in TFRecord

import imgaug as ia
from imgaug import augmenters as iaa
import glob
import os
import numpy as np
import cv2
from scipy import misc
from skimage import data
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import PIL.Image

# setup

img_source_folder = 'c:/nn/dataset/source' # cartella con immagini e annotazioni da aumentare
img_out_folder = 'c:/nn/dataset/dest_100' # cartelle di destinazione
num_immagini_da_generare = 100 # genera x immagini per ogni immagine presente in source
write_boxed_images = False # write image con box per test
xml_database_info = 'BurracoPoints'

# start ..

file_list = os.listdir(img_source_folder)

# prepara la lista delle immagini dalla cartella img_source_folder

images_list = []

for j, item in enumerate(file_list):
  
  # filtra solo le immagini jpg
  if item.lower().endswith(('.pngXX', '.jpg')):
    img_path = img_source_folder + '/' + item
    images_list.append(img_path)

# print(images_list)    

pic_num=1
num_of_image_to_generate = num_immagini_da_generare
image_number = len(images_list)

# loop sulla lista con le trasformazioni
# legge il file .xml per la posizione VOC

for j, img_path in enumerate(images_list):
 
    pic_num=1
    #print('img_path:', img_path)

    # using cv2    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(image)
    rgb_img = cv2.merge([r,g,b])

    # using PIL
    # im = PIL.Image.open(img_path)
    # print(im.size)
    # rgb_img = im.convert('RGB')


    # read xml file con VOC - annotazioni
    xml_path = img_path[:-4] + '.xml'

    # split filename path ed estensione
    fname, fextension = os.path.splitext(img_path)
    fpath, fname2 = os.path.split(fname)
    #print(fpath, fname, fname2, fextension)
    #print('xml_path:', xml_path)

    et = ET.parse(xml_path)
    #print(et)
    root = et.getroot()
    #print(root.tag)
    
    bbs1 = []
    bbs_name = []

    xml_objects = root.findall('./object')

    #for xml_object in xml_objects:
    #    print(xml_object.find('name').text)
    #    print(xml_object.find('pose').text)
    #    print(xml_object.find('truncated').text)
    #    print(xml_object.find('difficult').text)
        
    #exit()

    for bb in root.findall('./object/bndbox'):
        xmin = bb.find('xmin').text
        xmax = bb.find('xmax').text
        ymin = bb.find('ymin').text
        ymax = bb.find('ymax').text
        # print(xmin, ymin, xmax, ymax)
        bbs1.append(ia.BoundingBox(x1=int(xmin), y1=int(ymin), x2=int(xmax), y2=int(ymax) ) )

    # remove object node
    for child in root.findall("object"):
        root.remove(child)

    bbs = ia.BoundingBoxesOnImage(bbs1,shape=rgb_img.shape) 
    # bbs = ia.BoundingBoxesOnImage(bbs1,shape=im.size) 

    #print(bbs)
  
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    seq = iaa.Sequential([
    # iaa.Multiply((1.2, 1.5)), 
    # iaa.Superpixels(p_replace=0.5, n_segments=128),
    # iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128)),
        iaa.Affine( 
            # mode='reflect',
            # translate_px={"x": 10, "y": 15},
            translate_px={"x": (-20, 20), "y": (-20, 20)},
            rotate=(-20, 20), 
            scale=(0.5, 0.9)
        ),
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
    ])
 
    # seq_det = seq
    print('start augment ' + img_path + ' (' + str(j + 1) + '/' + str(image_number) + ')')

    for x in range(num_of_image_to_generate):
      
        seq_det = seq.to_deterministic()
  
        image_aug = seq_det.augment_images([rgb_img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    
        # print coordinates before/after augmentation (see below)
        # use .x1_int, .y_int, ... to get integer coordinates
        #print('BB for :', x)
        # in case of multiple BB's
        for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbs_aug.bounding_boxes[i]
            #print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            #    i,
            #    before.x1, before.y1, before.x2, before.y2,
            #    after.x1, after.y1, after.x2, after.y2)
            #)

            # set new data for xml_objects
            xml_objects[i].find('bndbox').find('xmin').text = str(int(after.x1))
            xml_objects[i].find('bndbox').find('ymin').text = str(int(after.y1))
            xml_objects[i].find('bndbox').find('xmax').text = str(int(after.x2))
            xml_objects[i].find('bndbox').find('ymax').text = str(int(after.y2))
      
        image_before = bbs.draw_on_image(rgb_img, thickness=4)
        image_after = bbs_aug.draw_on_image(image_aug, thickness=4, color=[0, 0, 255])
  
        # PLOT image on stack
        # both = np.hstack((image_before, image_after))
        # f, ax = plt.subplots()
        # ax.set_title(img_path + str(x))
        # plt.imshow(both)
    
        # write image and xml
        # cv2.imwrite(img_out_folder + '/' + 'test.png', image_after)

        progress_number = '{0:05d}'.format(pic_num)
        img_out_name = fname2 + '-' + progress_number + fextension
        img_out_name_boxed = fname2 + '-' + progress_number + '-boxed' + fextension
        xml_out_name = fname2 + '-' + progress_number + '.xml'

        xml_filename = root.find('./filename')
        xml_filename.text = img_out_name

        xml_database = root.find('./source/database')
        xml_database.text = xml_database_info

        xml_path_tag = root.find('./path')
        xml_path_tag.text = img_out_folder + '/' + img_out_name

        xml_folder_tag = root.find('./folder')
        xml_folder_tag.text = img_out_folder 

        # print(img_out_name)
        # print(xml_out_name)

        # remove old element object
        for child in root.findall("object"):
            root.remove(child)

        # add modified object element  
        for item in xml_objects:
            # print(item)
            root.append(item)

        et.write( img_out_folder + '/' + xml_out_name)
        
        if write_boxed_images:
            cv2.imwrite(img_out_folder + '/' + img_out_name_boxed, image_after)

        # cv2.imwrite(img_out_folder + '/' + img_out_name, cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR))  
        jpeg_file_name = img_out_folder + '/' + img_out_name;
        new_img = PIL.Image.fromarray(image_aug) 
        new_img.save(jpeg_file_name, format="JPEG", quality=100)
        # cv2.imwrite(img_out_folder + '/' + img_out_name, image_aug)

        pic_num+=1
        # cv2.waitKey(20)
    print('augmented! ' + str(pic_num) + ' for ' + img_path)

print('End!')
exit()
# ---------------------------------------------------------------------

import xml.etree.ElementTree
# Open original file

et = xml.etree.ElementTree.parse('/content/imgsource/bg-asso.xml')

root = et.getroot();

bndbox = root.find("./object/")

print(bndbox)

xmin = root.find("./object/bndbox/xmin")
xmax = root.find("./object/bndbox/xmax")
ymin = root.find("./object/bndbox/ymin")
ymax = root.find("./object/bndbox/ymax")

print('xmin:', xmin.text)
print('xmax:', xmax.text)
print('ymin:', ymin.text)
print('ymax:', ymax.text)
		
# Append new tag: <a x='1' y='abc'>body text</a>
# new_tag = xml.etree.ElementTree.SubElement(et.getroot(), 'bndbox')
# new_tag.text = 'body text'
# new_tag.attrib['x'] = '1' # must be str; cannot be an int
# new_tag.attrib['y'] = 'abc'

ymax.text = '9999'

# Write back to file
#et.write('file.xml')
et.write('file_new.xml')

print(et)


"""
        root = ET.Element("root")
        doc = ET.SubElement(root, "doc")

        ET.SubElement(doc, "field1", name="blah").text = "some value1"
        ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"

        tree = ET.ElementTree(root)
        tree.write("filename.xml")



>>> for country in root.findall('country'):
...     rank = int(country.find('rank').text)
...     if rank > 50:
...         root.remove(country)


tree = ET.parse('myfile.xml')
root = tree.getroot()

for test in root.iter('test'):
    for stuff in test.findall('stuff'):
       test.remove(stuff)




"""