# Genera i TFRecord per il training della rete
# deve essere avviato da \models\research\

import sys
import random
import PIL
from PIL import Image
import os
import io
import numpy as np
import hashlib
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from sklearn.model_selection import train_test_split

# from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
# from datasets.pascalvoc_common import VOC_LABELS

DIRECTORY_HOME = 'c:/nn/dataset/'
DIRECTORY_ANNOTATIONS = 'dest_small/'
DIRECTORY_IMAGES = DIRECTORY_ANNOTATIONS
TFRECORD_PREFIX = 'poker'
TFRECORD_SHUFFLE = True

DIRECTORY_TFRECORD = 'c:/nn/dataset/tfrecord2'
LABEL_MAP_PBTXT = 'c:/nn/mazzo_pascal_label_map.pbtxt'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 100

# label map read

label_map_dict = label_map_util.get_label_map_dict(LABEL_MAP_PBTXT)

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _process_image(directory, name):
    """Process a image and annotation file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    # print('read image file:', filename)
    # image_data = tf.gfile.FastGFile(filename, 'r').read()
    
    '''
    with tf.gfile.GFile(filename, 'rb') as fid:
      encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image_data = PIL.Image.open(encoded_jpg_io)
    '''

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    # print('read xml   file:', filename)
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # print(shape)
    
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        # print('label:', label, label_map_dict[label] )
                
        # class_name = get_class_name_from_filename(data['filename'])
        # classes_text.append(class_name.encode('utf8'))
        # classes.append(label_map_dict[class_name])
        
        labels.append(int(label_map_dict[label]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
        # print(bbox)
    return encoded_jpg, shape, bboxes, labels, labels_text, difficult, truncated

def _convert_to_example(name, image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.
    Args:
      name: imagename
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'

    #print(xmin)
    #print(xmax)
    #print(ymin)
    #print(ymax)

    '''
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),

                        'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated)
    '''

    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/filename': dataset_util.bytes_feature(name.encode('utf-8')),
            'image/source_id': dataset_util.bytes_feature(name.encode('utf-8')),
            'image/encoded': bytes_feature(image_data),
            'image/format': bytes_feature(image_format),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(labels_text),
            'image/object/class/label': dataset_util.int64_list_feature(labels)
            }))
    return example

def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.
    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    # print(dataset_dir, name)
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(name, image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def go_run(dataset_dir, output_dir, name='poker_train', shuffling=False):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    X_train, X_test = train_test_split(filenames, test_size=0.33, random_state=42)
    # print(X_train)
    print(len(X_train))
    # print(X_test)
    print(len(X_test))

    print('Building train dataset')

    # Process dataset train files.
    i = 0
    fidx = 0
    count_files = len(X_train)
    while i < len(X_train):
        tf_filename = _get_output_filename(output_dir, 'train-' + name, fidx)
        # print(tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(X_train) and j < SAMPLES_PER_FILES:
                # print('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                filename = X_train[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                
                i += 1
                j += 1
            fidx += 1
        print(str(i) + '/' + str(count_files))
        

    # Process dataset test files.
    print('Building test dataset')
    i = 0
    fidx = 0
    count_files = len(X_test)
    while i < len(X_test):
        tf_filename = _get_output_filename(output_dir, 'test-' + name, fidx)
        # print(tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(X_test) and j < SAMPLES_PER_FILES:
                # print('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                filename = X_test[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                
                i += 1
                j += 1
            fidx += 1
        print(str(i) + '/' + str(count_files))
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')
    
# run tfrecord   
go_run(DIRECTORY_HOME, DIRECTORY_TFRECORD, TFRECORD_PREFIX, TFRECORD_SHUFFLE)