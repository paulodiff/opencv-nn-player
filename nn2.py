# OpenCV nn player
# Usa opencv per aprire immagini video e webcam
# Usa Tensorflow per inferenza su modelli di rete


import numpy as np
import tensorflow as tf
import cv2 as cv
import gc # free memory
import simple_parser
import tensorflow as tf; print('TF version:', tf.__version__)



args_prototxt = 'c:/nn/mazzo_pascal_label_map.pbtxt'
args_weights = 'C:/nn/model/train1/frozen_inference_graph.pb'
args_image = 'c:/nn/sample_imgs/test-panno-verde-900x900.jpg'

NUM_CLASSES = 54
label_map = simple_parser.dict_from_pbtxt_file(args_prototxt)

print(label_map)


len_CLASSES = 54

COLORS = np.random.uniform(0, 255, size=(len_CLASSES, 3))

# Read the graph.
print('reading graph: ' + args_weights)
with tf.gfile.FastGFile(args_weights, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    print('reading image: ' + args_image)
    img = cv.imread(args_image)

    rows = img.shape[0]
    cols = img.shape[1]

    print(rows, cols)
    
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            label = "{}: {:.2f}%".format(label_map[classId],	score * 100)
            print(label)
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (255, 10, 10), thickness=2)
            yoffset = 5
            y2 = y - yoffset if y - yoffset > yoffset else y + yoffset
            cv.putText(img, label, (int(x), int(y2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classId], 2)

    sess.close()
    
del sess

cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey()

gc.collect() # FREE MEMORY

print('End!')