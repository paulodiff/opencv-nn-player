# check TFRecord
# deve essere avviato da \models\research\


import tensorflow as tf
from google.protobuf.json_format import MessageToJson

tfrecord_filename = 'C:/nn/dataset/tfrecord/poker_train_000.tfrecord'
for example in tf.python_io.tf_record_iterator(tfrecord_filename):
    result = tf.train.Example.FromString(example)
    jsonMessage = MessageToJson(tf.train.Example.FromString(example))
    print(jsonMessage."image/shape")
    exit()