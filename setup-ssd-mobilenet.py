# From model zoo
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
import os

!wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz -O /content/zoo_models/ssd_mobilenet.tar.gz
directory = '/content/zoo_models'
os.chdir(directory)
!pwd
!tar xzf ssd_mobilenet.tar.gz



# set ...


# generate config file
file = '/content/models/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config'
f_in = open(file)
f_out = open('/content/data_set_out/ssd_mobilenet_v1_coco_2018_01_28.config', 'w')
for line in f_in:
  #print(line)
  if 'num_classes' in line:
    line = line.replace("90", "54")
    # line = line.strip()
    # line = line.replace("\n", "")

  if 'PATH_TO_BE_CONFIGURED/model.ckpt' in line:
    line = line.replace("PATH_TO_BE_CONFIGURED/model.ckpt", 
                        "/content/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt")
    
  if 'PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100' in line:
    line = line.replace("PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100", 
                        "/content/data_set_out/data_train.record-?????-of-00100")
    
  if 'PATH_TO_BE_CONFIGURED/mscoco_val.record-?????-of-00010' in line:
    line = line.replace("PATH_TO_BE_CONFIGURED/mscoco_val.record-?????-of-00010", 
                        "/content/data_set_out/data_val.record-?????-of-00100")
        
  if 'PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt' in line:
    line = line.replace("PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt", 
                        "/content/data_set/mazzo_pascal_label_map.pbtxt")
    

  if line:
    print(line)
    f_out.write(line)
  
f_in.close()
f_out.close()