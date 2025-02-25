### car_detection pipeline v1
### folders
model/ storing the stage 2 CNN car brands classifier
models/ store the Yolo model, it can acoomodate any yolo model
db/ currently only store a single data name file called coco,names, which is the dataset that yolo got pre-trained on
same_obj_algorithm only stores an algorithm framework.
pipeline.py is the code to run. 