### car_detection pipeline v1

### Folders

- **model/**  
  Storing the stage 2 CNN car brands classifier

- **models/**  
  Store the Yolo model. It can accommodate any Yolo model

- **db/**  
  Currently only stores a single data name file called `coco.names`, which is the dataset that Yolo got pre-trained on

- **same_obj_algorithm.py**  
  Only stores an algorithm framework, no need to run

- **pipeline.py**  
  The code to run
