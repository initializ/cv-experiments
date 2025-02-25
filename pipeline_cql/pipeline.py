import time
import cv2
import queue
import threading
import numpy as np
import torch
from torchvision import transforms, models
from ultralytics import YOLO
import torch
import torch.nn as nn

class EnhancedCarTracker:
    def __init__(self):
        # initialize YOLO
        self.detection_model = YOLO("models/yolo12m.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        # get classifier model
        self.classify_model = self._load_brand_classifier("model/mobilenetv3_finetuned_half_v2.pth")
        self.classify_model.eval()
        
        # image pre_process
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # tracking related hyperparameter
        self.tracked_objects = {}
        self.next_id = 0
        self.point_radius = 3
        self.min_confidence = 0.5
        self.brand_cache = {}  # store brand
        
        # video thread
        self.cap = cv2.VideoCapture("xhs_car.mp4")
        self.frame_queue = queue.Queue(maxsize=50)
        
        # run video thread
        self.running = True
        threading.Thread(target=self._video_capture, daemon=True).start()

        self.class_names = [
            "Acura", "Alfa Romeo", "Aston Martin", "Audi", "Bentley",
            "BMW", "Bugatti", "Buick", "Cadillac", "Chevrolet",
            "Chrysler", "Citroen", "Daewoo", "Dodge", "Ferrari",
            "Fiat", "Ford", "Genesis", "GMC", "Honda",
            "Hudson", "Hyundai", "Infiniti", "Jaguar", "Jeep",
            "Kia", "Land Rover", "Lexus", "Lincoln", "Maserati",
            "Mazda", "Mercedes-Benz", "MG", "Mini", "Mitsubishi",
            "Nissan", "Oldsmobile", "Peugeot", "Pontiac", "Porsche",
            "Ram Trucks", "Renault", "Saab", "Studebaker", "Subaru",
            "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo"
        ]
        
        # display hyperparameter
        self.text_color = (255, 255, 255)  # white
        self.bg_color = (0, 0, 0)          # black
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_offset = 5
    
    def _load_brand_classifier(self, model_path):
        """Noted that this classfier should be loaded and modified the output layer to 50 classes since the current dataset I use have 50 car brands"""
        model = models.mobilenet_v3_large()
        model.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 50)  # 50 brands
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model.to("cuda" if torch.cuda.is_available() else "cpu")

    def _video_capture(self):
        """video capture thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_queue.put(frame)
            else:
                self.cap.release()
                break

    def _classify_car_brand(self, img_crop):
        """classifier"""
        with torch.no_grad():
            input_tensor = self.preprocess(img_crop).unsqueeze(0)
            device = next(self.classify_model.parameters()).device
            outputs = self.classify_model(input_tensor.to(device))
            _, pred = torch.max(outputs, 1)
            return self.class_names[pred.item()]  

    def _update_tracking(self, detections):
        current_ids = []
        for *xyxy, conf, cls in detections:
            if conf < self.min_confidence:
                continue
                
            x1, y1, x2, y2 = map(int, xyxy)
            center = ((x1+x2)//2, (y1+y2)//2)
            class_name = self.detection_model.names[int(cls)]
            
            # car truck and person, more category to be added
            if class_name not in ["car", "truck","person"]:
                continue
                
            # use 30 px as the distance threshold for tracking same object
            matched_id = None
            min_dist = 30
            for obj_id, obj in self.tracked_objects.items():
                if obj['class'] != class_name:
                    continue
                dist = np.linalg.norm(np.array(center) - np.array(obj['position']))
                if dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id
            
            # update the object
            if matched_id is not None:
                obj = self.tracked_objects[matched_id]
                obj['position'] = center
                obj['last_seen'] = time.time()
                
                # every 300 frame, redo car brand classification
                if obj.get('frame_count', 0) % 300 == 0:
                    img_crop = self.current_frame[y1:y2, x1:x2]
                    if img_crop.size > 0:
                        brand_name = self._classify_car_brand(img_crop)
                        obj['brand'] = brand_name  # brand_name
                obj['frame_count'] = obj.get('frame_count', 0) + 1

            
            else:
                # detect new car, do classification immediately
                new_id = self.next_id
                img_crop = self.current_frame[y1:y2, x1:x2]
                brand_name = self._classify_car_brand(img_crop) if img_crop.size > 0 else "Unknown"
                
                self.tracked_objects[new_id] = {
                    'position': center,
                    'class': class_name,
                    'brand': brand_name,  
                    'last_seen': time.time(),
                    'frame_count': 1
                }
                current_ids.append(new_id)
                self.next_id += 1
        
        # when car goes out of camera, clean its mark
        expired = [k for k,v in self.tracked_objects.items() 
                 if time.time()-v['last_seen'] > 0.1]
        for k in expired:
            del self.tracked_objects[k]

    def _draw_annotations(self, frame):
        """draw annotation"""
        for obj_id, obj in self.tracked_objects.items():
            x, y = obj['position']
            
       
            cv2.circle(frame, (x, y), self.point_radius, self.text_color, -1)
            
            info_text = f"{obj['class']} {obj.get('brand', 'Unknown')}"
            cv2.putText(
                frame, 
                info_text, 
                (x + self.text_offset, y - self.text_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.font_scale, 
                self.text_color, 
                self.font_thickness,
                lineType=cv2.LINE_AA
            )
        
        cv2.putText(
            frame, 
            f"Tracking: {len(self.tracked_objects)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6,  
            self.text_color, 
            1      
        )
        return frame

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                self.current_frame = self.frame_queue.get()
                
                # obj detection
                results = self.detection_model(self.current_frame, verbose=False)[0]
                
                # update
                self._update_tracking(results.boxes.data.cpu().numpy())
                
                # draw marks
                annotated_frame = self._draw_annotations(self.current_frame)
                
                # display
                cv2.imshow("Car Brand Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EnhancedCarTracker()
    tracker.run()