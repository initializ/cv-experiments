import time
import cv2
import queue
import threading
import numpy as np
import torch
from ultralytics import YOLO

'''
This scripts is only the logic of tracking the same car is the same car. no need to implement it.
Can modify this later on to make it track the same person.'''

class RealTimeTracker:
    def __init__(self):
        self.model = YOLO("models/yolo11s.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        self.tracked_objects = {}
        self.next_id = 0
        self.colors = np.random.randint(0, 255, (100, 3)) 
        self.cap = cv2.VideoCapture("xhs_car.mp4")
        self.frame_queue = queue.Queue(maxsize=50)
        
        self.running = True
        threading.Thread(target=self._video_capture).start()

    def _video_capture(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_queue.put(frame)
            else:
                self.cap.release()
                break

    def _update_tracking(self, detections):
        current_ids = []
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            center = ((x1+x2)//2, (y1+y2)//2)

            matched_id = None
            min_dist = 50 
            for obj_id, obj in self.tracked_objects.items():
                dist = np.linalg.norm(np.array(center) - np.array(obj['position']))
                if dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id
            
            if matched_id is not None:
                self.tracked_objects[matched_id] = {
                    'position': center,
                    'bbox': (x1, y1, x2, y2),
                    'last_seen': time.time()
                }
                current_ids.append(matched_id)
            else:
                self.tracked_objects[self.next_id] = {
                    'position': center,
                    'bbox': (x1, y1, x2, y2),
                    'last_seen': time.time()
                }
                current_ids.append(self.next_id)
                self.next_id += 1
        expired = [k for k,v in self.tracked_objects.items() 
                 if time.time()-v['last_seen'] > 2]
        for k in expired:
            del self.tracked_objects[k]

    def _draw_annotations(self, frame):

        for obj_id, obj in self.tracked_objects.items():
            x1, y1, x2, y2 = obj['bbox']
            color = self.colors[obj_id % 100].tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{obj_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1-20), (x1+w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            if 'trajectory' in obj:
                for i in range(1, len(obj['trajectory'])):
                    cv2.line(frame, 
                            obj['trajectory'][i-1], 
                            obj['trajectory'][i], 
                            color, 1)
        
        cv2.putText(frame, f"Tracking Objects: {len(self.tracked_objects)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        return frame

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                results = self.model(frame, verbose=False)[0]
                detections = results.boxes.data.cpu().numpy()
                
                self._update_tracking(detections)

                frame = self._draw_annotations(frame)
                
                cv2.imshow("Object Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = RealTimeTracker()
    tracker.run()