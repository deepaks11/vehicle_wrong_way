import os
import cv2
import threading
import queue

import torch
from ultralytics import YOLO
from multiprocessing.pool import ThreadPool
from main import PersonWrongWay
queue_name = queue.Queue()
pool = ThreadPool(processes=1)
root = os.getcwd()


class VideoCapture:

    def __init__(self, rtsp_url):

        self.cap = cv2.VideoCapture(rtsp_url)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True  # Set the thread as daemon to stop it when the main program exits
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class PlayVideo:

    def __init__(self, source, window_name, q):
        self.cap_line = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO("yolo11s.pt")
        self.model.to(self.device)
        self.yolo_detector = PersonWrongWay(self.model)
        self.detections = None
        self.image = None
        self.obj = None
        self.frame = None
        self.resized_img = None
        self.source = source
        self.window_name = window_name
        self.q_img = q

    def vdo_cap(self):

        try:

            if self.source.startswith("rtsp"):
                self.cap_line = VideoCapture(self.source)
            else:
                self.cap_line = cv2.VideoCapture(self.source)

            while True:
                # if you want to read any different video file format just add below
                if self.source.endswith((".mp4", ".avi")):
                    ret, self.image = self.cap_line.read()

                else:
                    self.image = self.cap_line.read()

                self.q_img.put(self.image)
                # passing frame to object detection and tracking
                frame = pool.apply_async(self.yolo_detector.predict, (self.q_img,)).get()
                self.resized_img = cv2.resize(frame, (1080, 720))

                cv2.imshow(self.window_name, self.resized_img)
                if cv2.waitKey(1) == ord('q'):  # Exit if 'q' is pressed
                    break
        except Exception as e:
            print(e)


if __name__ == "__main__":
    urls = [

        {"name": "rtsp1", "url": r"VechileTripWire.mp4"},
        # {"name": "rtsp2", "url": r"rtsp://admin:Admin123$@10.11.25.60:554/stream1"},
        # {"name": "rtsp3", "url": r"rtsp://admin:Admin123$@10.11.25.64:554/stream1"},
        # # {"name": "rtsp4", "url": r"rtsp://admin:Admin123$@10.11.25.57:554/stream1"},
        # {"name": "rtsp5", "url": r"rtsp://admin:Admin123$@10.11.25.57:554/stream1"},
        # {"name": "rtsp6", "url": r"rtsp://admin:Admin123$@10.11.25.51:554/stream1"},
        # {"name": "rtsp7", "url": r"rtsp://admin:Admin123$@10.11.25.52:554/stream1"},
        # {"name": "rtsp8", "url": r"rtsp://admin:Admin123$@10.11.25.53:554/stream1"},
        # {"name": "rtsp9", "url": r"rtsp://admin:Admin123$@10.11.25.54:554/stream1"},
        # {"name": "rtsp10", "url": r"rtsp://admin:Admin123$@10.11.25.55:554/stream1"},
        # {"name": "rtsp11", "url": r"rtsp://admin:Admin123$@10.11.25.59:554/stream1"},
        # {"name": "rtsp12", "url": r"rtsp://admin:Admin123$@10.11.25.63:554/stream1"}
    ]
    queue_list = []
    threads = []
    for i in urls:
        url = i['url']
        name = i["name"]
        queue_name.name = name
        queue_list.append(queue_name)
        td = threading.Thread(
            target=PlayVideo(url, name, queue_name).vdo_cap)
        td.start()
        threads.append(td)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()