import cv2
import numpy as np
import supervision as sv
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class PersonWrongWay:

    def __init__(self, model):

        self.frame = None
        self.model = model
        self.detections = None
        self.tracked_ids = {}
        self.detected_image_image_path = None
        self.original_image_path = None
        self.alarm_list = None
        self.alarm_flag = None
        self.wrong_way_flag = False
        self.xyxy = None
        self.movement_direction = ""
        self.corner_annotator = sv.BoxCornerAnnotator()

    def find_direction(self, tracker_id, xyxy, direction="Backward", movement_threshold=7):
        try:

            x_min, y_min, x_max, y_max = xyxy
            person_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

            # Initialize tracking for new persons
            if tracker_id not in self.tracked_ids:

                self.tracked_ids[tracker_id] = {"prev_position": person_center, "direction": "Stationary"}
                return  # Exit function since no movement to compare for new person

            # Get previous position
            prev_position = self.tracked_ids[tracker_id]["prev_position"]

            # Calculate movement by comparing current center with the previous center
            delta_x = person_center[0] - prev_position[0]
            delta_y = person_center[1] - prev_position[1]

            # Check if movement exceeds threshold; if not, consider stationary
            if abs(delta_x) < movement_threshold and abs(delta_y) < movement_threshold:
                self.tracked_ids[tracker_id]["direction"] = "Stationary"
                return  # No significant movement, exit the function

            # Determine direction based on movement in x and y axes
            if abs(delta_x) > abs(delta_y):  # Horizontal movement is dominant
                if delta_x > 0 and direction == "Right":

                    self.frame = cv2.putText(self.frame, "wrong way",
                                             (int(person_center[0]), int(person_center[1])),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                             (0, 0, 255), 2)

                elif delta_x < 0 and direction == "Left":

                    self.frame = cv2.putText(self.frame, "wrong way",
                                             (int(person_center[0]), int(person_center[1])),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                             (0, 0, 255), 2)

            else:  # Vertical movement is dominant
                if delta_y > 0 and direction == "Backward":

                    self.frame = cv2.putText(self.frame, "wrong way",
                                             (int(person_center[0]), int(person_center[1])),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                             (0, 0, 255), 2)

                elif delta_y < 0 and direction == "Forward":

                    self.frame = cv2.putText(self.frame, "wrong way",
                                             (int(person_center[0]), int(person_center[1])),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                             (0, 0, 255), 2)

            # Update tracked info with the new center and direction
            self.tracked_ids[tracker_id]["prev_position"] = person_center
            self.tracked_ids[tracker_id]["direction"] = self.movement_direction

        except Exception as er:
            print(f"Error: {er}")

    def predict(self, q_img):
        try:
            directions = "Left"
            self.alarm_list = []
            self.frame = q_img.get()

            result = self.model.track(source=self.frame, conf=0.5, classes=[2, 3, 7], persist=True, verbose=False)
            result = result[0]

            self.detections = sv.Detections.from_ultralytics(result)

            if self.detections.tracker_id is not None:

                for self.xyxy, mask, confidence, class_id, tracker_id, class_name in self.detections:

                    self.find_direction(tracker_id, self.xyxy, directions)
                    self.corner_annotator.annotate(scene=self.frame, detections=self.detections)

                return self.frame
            else:
                return self.frame
        except Exception as er:
            print(er)

