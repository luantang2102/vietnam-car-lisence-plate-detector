from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv
from constant import input

car_model = os.path.join('.', 'runs', 'detect', 'car-train', 'weights', 'last.pt')
license_model = os.path.join('.', 'runs', 'detect', 'license-plate-train-3', 'weights', 'last.pt')

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO(car_model)
license_plate_detector = YOLO(license_model)

# load video
cap = cv2.VideoCapture(input)

vehicles = [0, 1, 3]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if ret:

        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        # print("track id:" + str(track_ids))
        # for j in range(len(track_ids)):
        #     xcar, ycar, xcar, ycar, car_id = track_ids[j]
        #     results[frame_nmr][car_id] = {'car': {'bbox': [xcar, ycar, xcar, ycar]},
        #                                   'license_plate': {'bbox': [0, 0, 0, 0],
        #                                                     'text': "",
        #                                                     'bbox_score': 0,
        #                                                     'text_score': 0}}

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            print("get car: ")
            print(xcar1, ycar1, xcar2, ycar2, car_id)


            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Check if the license_plate_crop is not empty
                if license_plate_crop.size == 0:
                    continue
                # cv2.imwrite("license_plate_crop_" + str(frame_nmr) + ".jpg", license_plate_crop)
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
# write results
write_csv(results, 'output/test.csv')
