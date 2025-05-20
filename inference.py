from ultralytics import YOLO

# Load a model
model1 = YOLO("/home/byounggun/ultralytics/runs/detect/train15/weights/best.pt")  # pretrained YOLO11n model
# model2 = YOLO("/home/byounggun/ultralytics/runs/detect/train9/weights/best.pt")  # pretrained YOLO11n model
# model3 = YOLO("/home/byounggun/ultralytics/runs/detect/train10/weights/best.pt")  # pretrained YOLO11n model
# model4 = YOLO("/home/byounggun/ultralytics/runs/detect/train11/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
# results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects
#results images in folder
import glob
import os

# 'wide'로 시작하는 이미지 경로만 필터링
image_dir = "/home/byounggun/ultralytics/traffic_dataset/test_2/images"
wide_images = glob.glob(os.path.join(image_dir, "wide*"))

if wide_images:
    results1 = model1.predict(wide_images, save=True)
else:
    print("'wide'로 시작하는 이미지 파일을 찾을 수 없습니다.")
    results1 = []
# if wide_images:
#     results2 = model2.predict(wide_images, save=True)
# else:
#     print("'wide'로 시작하는 이미지 파일을 찾을 수 없습니다.")
#     results2 = []

# if wide_images:
#     results3 = model3.predict(wide_images, save=True)
# else:
#     print("'wide'로 시작하는 이미지 파일을 찾을 수 없습니다.")
#     results3 = []

# if wide_images:
#     results4 = model4.predict(wide_images, save=True)
# else:
#     print("'wide'로 시작하는 이미지 파일을 찾을 수 없습니다.")
#     results4 = []

# Process results list
# print("results1", len(results1))
# for result in results1:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result1.jpg")  # save to disk

# print("results2", len(results2))
# for result in results2:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result2.jpg")  # save to disk

# print("results3", len(results3))
# for result in results3:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result3.jpg")  # save to disk

# print("results4", len(results4))
# for result in results4:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result4.jpg")  # save to disk