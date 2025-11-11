# import onnx

# onnx.utils.extract_model(
#             "best_dual_input_depth.onnx",
#             "best_dual_input_depth_extracted.onnx",
#             ["images_wide", "images_narrow"],
#             ["/model/model.23/cv4.0/cv4.0.2/Conv_output_0","/model/model.23/cv3.0/cv3.0.2/Conv_output_0","/model/model.23/cv2.0/cv2.0.2/Conv_output_0","/model/model.23/cv4.1/cv4.1.2/Conv_output_0","/model/model.23/cv3.1/cv3.1.2/Conv_output_0","/model/model.23/cv2.1/cv2.1.2/Conv_output_0","/model/model.23/cv4.2/cv4.2.2/Conv_output_0","/model/model.23/cv3.2/cv3.2.2/Conv_output_0","/model/model.23/cv2.2/cv2.2.2/Conv_output_0"]
#         )

# # /model/model.23/cv2.0/cv2.0.2/Conv_output_0 /model/model.23/cv3.0/cv3.0.2/Conv_output_0 /model/model.23/cv4.0/cv4.0.2/Conv_output_0 /model/model.23/cv2.1/cv2.1.2/Conv_output_0 /model/model.23/cv3.1/cv3.1.2/Conv_output_0 /model/model.23/cv4.1/cv4.1.2/Conv_output_0 /model/model.23/cv2.2/cv2.2.2/Conv_output_0 /model/model.23/cv3.2/cv3.2.2/Conv_output_0 /model/model.23/cv4.2/cv4.2.2/Conv_output_0
# /model/model.23/cv4.0/cv4.0.2/Conv_output_0 /model/model.23/cv3.0/cv3.0.2/Conv_output_0 /model/model.23/cv2.0/cv2.0.2/Conv_output_0 /model/model.23/cv4.1/cv4.1.2/Conv_output_0 /model/model.23/cv3.1/cv3.1.2/Conv_output_0 /model/model.23/cv2.1/cv2.1.2/Conv_output_0 /model/model.23/cv4.2/cv4.2.2/Conv_output_0 /model/model.23/cv3.2/cv3.2.2/Conv_output_0 /model/model.23/cv2.2/cv2.2.2/Conv_output_0


import onnx
onnx.utils.extract_model(
    # "/home/byounggun/ultralytics/runs/train/exp352/weights/best_dual_input_depth.onnx",
    # "/home/byounggun/ultralytics/runs/train/exp354/weights/best_dual_input_depth.onnx",
    # "/home/byounggun/ultralytics/runs/train/exp361/weights/best_dual_input_depth.onnx",
    "/home/byounggun/ultralytics/runs/finetune/katri_overfit2/weights/epoch35_dual_input_depth.onnx",
    # "/home/byounggun/ultralytics/runs/train/exp352/weights/best_dual_input_depth_extracted.onnx",
    # "/home/byounggun/ultralytics/runs/train/exp354/weights/best_dual_input_depth_extracted.onnx",
    # "/home/byounggun/ultralytics/runs/train/exp361/weights/best_dual_input_depth_extracted.onnx",
    "/home/byounggun/ultralytics/runs/finetune/katri_overfit2/weights/epoch35_dual_input_depth_extracted.onnx",
    ["images_wide", "images_narrow"],
    [
      # P3 (80x80): l, c, o
      "/model/model.23/cv2.0/cv2.0.2/Conv_output_0",  # DFL 64
      "/model/model.23/cv3.0/cv3.0.2/Conv_output_0",  # CLS 28
      "/model/model.23/cv4.0/cv4.0.2/Conv_output_0",  # DEP 1

      # P4 (40x40): l, c, o
      "/model/model.23/cv2.1/cv2.1.2/Conv_output_0",
      "/model/model.23/cv3.1/cv3.1.2/Conv_output_0",
      "/model/model.23/cv4.1/cv4.1.2/Conv_output_0",

      # P5 (20x20): l, c, o
      "/model/model.23/cv2.2/cv2.2.2/Conv_output_0",
      "/model/model.23/cv3.2/cv3.2.2/Conv_output_0",
      "/model/model.23/cv4.2/cv4.2.2/Conv_output_0",
    ]
)
