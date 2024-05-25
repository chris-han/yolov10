from ultralytics import YOLO
import torch
# import mlflow
from clearml import Task
# from upload_to_cloud import upload_file_to_azure
# from datetime import datetime

# yolo detect train data=coco.yaml model=/home/chris/repo/yolov10/hand_datasets/hand_data.yaml epochs=5 batch=256 imgsz=640 device=0

# def train_callback(trainer):
#     model_path = f"{trainer.best}"  # Assuming trainer has a save_dir attribute and model_name attribute
    # upload_file_to_azure(model_path)
    # model_name = trainer.best PosixPath('runs/detect/train14/weights/best.pt')
    # print(f"Model saved at {model_path}")
    # metrics = trainer.metrics
    # print(metrics)
    # step=trainer.epoch
    # print(step)

# Step 1: Creating a ClearML Task
# task1 = Task.init(project_name="yolo8-mt-lab", task_name="test_lab_register")
# task1 = Task.init(project_name="yolo8-mt-lab", task_name="train_tensorRT", output_uri="s3://39.107.203.165:9000/clearlm")
task = Task.init(project_name="yolov10-chris", task_name="train_yolov10",output_uri="s3://39.107.203.165:9000/clearlm")
# s3://39.107.203.165:9000/clearlm/yolo8-chris-lab/test_ali_cloud.86fec6a84582461e8dcefd5ece5d6182/models/best.pt
# output_model = OutputModel(task=task, framework="PyTorch")
# Step 2: Selecting the YOLOv8 Model
model_variant = "/home/chris/repo/yolov10/ultralytics/cfg/models/v10/yolov10n.yaml"
task.set_parameter("model_variant", model_variant)
# Specify the YOLO foundation model weight
# weights_path = "/home/chris/repo/yolov10/yolov10n.pt"

# Step 3: Loading the YOLOv8 Model
device = torch.device("cuda")
# model = YOLO(f'{model_variant}.pt').to(device)
model = YOLO(model_variant).to(device)

# Step 4: Setting Up Training Arguments
# model.add_callback("on_model_save",train_callback)
# model.add_callback("on_train_epoch_end",train_callback)
args = dict(data="/home/chris/repo/yolov10/hand_datasets/hand_data.yaml", epochs=60, batch=256, imgsz=640)
task.connect(args)
# Step 5: Initiating Model Training
results = model.train(**args)
task.close()
# task = Task.init(project_name="yolov10-chris", task_name="Export to TensorRT")
# # Export the model in TensorRT format
# # pip install onnx>=1.12.0 onnxsim>=0.4.33 onnxruntime-gpu
# # pip install nvidia-tensorrt
# out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, half=True)
# task.upload_artifact('TensorRT_FP16', out)
# task.close()

# Log the exported model to the task
# task.upload_artifact('TensorRT_FP16', out)
# Get the trainer pt file
# task2 = Task.create(project_name="yolo8-chris-lab", task_name="deploy_pt")
# preprocess_task = Task.get_task(task_id=task1.id)

# # Get the artifact's URI
# artifact = preprocess_task.artifacts #['best']
# artifact_uri = artifact.uri

# print("Artifact URI:", artifact_uri)

