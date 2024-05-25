from ultralytics import YOLO
import torch
# import mlflow
from clearml import Task


# Step 1: Creating a ClearML Task
# task1 = Task.init(project_name="yolo8-mt-lab", task_name="test_lab_register")
# task1 = Task.init(project_name="yolo8-mt-lab", task_name="train_tensorRT", output_uri="s3://39.107.203.165:9000/clearlm")
task = Task.init(project_name="yolov-chris", task_name="train_yolov8-hand",output_uri="s3://39.107.203.165:9000/clearlm")
# s3://39.107.203.165:9000/clearlm/yolo8-chris-lab/test_ali_cloud.86fec6a84582461e8dcefd5ece5d6182/models/best.pt
# output_model = OutputModel(task=task, framework="PyTorch")
# Step 2: Selecting the YOLOv8 Model
model_variant = "yolov8n.yaml"
task.set_parameter("model_variant", model_variant)

# Specify the YOLO foundation model weightconda env remove --name your_env_name
# weights_path = "/home/chris/repo/yolov10/yolov10n.pt"

# Step 3: Loading the YOLOv8 Model
device = torch.device("cuda")
# model = YOLO(f'{model_variant}.pt').to(device)
model = YOLO(model_variant).to(device)

# Step 4: Setting Up Training Arguments
# model.add_callback("on_model_save",train_callback)
# model.add_callback("on_train_epoch_end",train_callback)
args = dict(data='/home/chris/repo/yolov10/EgoHands-Public.v9-generic-fast2.yolov8/data.yaml', epochs=3, batch=16, imgsz=416)
task.connect(args)
# Step 5: Initiating Model Training
results = model.train(**args)
model('/home/chris/repo/yolov10/examples/cooler.png')
task.close()

