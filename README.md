# YOLO-CROWD

**YOLO-CROWD** is a lightweight crowd counting and face detection model based on YOLOv5s, optimized for edge devices. It addresses challenges like face occlusion and varying face scales in crowded scenarios.

## Description

Recent advancements in deep learning for face and crowd detection can be categorized into one-stage detectors (e.g., YOLO) and two-stage detectors (e.g., Faster R-CNN). YOLO-based algorithms are widely used due to their balance between accuracy and speed. However, face occlusion in crowded scenarios remains a challenge. 

**YOLO-CROWD** improves on this by introducing:
- **RFE (Receptive Field Enhancement)** to improve small face detection
- **NWD Loss** to enhance the model's sensitivity to small object deviations
- **Repulsion Loss** to minimize face occlusion
- **SEAM Attention Module** for better focus

Inference speed: **10.1 ms**  
Model size: **461 layers**, **18,388,982 parameters**

## Demo

### Images

![test-yolo-crowd](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/6aed4956-1da5-4b98-ae8a-e7d9574b4054)

![Screenshot from 2023-04-07 15-49-11](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/e435d92b-42f2-4152-bcad-b72268db8d0e)

![Screenshot from 2023-04-07 15-48-52](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/2b5e3273-a697-472c-a201-0b23e5b2faa6)

### Videos
- Without labels: [Watch](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/b0a57b00-ae72-4a5c-ad68-442be1889e0a)
- With labels (name + conf): [Watch](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/44753430-c5ef-4c15-80c7-e0f328670aac)

## Performance Comparison

| Model         | mAP@0.5 | mAP@0.5-0.95 | Precision | Recall | Box Loss | Object Loss | Inference Time |
|---------------|---------|--------------|-----------|--------|----------|-------------|----------------|
| YOLOv5s       | 39.4    | 0.15         | 0.754     | 0.382  | 0.120    | 0.266       | **7 ms**       |
| YOLO-CROWD    | **43.6**| **0.158**    | **0.756** | **0.424**| **0.091** | **0.158**   | 10.1 ms        |

## Environment Setup

1. Create a Python Virtual Environment:  
   ```shell
   conda create -n {name} python=x.x
