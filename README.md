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

### Dataset

Download our Dataset [crowd-counting-dataset-w3o7w](https://universe.roboflow.com/crowd-dataset/crowd-counting-dataset-w3o7w), while exporting the dataset select **YOLO v5 PyTorch** Format.

![our-dataset](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/7c574121-7eb5-450c-a61d-d259643d22fb)


## Preweight
The link is [yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt)


### Training
Train your model on **crowd-counting-dataset-w3o7w** dataset.
```shell
python train.py --img 416
                --batch 16
                --epochs 200
                --data {dataset.location}/data.yaml
                --cfg models/yolo-crowd.yaml    
                --weights yolov5s.pt      
                --name yolo_crowd_results
                --cache
```

## Postweight
The link is [yolo-crowd.pt](https://drive.google.com/file/d/1xxXVCzseuzmHv7NoMQ03RVU_tDisWXjM/view?usp=sharing)
If you want to have more inference speed try to install TensorRt and use this vesion [yolo-crowd.engine](https://drive.google.com/file/d/1-189sscpNZBFaSHOz7dnEgAaFeUALiow/view?usp=sharing)


### Test
```shell
python detect.py --weights yolo-crowd.pt --source 0                               # webcam
                                                  img.jpg                         # image
                                                  vid.mp4                         # video
                                                  screen                          # screenshot
                                                  path/                           # directory
                                                  list.txt                        # list of images
                                                  list.streams                    # list of streams
                                                  'path/*.jpg'                    # glob
                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```



## Results

![results-yolo-crowd](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/9e2d18ce-aaf6-4a20-91f0-d8d1eb88728c)


## Finetune
see in *[https://github.com/ultralytics/yolov5/issues/607](https://github.com/ultralytics/yolov5/issues/607)*
```shell
# Single-GPU
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve

# Multi-GPU
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > evolve_gpu_$i.log &
done

# Multi-GPU bash-while (not recommended)
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  "$(while true; do nohup python train.py... --device $i --evolve 1 > evolve_gpu_$i.log; done)" &
done
```

## Reference
*[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)*
*[https://github.com/deepcam-cn/yolov5-face](https://github.com/Krasjet-Yu/YOLO-FaceV2)*
*[https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)*
*[https://github.com/dongdonghy/repulsion_loss_pytorch](https://github.com/dongdonghy/repulsion_loss_pytorch)*
*[https://github.com/zaki1003/YOLO-CROWD]*
