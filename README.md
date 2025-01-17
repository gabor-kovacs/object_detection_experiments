# Object Detection

## Downloading a dataset

See `get_dataset.ipynb`.

## YOLOv3

See `yolov3/yolov3.ipynb`.

48 epochs completed in 0.181 hours.
Optimizer stripped from runs/detect/train3/weights/last.pt, 207.8MB
Optimizer stripped from runs/detect/train3/weights/best.pt, 207.8MB

Validating runs/detect/train3/weights/best.pt...
Ultralytics 8.3.62 🚀 Python-3.10.14 torch-2.3.1 CUDA:0 (GRID A100-20C, 20476MiB)
YOLOv3 summary (fused): 226 layers, 103,669,637 parameters, 0 gradients, 282.2 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:02<00:00,  1.64it/s]
                   all        127        909      0.732      0.693      0.727      0.438
                  fish         63        459      0.765      0.757       0.81       0.45
             jellyfish          9        155      0.832      0.845      0.896      0.511
               penguin         17        104      0.614       0.75      0.656      0.305
                puffin         15         74      0.687      0.541      0.577      0.286
                 shark         28         57      0.744      0.613       0.66       0.44
              starfish         17         27      0.604      0.679      0.663      0.491
              stingray         23         33       0.88      0.665      0.829      0.579
Speed: 0.1ms preprocess, 4.1ms inference, 0.0ms loss, 4.5ms postprocess per image
Results saved to runs/detect/train3

## YOLOv11

See `yolov11/yolov11.ipynb`.

46 epochs completed in 0.242 hours.
Optimizer stripped from runs/detect/train/weights/last.pt, 51.2MB
Optimizer stripped from runs/detect/train/weights/best.pt, 51.2MB

Validating runs/detect/train/weights/best.pt...
Ultralytics 8.3.62 🚀 Python-3.10.14 torch-2.3.1 CUDA:0 (GRID A100-20C, 20476MiB)
YOLO11l summary (fused): 464 layers, 25,284,709 parameters, 0 gradients, 86.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:02<00:00,  1.49it/s]
                   all        127        909      0.759      0.714      0.746      0.459
                  fish         63        459      0.769      0.763      0.786      0.464
             jellyfish          9        155      0.772      0.839       0.89      0.524
               penguin         17        104      0.506       0.74      0.627      0.308
                puffin         15         74      0.764      0.438      0.587      0.259
                 shark         28         57      0.743      0.559      0.629      0.413
              starfish         17         27      0.943      0.815      0.846      0.629
              stingray         23         33      0.818      0.848      0.856      0.615
Speed: 0.1ms preprocess, 9.4ms inference, 0.0ms loss, 4.1ms postprocess per image
Results saved to runs/detect/train

## RT-DETR

See `rt-detr/rt-detr.ipynb`.

47 epochs completed in 0.184 hours.
Optimizer stripped from runs/detect/train/weights/last.pt, 66.2MB
Optimizer stripped from runs/detect/train/weights/best.pt, 66.2MB

Validating runs/detect/train/weights/best.pt...
Ultralytics 8.3.62 🚀 Python-3.10.14 torch-2.3.1 CUDA:0 (GRID A100-20C, 20476MiB)
rt-detr-l summary: 502 layers, 31,998,125 parameters, 0 gradients, 103.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  2.12it/s]
                   all        127        909      0.803      0.736      0.777      0.512
                  fish         63        459      0.842      0.734      0.822      0.492
             jellyfish          9        155      0.864      0.923      0.937      0.542
               penguin         17        104      0.709      0.683      0.711      0.381
                puffin         15         74      0.763      0.566      0.682      0.354
                 shark         28         57      0.809      0.594       0.61      0.449
              starfish         17         27      0.848      0.741      0.779      0.645
              stingray         23         33      0.789      0.909      0.896      0.726
Speed: 0.1ms preprocess, 3.7ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/detect/train


## Speed

See `speed.ipynb`.


```md
Inference Speed Results (FPS):
------------------------------------------------------------
Model                          Mean FPS        Std Dev        
------------------------------------------------------------
YOLOv3_pytorch                            74.0             8.5
YOLOv3_torchscript                        74.3             5.4
YOLOv3_onnx                               66.4            10.9
YOLOv3_tensorrt                          100.9             6.7
YOLOv11_pytorch                           47.3             4.0
YOLOv11_torchscript                       68.5             8.8
YOLOv11_onnx                              67.1             9.7
YOLOv11_tensorrt                         109.2            40.5
RT-DETR_pytorch                           23.4             2.0
RT-DETR_onnx                              15.6             6.7
RT-DETR_tensorrt                          93.8            12.5
```

CPP RT-DETR torchscript: 
Average time: 14.4691 ms
Standard deviation: 7.8112 ms
FPS: 69.1128