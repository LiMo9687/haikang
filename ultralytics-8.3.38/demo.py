from ultralytics import YOLO

model = YOLO("yolo11.yaml").load("yolo11l.pt")

results = model.train(data=r"D:\Code\PyCharm\ultralytics-8.3.38\datasets\VisDrone.yaml",
                      epochs=300,
                      imgsz=640,
                      cos_lr=True)

# model = YOLO(r'D:\Code\PyCharm\runs\detect\train20\weights\best.pt')
# results = model.predict(source=r'D:\Code\PyCharm\Yolov5-Deepsort-main\test2.mp4', save=True, show=False)
