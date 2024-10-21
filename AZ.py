from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.pt")

    # Train the model
    train_results = model.train(
        data="D:/saisagarsp/counts and classification.v1i.yolov8/data.yaml",
        epochs=100,
        imgsz=640,
        device="0"
    )


    metrics = model.val()


    results = model("D:/saisagarsp/counts and classification.v1i.yolov8/test/images/1729423941905_jpg.rf.967fb084b41d55dd7ee3f1dcc66b515e.jpg")
    results[0].plot()

    # Export
    path = model.export(format="onnx")
