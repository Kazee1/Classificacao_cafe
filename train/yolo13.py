from multiprocessing import freeze_support
from ultralytics import YOLO
def main():
    model = YOLO('yolov13n.pt')

    # Train the model
    results = model.train(
    data="A:/Visao_computacional_projeto/USK-COFFEE_Dataset/data.yaml",
    epochs=600, 
    batch=4, 
    imgsz=640,
    device="0",
    )

if __name__ == '__main__':
    freeze_support() #windows roda isso 
    main()