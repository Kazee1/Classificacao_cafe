import os
import cv2

# Caminho do dataset sintético gerado
dataset_dir = "synthetic_physics/train/images"
labels_dir = "synthetic_physics/train/labels"

# Pega as 10 primeiras imagens
img_files = sorted(os.listdir(dataset_dir))[:10]

for img_file in img_files:
    img_path = os.path.join(dataset_dir, img_file)
    lbl_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # Lê as bounding boxes
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                
                # Converte de YOLO para coordenadas xyxy
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)

                # Desenha retângulo
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostra a imagem
    cv2.imshow("Bounding Boxes", img)
    key = cv2.waitKey(0)  # pressione qualquer tecla para ir para a próxima
    if key == 27:  # ESC para sair
        break

cv2.destroyAllWindows()
