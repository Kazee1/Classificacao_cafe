import os, cv2, math, random, numpy as np
import pymunk

# ================== CONFIG ==================
solo_img_dir = r"A:/Visao_computacional_projeto/Classificacao_cafe/USK-COFFEE_Dataset/test/images"
solo_lbl_dir = r"A:/Visao_computacional_projeto/Classificacao_cafe/USK-COFFEE_Dataset/test/labels"

output_base_dir = "synthetic_physics/"
os.makedirs(output_base_dir, exist_ok=True)
for sp in ["train","val","test"]:
    os.makedirs(os.path.join(output_base_dir, sp, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, sp, "labels"), exist_ok=True)

canvas_size = 640
splits = {"train": 800, "val": 100, "test": 100}
grains_per_img = (120, 200)

# Novas escalas realistas
SCALE_MIN, SCALE_MAX = 0.15, 0.25
GRAVITY = 1800
FRICTION = 0.9
ELASTICITY = 0.1
RAD_SHAPE_FACTOR = 0.4
MAX_SIM_STEPS = 2000
DT = 1/120.0
SLEEP_VEL = 5.0
WALL_THICK = 8

# ============== FUNÇÕES AUXILIARES ==============

def yolo_to_xyxy(label_line, img_w, img_h):
    cls, cx, cy, w, h = map(float, label_line.strip().split())
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return int(cls), x1, y1, x2, y2

def load_grain_crops():
    crops = []
    for fname in os.listdir(solo_img_dir):
        if not fname.lower().endswith((".jpg",".jpeg",".png")): 
            continue
        imgp = os.path.join(solo_img_dir, fname)
        lblp = os.path.join(solo_lbl_dir, os.path.splitext(fname)[0]+".txt")
        if not os.path.exists(lblp): 
            continue
        img = cv2.imread(imgp)
        if img is None: 
            continue
        H,W = img.shape[:2]
        with open(lblp) as f:
            for line in f:
                try:
                    _, x1,y1,x2,y2 = yolo_to_xyxy(line, W, H)
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(W,x2), min(H,y2)
                    if x2>x1 and y2>y1:
                        crop = img[y1:y2, x1:x2]
                        if crop.size>0:
                            crops.append(crop)
                except:
                    pass
    return crops

def make_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
    m = cv2.medianBlur(m, 3)
    return m

def resize_with_scale(img, scale):
    h,w = img.shape[:2]
    nh = max(6, int(h*scale))
    nw = max(6, int(w*scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def rotate_keep(img, angle_deg):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw, nh = int((h*sin + w*cos)), int((h*cos + w*sin))
    M[0,2] += (nw/2) - w/2
    M[1,2] += (nh/2) - h/2
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return out

def paste_with_mask(bg, patch, x, y):
    bg_h, bg_w = bg.shape[:2]
    patch_h, patch_w = patch.shape[:2]
    
    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = min(patch_w, bg_w - x)
    src_y2 = min(patch_h, bg_h - y)
    
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(bg_w, x + patch_w)
    dst_y2 = min(bg_h, y + patch_h)
    
    if src_x1 >= src_x2 or src_y1 >= src_y2 or dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return bg, None
    
    visible_patch = patch[src_y1:src_y2, src_x1:src_x2]
    mask = make_mask(visible_patch)
    
    bg_roi = bg[dst_y1:dst_y2, dst_x1:dst_x2]
    
    inv_mask = cv2.bitwise_not(mask)
    bg_masked = cv2.bitwise_and(bg_roi, bg_roi, mask=inv_mask)
    patch_masked = cv2.bitwise_and(visible_patch, visible_patch, mask=mask)
    combined = cv2.add(bg_masked, patch_masked)
    
    bg[dst_y1:dst_y2, dst_x1:dst_x2] = combined
    return bg, (dst_x1, dst_y1, dst_x2, dst_y2)

def make_tray_background(sz=640):
    base = np.full((sz,sz,3), random.randint(210,236), np.uint8)
    noise = np.random.normal(0, 7, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16)+noise, 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5,5), 0)
    yy,xx = np.mgrid[0:sz,0:sz]
    cx,cy = sz/2, sz/2
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2)/(sz*0.75)
    vignette = (1 - 0.12*r).clip(0.85,1.0)
    base = (base.astype(np.float32)*vignette[...,None]).astype(np.uint8)
    return base

# ============== CARREGAR GRÃOS ==============
grain_crops = load_grain_crops()
print(f"Total de grãos recortados: {len(grain_crops)}")
assert len(grain_crops)>0, "Não achei recortes. Confira os caminhos/labels."

# ============== LOOP DE GERAÇÃO ==============
for split, count in splits.items():
    for idx in range(count):
        # --- Física ---
        space = pymunk.Space()
        space.gravity = (0, GRAVITY)

        static = space.static_body
        walls = [
            pymunk.Segment(static, (0, canvas_size), (canvas_size, canvas_size), WALL_THICK),
            pymunk.Segment(static, (0, 0), (0, canvas_size), WALL_THICK),
            pymunk.Segment(static, (canvas_size, 0), (canvas_size, canvas_size), WALL_THICK)
        ]
        for w in walls:
            w.friction = FRICTION
            w.elasticity = ELASTICITY
        space.add(*walls)

        items = []
        N = random.randint(*grains_per_img)
        
        for i in range(N):
            g = random.choice(grain_crops).copy()
            scale = random.uniform(SCALE_MIN, SCALE_MAX)
            g = resize_with_scale(g, scale)

            angle = random.uniform(-180, 180)
            g_rot = rotate_keep(g, angle)

            gh, gw = g_rot.shape[:2]
            rad = max(gw, gh) * RAD_SHAPE_FACTOR
            mass = max(2.0, rad * 0.3)
            moment = pymunk.moment_for_circle(mass, 0, rad)
            body = pymunk.Body(mass, moment)
            
            cx_center = canvas_size / 2
            spread = 5
            x = cx_center + random.uniform(-spread, spread)
            y = -random.uniform(10, 30)
            body.position = (x, y)
            body.angular_velocity = random.uniform(-8, 8)
            body.velocity = (random.uniform(-40, 40), random.uniform(0, 80))

            shape = pymunk.Circle(body, rad)
            shape.friction = FRICTION
            shape.elasticity = ELASTICITY
            space.add(body, shape)

            items.append((g_rot, angle, body))

        # Simulação
        for step in range(MAX_SIM_STEPS):
            space.step(DT)
            
            if step % 20 == 0:
                moving_count = 0
                for _, _, body in items:
                    if (abs(body.velocity.x) + abs(body.velocity.y)) > SLEEP_VEL:
                        moving_count += 1
                if moving_count <= len(items) * 0.2:
                    break

        # --- Render ---
        bg = make_tray_background(canvas_size)
        annotations = []

        for patch, angle_deg, body in items:
            gh, gw = patch.shape[:2]
            cx, cy = body.position
            x = int(cx - gw/2)
            y = int(cy - gh/2)

            bg, bbox = paste_with_mask(bg, patch, x, y)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox

            # Descarta apenas se alguma coordenada estiver fora do canvas
            if x1 < 0 or y1 < 0 or x2 > canvas_size or y2 > canvas_size:
                continue

            cx_n = ((x1 + x2) / 2) / canvas_size
            cy_n = ((y1 + y2) / 2) / canvas_size
            bw_n = (x2 - x1) / canvas_size
            bh_n = (y2 - y1) / canvas_size

            annotations.append((0, cx_n, cy_n, bw_n, bh_n))

        # salva
        img_name = f"{split}_{idx:04d}.jpg"
        img_path = os.path.join(output_base_dir, split, "images", img_name)
        cv2.imwrite(img_path, bg)

        lbl_path = os.path.join(output_base_dir, split, "labels", img_name.replace(".jpg",".txt"))
        with open(lbl_path, "w") as f:
            for (cls, cx, cy, bw, bh) in annotations:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if (idx+1) % 10 == 0:
            avg_size = np.mean([(a[3] + a[4])/2 * canvas_size for a in annotations]) if annotations else 0
            print(f"[{split}] {idx+1}/{count} - {len(annotations)} grãos (avg: {avg_size:.1f}px)")

print("✅ Dataset sintético com tamanhos realistas gerado!")
