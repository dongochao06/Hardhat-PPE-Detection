import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import tensorflow as tf
import pandas as pd

# PHẦN 1: TIỀN XỬ LÝ VÀ TẠO BĂNG CHUYỀN (ĐÃ NÂNG CẤP CHO HARDHAT)
# ==========================================
def Convert_csv_to_number(input_csv_path):
    df = pd.read_csv(input_csv_path)
    print("DANH SÁCH CÁC CỘT TRONG CSV LÀ:", df.columns.tolist())

    # 1. Dịch tên Class (Chữ) thành Mã (Số từ 0-9)
    class_dict = {
        'Hardhat': 0, 'Mask': 1, 'NO-Hardhat': 2, 'NO-Mask': 3, 
        'NO-Safety_Vest': 4, 'Person': 5, 'Safety_Cone': 6, 
        'Safety_Vest': 7, 'machinery': 8, 'vehicle': 9
    }
    df['class_id'] = df['class'].map(class_dict)

    # 2. Quy đổi [xmin, ymin, xmax, ymax] thành [x_center, y_center, w, h] 
    df['x_center'] = (df['xmin'] + df['xmax']) / 2.0
    df['y_center'] = (df['ymin'] + df['ymax']) / 2.0
    df['w'] = df['xmax'] - df['xmin']
    df['h'] = df['ymax'] - df['ymin']

    # Gộp 4 con số này thành một mảng (list) trong cột bbox
    df['bbox'] = df[['x_center', 'y_center', 'w', 'h']].values.tolist()

    # 3. GỘP DÒNG (Groupby): Gom tất cả các Bounding Box của cùng 1 bức ảnh vào 1 hàng duy nhất
    df_grouped = df.groupby('img_path').agg({
        'bbox': list,      # Bọc tất cả bbox lại thành [[box1], [box2],...]
        'class_id': list   # Bọc tất cả class lại thành [0, 5, 5, ...]
    }).reset_index()

    return df_grouped

def create_dataset_from_dataframe(dataframe):
    # Biến các list gom cục ở trên thành Tensor răng cưa (Ragged Tensor) để AI đọc
    tensor_bbox = tf.ragged.constant(dataframe['bbox'].tolist(), ragged_rank=1) 
    
    # Đã gỡ bỏ lambda i-1 vì class_id đã chuẩn từ 0-9
    tensor_class_id = tf.ragged.constant(dataframe['class_id'].tolist()) 
    
    tensor_path = tf.constant(dataframe['img_path'].tolist())
    dataset = tf.data.Dataset.from_tensor_slices((tensor_path, tensor_bbox, tensor_class_id))
    return dataset

# PHẦN 2: XỬ LÝ ĐIỂM ẢNH (PIXEL)
# ==========================================
def read_img_label(path, bbox, class_id):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    return img, bbox, class_id

def resize_and_padding_img(img, bbox, class_id, target_size):
    shape = tf.cast(tf.shape(img)[:2], dtype=tf.float32)
    img_h = shape[0]
    img_w = shape[1]
    ratio = tf.cast(target_size, tf.float32) / tf.math.maximum(img_w, img_h)
    new_w = tf.cast(img_w * ratio, tf.int32)
    new_h = tf.cast(img_h * ratio, tf.int32)
    resize_img = tf.image.resize(img, [new_h, new_w])
    img = tf.image.pad_to_bounding_box(
        resize_img, offset_height=0, offset_width=0, 
        target_height=target_size, target_width=target_size
    )
    bbox = bbox * ratio
    return img, bbox, class_id

# PHẦN 3: TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION)
# ==========================================
def data_augment(img, bbox, class_id):
    target_size = tf.cast(tf.shape(img)[1], tf.float32)
    flip_cond = tf.random.uniform([]) > 0.5
    img = tf.cond(flip_cond, lambda: tf.image.flip_left_right(img), lambda: img)
    
    def flip_boxes(b):
        x_center, y_center, w, h = tf.split(b, 4, axis=-1)
        x_center = target_size - x_center
        return tf.concat([x_center, y_center, w, h], axis=-1)
    
    bbox = tf.cond(flip_cond, lambda: flip_boxes(bbox), lambda: bbox)
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, bbox, class_id

# PHẦN 4: GẮN NHÃN (LABEL ENCODING) 
# ==========================================
def get_label_encode(point):
    """
    Xét các điểm mà anchor point đưa ra
    """
    def label_encode(img, gt_box, gt_class):
        # Kiểm tra ảnh đầu vào có gt_box không
        if tf.shape(gt_box)[0] == 0:
            target_box = tf.zeros_like(point) 
            target_class = tf.fill([tf.shape(point)[0], 1], -1.0) 
            y_true = tf.concat([target_box, target_class], axis=-1) 
            return img, y_true
            
        # Bước 1.1: Tách tọa độ point
        pc_x = tf.expand_dims(point[:, 0], 1) 
        pc_y = tf.expand_dims(point[:, 1], 1)

        # Bước 1.2: Tách tọa độ box (gt_box)
        gc_x = tf.expand_dims(gt_box[:, 0], 0)
        gc_y = tf.expand_dims(gt_box[:, 1], 0)
        gw = tf.expand_dims(gt_box[:, 2], 0)
        gh = tf.expand_dims(gt_box[:, 3], 0)

        # Bước 1.3: Kiểm tra điều kiện nằm trong box
        inside_x = tf.logical_and(pc_x >= gc_x - gw/2.0, pc_x <= gc_x + gw/2.0) 
        inside_y = tf.logical_and(pc_y >= gc_y - gh/2.0, pc_y <= gc_y + gh/2.0)
        is_inside = tf.logical_and(inside_x, inside_y)

        # Bước 2.1: Tính khoảng cách Euclidean
        diff = tf.expand_dims(point[:, :2], 1) - tf.expand_dims(gt_box[:, :2], 0)
        distance = tf.reduce_sum(tf.square(diff), axis=-1) 
        
        # Bước 2.2: Lấy vị trí gần nhất
        closest_point_idx = tf.argmin(distance, axis=0) 
        
        # Bước 2.3: Tạo ma trận True/False cho điểm gần nhất
        is_closest = tf.equal(
            tf.expand_dims(tf.range(tf.shape(point)[0], dtype=tf.int64), 1),
            tf.expand_dims(closest_point_idx, 0)
        )
        
        # Bước 2.4: Gán point nằm trong hoặc gần box
        is_assigned = tf.logical_or(is_inside, is_closest)

        # Bước 3.1: Tính diện tích các GT box
        area_GT_box = gw * gh
        
        # Bước 3.2: Đặt diện tích bằng vô cực nếu point đó không được gán
        area_matrix = tf.where(is_assigned, area_GT_box, tf.fill(tf.shape(is_assigned), float('inf'))) 
        
        # Bước 3.3: Chọn box có diện tích nhỏ nhất theo từng point
        min_area_idx = tf.argmin(area_matrix, axis=1)
        min_area = tf.reduce_min(area_matrix, axis=1)
        
        # Bước 3.4: Lọc mẫu dương
        positive_mask = tf.less(min_area, float('inf')) 
        
        # Bước 3.5: Trích xuất đúng thông tin
        matched_gt_box = tf.gather(gt_box, min_area_idx)
        matched_class = tf.gather(gt_class, min_area_idx)

        # Bước 4.1: Encode box (Tính toán dxy, dwh)
        box_target_xy = (matched_gt_box[:, :2] - point[:, :2]) / point[:, 2:]
        safe_gt_wh = tf.math.maximum(matched_gt_box[:, 2:], 1e-7)
        box_target_wh = tf.math.log(safe_gt_wh / point[:, 2:])
        target_box = tf.concat([box_target_xy, box_target_wh], axis=-1)

        # Bước 4.2: Encode class
        target_class = tf.where(positive_mask, tf.cast(matched_class, tf.float32), -1) 
        target_class = tf.expand_dims(target_class, axis=-1)

        # Bước 4.3: Gộp chung box và class thành output cuối cùng
        y_true = tf.concat([target_box, target_class], axis=-1)
        return img, y_true

    return label_encode

# ==========================================
# CÔNG TẮC KHỞI ĐỘNG VÀ KIỂM THỬ
# ==========================================
if __name__ == '__main__':
    # ---------------------------------------------------------
    # KHU VỰC 1: KIỂM THỬ BĂNG CHUYỀN (PHẦN 1 -> 3)
    # ---------------------------------------------------------
    print("="*50)
    print("🚀 ĐANG KIỂM THỬ BĂNG CHUYỀN DỮ LIỆU THỰC TẾ (PHẦN 1-3)")
    print("="*50)
    
    # Cập nhật đường dẫn tới file CSV dự án Hardhat của em
    csv_path = r'D:\Do_An_AI_Hardhat\csv_data\hardhat_labels.csv'
    
    df_da_convert = Convert_csv_to_number(csv_path)
    my_dataset = create_dataset_from_dataframe(df_da_convert)
    
    for path, bbox, label in my_dataset.take(1):
        print("\n[KẾT QUẢ PHẦN 1] Khối hàng gốc vừa bốc từ băng chuyền:")
        print(f"- Đường dẫn ảnh: {path.numpy().decode('utf-8')}")
        print(f"- Tọa độ Bounding Box (Đã quy về dạng Tâm và Kích thước): \n{bbox.numpy()}")
        print(f"- Mã số nhãn (Class ID): {label.numpy()}")
        
        img, bbox, label = read_img_label(path, bbox, label)
        img, bbox, label = resize_and_padding_img(img, bbox, label, target_size=512)
        print("\n[KẾT QUẢ PHẦN 2] SAU KHI RESIZE & PADDING (512x512):")
        print(f"- Tọa độ Bounding Box chuẩn: \n{bbox.numpy()}")
        
        img, bbox, label = data_augment(img, bbox, label)
        print("\n[KẾT QUẢ PHẦN 3] SAU KHI QUA LÒ DATA AUGMENTATION:")
        print(f"- Kích thước ma trận ảnh chốt hạ: {img.shape}")
        print(f"- Tọa độ Bounding Box (Có thể đã lật ngược): \n{bbox.numpy()}")
    
    # KHU VỰC 2: PHÒNG THÍ NGHIỆM TOÁN HỌC (TEST PHẦN 4)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🧠 ĐANG KIỂM THỬ THUẬT TOÁN LABEL ENCODE (PHẦN 4)")
    print("="*50)
    
    points_gia_lap = tf.constant([
        [10.0, 10.0, 8.0, 8.0],
        [35.0, 25.0, 8.0, 8.0],
        [50.0, 50.0, 8.0, 8.0]
    ], dtype=tf.float32)
    
    gt_boxes_gia_lap = tf.constant([
        [12.0, 15.0, 10.0, 10.0],
        [45.0, 20.0, 20.0, 20.0]
    ], dtype=tf.float32)

    gt_class_gia_lap = tf.constant([0.0, 1.0], dtype=tf.float32)

    may_gan_nhan = get_label_encode(points_gia_lap)
    _, y_true_ket_qua = may_gan_nhan(img=None, gt_box=gt_boxes_gia_lap, gt_class=gt_class_gia_lap)
    
    print("\n🎯 KẾT QUẢ XUẤT XƯỞNG TỪ MÁY GẮN NHÃN (y_true):")
    print(y_true_ket_qua.numpy())
    print("\n💡 Ý NGHĨA: 4 cột đầu là tọa độ biến đổi, cột cuối là Class (0, 1 hoặc -1 nếu trượt).")