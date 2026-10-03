import os
import glob
import cv2
import pandas as pd

if __name__ == '__main__':
    # 1. Đường dẫn thư mục
    base_dir = r'D:\Do_An_AI_Hardhat'
    anno_dir = os.path.join(base_dir, 'anno')
    img_dir = os.path.join(base_dir, 'img')
    csv_dir = os.path.join(base_dir, 'csv_data')
    
    # Tạo thư mục csv_data nếu chưa có
    os.makedirs(csv_dir, exist_ok=True)

    # 2. TỪ ĐIỂN PHÂN LỚP (Tạm thời để trống chờ em tìm file data.yaml)
    # 2. TỪ ĐIỂN PHÂN LỚP CHUẨN (PPE Class Map)
    classes_map = {
        '0': 'Hardhat', 
        '1': 'Mask', 
        '2': 'NO-Hardhat', 
        '3': 'NO-Mask', 
        '4': 'NO-Safety_Vest', 
        '5': 'Person', 
        '6': 'Safety_Cone', 
        '7': 'Safety_Vest', 
        '8': 'machinery', 
        '9': 'vehicle'
    }


    # 3. Quét toàn bộ file nhãn txt
    txt_files = glob.glob(os.path.join(anno_dir, '*.txt'))
    csv_list = []
    
    print(f"Đang tiến hành dịch {len(txt_files)} file TXT sang CSV. Vui lòng đợi...")

    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        
        # Tìm bức ảnh tương ứng với file nhãn
        img_filename = filename.replace('.txt', '.jpg') 
        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            img_filename = filename.replace('.txt', '.png')
            img_path = os.path.join(img_dir, img_filename)

        if not os.path.exists(img_path):
            continue 

        # Dùng OpenCV đọc kích thước ảnh thực tế (chiều cao h, chiều rộng w)
        img = cv2.imread(img_path)
        if img is None: 
            continue
        h, w, _ = img.shape

        # Đọc nội dung file txt
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split()
            if len(data) != 5: continue
            class_id, x_center, y_center, width, height = data

            # THUẬT TOÁN DỊCH TỌA ĐỘ YOLO SANG TỌA ĐỘ PIXEL GỐC
            x_center_pixel = float(x_center) * w
            y_center_pixel = float(y_center) * h
            box_w_pixel = float(width) * w
            box_h_pixel = float(height) * h

            xmin = int(x_center_pixel - (box_w_pixel / 2))
            ymin = int(y_center_pixel - (box_h_pixel / 2))
            xmax = int(x_center_pixel + (box_w_pixel / 2))
            ymax = int(y_center_pixel + (box_h_pixel / 2))

            # Ràng buộc tọa độ không cho tràn ra ngoài mép ảnh
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            # Lấy tên class từ từ điển
            class_name = classes_map.get(class_id, f"class_{class_id}")

            # Đóng gói dữ liệu
            csv_list.append((img_path, w, h, class_name, xmin, ymin, xmax, ymax))

    # 4. Xuất mẻ hàng ra file CSV
    column_name = ['img_path', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csv_df = pd.DataFrame(csv_list, columns=column_name)
    csv_df.to_csv(os.path.join(csv_dir, 'hardhat_labels.csv'), index=None)
    
    print(f"✅ Đã tạo thành công file CSV tại {csv_dir}!")
    print(f"Tổng số lượng hộp nhận diện (Bounding box) quét được là: {len(csv_list)}")