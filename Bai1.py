import os
import glob
import shutil

if __name__ == '__main__':
    # 1. Đổi tên cơ ngơi sang dự án Mũ bảo hộ
    base_dir = r'D:\Do_An_AI_Hardhat' 
    
    # 2. Khai báo 3 cái kho
    data_dir = os.path.join(base_dir, 'data')
    anno_dir = os.path.join(base_dir, 'anno')
    img_dir = os.path.join(base_dir, 'img')
    
    # Lệnh tự động tạo thư mục anno và img
    os.makedirs(anno_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # 3. Bật Radar quét mục tiêu (Quét TXT thay vì XML)
    txt_files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    # Quét cả ảnh JPG và PNG 
    img_files = glob.glob(os.path.join(data_dir, '*.jpg'))
    img_files.extend(glob.glob(os.path.join(data_dir, '*.png')))
    
    print(f"🔍 Đã tìm thấy: {len(txt_files)} file nhãn TXT và {len(img_files)} file Ảnh trong bãi 'data'.")
    print("Bắt đầu quá trình dọn dẹp phân loại...")

    # 4. Robot bốc vác file TXT
    for txt_path in txt_files:
        txt_file_name = os.path.basename(txt_path)
        new_path = os.path.join(anno_dir, txt_file_name)
        shutil.move(txt_path, new_path)

    # 5. Robot bốc vác file Ảnh
    for img_path in img_files:
        img_file_name = os.path.basename(img_path)
        new_path = os.path.join(img_dir, img_file_name)
        shutil.move(img_path, new_path)
        
    print("✅ Đã dọn dẹp hoàn tất! Các file đã về đúng vị trí.")