import tensorflow as tf

# ==========================================
# BÀI 4: KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)
# MẢNH GHÉP 1: MÁY TẠO LƯỚI MỎ NEO (ANCHOR GENERATOR)
# ==========================================

class AchorFree:
    def __init__(self):
        self._num_anchor = 1
        # Các mức độ nén của mạng CNN: 8, 16, 32, 64, 128
        self.strides = [2 ** i for i in range(3, 8)] 
    
    def _get_anchors(self, feature_w, feature_h, level):
        """
        Khối này sẽ rải các điểm (Anchor points) lên trên từng tầng của Feature Map
        """
        # Tạo trục tọa độ 1D (cộng 0.5 để lấy Tâm của từng ô lưới)
        rx = tf.range(feature_w, dtype=tf.float32) + 0.5 
        ry = tf.range(feature_h, dtype=tf.float32) + 0.5
        
        # Đan chéo rx, ry để tạo thành tọa độ XY 2D (Ma trận lưới)
        X, Y = tf.meshgrid(rx, ry)
        
        # Phóng to lưới ngược về kích thước ảnh gốc (nhân với stride)
        centers = tf.stack([X, Y], axis=-1) * self.strides[level - 3]
        
        # Duỗi lưới thành danh sách các điểm: Shape [N, 2]
        centers = tf.reshape(centers, [-1, 2]) 
        
        # Gắn thêm 2 cột kích thước (stride_w, stride_h) vào centers
        stride_tensor = tf.fill([tf.shape(centers)[0], 2], tf.cast(self.strides[level-3], tf.float32))
        
        # Đóng gói thành mỏ neo hoàn chỉnh: Shape [N, 4] chứa (cx, cy, w, h)
        anchors = tf.concat([centers, stride_tensor], axis=-1)
        return anchors 

    def get_anchors(self, img_w, img_h):
        """
        Hàm tổng: Rải lưới cho toàn bộ 5 cấp độ (từ stride 8 đến 128)
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(img_w / (2**i)),
                tf.math.ceil(img_h / (2**i)),
                i
            )
            for i in range(3, 8) # i chạy từ 3 đến 7
        ]
        # Gom 5 tấm lưới lại thành 1 danh sách khổng lồ duy nhất
        return tf.concat(anchors, axis=0)

# ==========================================
# CÔNG TẮC KIỂM THỬ KHỐI LỆNH (CHẠY ĐỘC LẬP)
# ==========================================
if __name__ == '__main__':
    print("="*50)
    print("🚀 ĐANG KIỂM THỬ MÁY RẢI MỎ NEO (ẢNH 512x512)")
    print("="*50)
    
    # 1. Kích thước ảnh thực tế sinh ra từ Bài 3
    img_w, img_h = 512.0, 512.0
    
    # 2. Khởi tạo cỗ máy
    anchor_generate = AchorFree()

    # 3. Tiến hành rải lưới
    real_anchors = anchor_generate.get_anchors(img_w, img_h)
    
    print(f"✅ TỔNG SỐ ĐIỂM NEO ĐƯỢC SINH RA: {real_anchors.shape[0]} điểm!")
    print(f"✅ Hình dáng ma trận Mỏ neo: {real_anchors.shape} (N hàng, 4 cột [cx, cy, stride, stride])\n")
    
    print("👉 Chi tiết 3 mỏ neo ĐẦU TIÊN (Lưới dày, chuyên bắt vật thể nhỏ):")
    print(real_anchors[:3].numpy())
    
    print("\n👉 Chi tiết 3 mỏ neo CUỐI CÙNG (Lưới thưa, chuyên bắt vật thể khổng lồ):")
    print(real_anchors[-3:].numpy())