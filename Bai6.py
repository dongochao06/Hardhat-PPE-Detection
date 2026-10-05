import tensorflow as tf
# (Giữ nguyên các hàm Conv_block, Res_id_block, resnet_50_backbone, head_of_model đã viết ở trên)

# ==========================================
# MẢNH GHÉP 4 (BÀI 6): TRẠM BIẾN ÁP VÀ TRỘN DỮ LIỆU (FPN)
# ==========================================
class FeaturePyramid(tf.keras.layers.Layer):
    """
    Xây dựng Mạng Kim tự tháp Đặc trưng (FPN).
    Ép 3 bản đồ gốc (C3, C4, C5) về cùng 256 kênh, cộng dồn tri thức và sinh thêm P6, P7.
    """
    def __init__(self, backbone=None, **kwargs):
        super().__init__(name='FeaturePyramid', **kwargs)
        # Nhận bộ não ResNet-50 vào
        self.backbone = backbone if backbone else resnet_50_backbone() 
        
        # 1. Ép kênh về 256
        self.conv_c3_1x1 = tf.keras.layers.Conv2D(256, 1, 1, padding='same', name='fpn_c3_1x1')
        self.conv_c4_1x1 = tf.keras.layers.Conv2D(256, 1, 1, padding='same', name='fpn_c4_1x1')
        self.conv_c5_1x1 = tf.keras.layers.Conv2D(256, 1, 1, padding='same', name='fpn_c5_1x1')
        
        # 2. Làm mịn sau khi cộng dồn
        self.conv_c3_3x3 = tf.keras.layers.Conv2D(256, 3, 1, padding='same', name='fpn_p3_3x3')
        self.conv_c4_3x3 = tf.keras.layers.Conv2D(256, 3, 1, padding='same', name='fpn_p4_3x3')
        self.conv_c5_3x3 = tf.keras.layers.Conv2D(256, 3, 1, padding='same', name='fpn_p5_3x3')
        
        # 3. Tạo bản đồ P6, P7 cho vật thể siêu to
        self.conv_c6_3x3 = tf.keras.layers.Conv2D(256, 3, 2, padding='same', name='fpn_p6_3x3')
        self.conv_c7_3x3 = tf.keras.layers.Conv2D(256, 3, 2, padding='same', name='fpn_p7_3x3')
        
        # Công cụ phóng to ảnh lên 2 lần
        self.upsample_2x = tf.keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        # Lấy 3 bản đồ thô từ ResNet
        c3, c4, c5 = self.backbone(images, training=training)

        # Ép đồng phục 256 kênh
        p3 = self.conv_c3_1x1(c3)
        p4 = self.conv_c4_1x1(c4)
        p5 = self.conv_c5_1x1(c5)

        # Cộng dồn tri thức từ trên xuống
        p4 = p4 + self.upsample_2x(p5)
        p3 = p3 + self.upsample_2x(p4)

        # Quét làm mịn
        p3 = self.conv_c3_3x3(p3)
        p4 = self.conv_c4_3x3(p4)
        p5 = self.conv_c5_3x3(p5)

        # Tạo P6, P7
        p6 = self.conv_c6_3x3(c5)
        p7 = self.conv_c7_3x3(tf.nn.relu(p6))

        return p3, p4, p5, p6, p7


# ==========================================
# MẢNH GHÉP 5: BẢN VẼ TỔNG THỂ (RETINANET LẮP RÁP HOÀN CHỈNH)
# ==========================================
def build_object_detection_model(num_classes, num_anchors=9):
    """
    Cắm phích toàn bộ: Ảnh -> FPN -> (Đầu phân loại + Đầu vẽ khung)
    """
    # 1. Cổng vào chính của hệ thống
    inputs = tf.keras.layers.Input(shape=(512, 512, 3))
    
    # 2. Chạy qua FPN (Bao gồm luôn ResNet bên trong)
    fpn = FeaturePyramid()
    p3, p4, p5, p6, p7 = fpn(inputs)
    features = [p3, p4, p5, p6, p7]
    
    # 3. Chế tạo 2 Đường ống Đầu dự đoán riêng biệt
    # Đường 1: Đoán Nhãn (Ví dụ 3 class: Mũ, Áo, Người) -> Cần num_classes * num_anchors đầu ra
    cls_head = head_of_model(output_filters=num_classes * num_anchors, 
                             bias_init=tf.keras.initializers.Constant(-4.6), # Mẹo ép AI ưu tiên background
                             name='classification_head')
    
    # Đường 2: Đoán Khung tọa độ (x, y, w, h) -> Cần 4 * num_anchors đầu ra
    box_head = head_of_model(output_filters=4 * num_anchors, 
                             bias_init='zeros', 
                             name='box_regression_head')
    
    # 4. Nhét 5 tấm bản đồ qua 2 cái Đầu này
    cls_outputs = []
    box_outputs = []
    
    for feature in features:
        cls_outputs.append(cls_head(feature))
        box_outputs.append(box_head(feature))
    
    # Đóng gói cỗ máy thành phẩm
    model = tf.keras.models.Model(inputs=inputs, outputs=cls_outputs + box_outputs, name='Safety_Gear_Detector')
    return model


# =========================================================
# KIỂM THỬ TOÀN DIỆN CỖ MÁY
# =========================================================
if __name__ == '__main__':
    print("="*50)
    print("🚀 BƯỚC 2: KHỞI ĐỘNG DÂY CHUYỀN NHẬN DIỆN HOÀN CHỈNH")
    print("="*50)
    
    # Giả sử dataset của em có 3 nhãn: Mũ bảo hộ, Áo phản quang, Người
    NUM_CLASSES = 3 
    
    # Lắp ráp cỗ máy
    detector = build_object_detection_model(num_classes=NUM_CLASSES)
    
    # Đưa ảnh giả 512x512 vào
    dummy_img = tf.random.normal([1, 512, 512, 3])
    outputs = detector(dummy_img)
    
    print("✅ CỖ MÁY ĐÃ CHẠY TRƠN TRU TỪ ĐẦU TỚI CUỐI!")
    print(f"👉 Tổng cộng có {len(outputs)} cổng xả dữ liệu (5 cổng Nhãn + 5 cổng Khung tọa độ).")
    
    # In thử cổng xả nhãn của bản đồ P3
    print(f"👉 Kết quả dự đoán Nhãn trên P3: {outputs[0].shape}")
    # In thử cổng xả tọa độ của bản đồ P3
    print(f"👉 Kết quả dự đoán Khung trên P3: {outputs[5].shape}")