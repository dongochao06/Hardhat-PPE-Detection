import tensorflow as tf

# ==========================================
# MẢNH GHÉP 2: BỘ NÃO TRÍCH XUẤT ẢNH (RESNET-50 BACKBONE)
# ==========================================

def Conv_block(x, filter, stride):
    """
    Khối có Bước nhảy (Stride): Dùng để bóp nhỏ kích thước bản đồ.
    Đường đi tắt (x_skip) cũng phải được bóp nhỏ tương ứng để cộng vào nhau.
    """
    x_skip = x
    f1, f2 = filter

    # Layer 1: Nén kênh (có thể có bước nhảy stride)
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=(stride, stride), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Layer 2: Trích xuất đặc trưng chính 3x3
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Layer 3: Mở rộng số kênh lên f2
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # ĐƯỜNG ĐI TẮT (Skip Connection): Ép x_skip về cùng kích thước và số kênh f2
    x_skip = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), strides=(stride, stride), padding='valid',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    # Cộng gộp: Đặc trưng học được (x) + Dữ liệu gốc (x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)
    return x 


def Res_id_block(x, filter):
    """
    Khối Giữ nguyên (Identity Block): Không làm thay đổi kích thước bản đồ.
    Chỉ làm nhiệm vụ học sâu thêm đặc trưng.
    """
    x_skip = x 
    f1, f2 = filter

    # Layer 1
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(3, 3), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Layer 3
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Cộng gộp trực tiếp vì kích thước không đổi
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)
    return x


def resnet_50_backbone():
    """
    Dây chuyền Kim tự tháp: Nhận ảnh 512x512 và nhả ra 3 bản đồ (Stride 8, 16, 32)
    """
    # Khai báo cổng vào: Ép cứng kích thước 512x512 để khớp 100% với Bài 3
    input_dim = tf.keras.layers.Input(shape=(512, 512, 3))

    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='padding_zero')(input_dim)

    # Cổng 1 (Bóp nhỏ 2 lần -> 256x256)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', use_bias=False, name='Conv7x7')(x)
    block_1_out = x
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Cổng 2 (Bóp nhỏ thêm 2 lần -> 128x128. Tổng bóp: Stride 4)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='overlapping')(x)

    # Khối 2: Giữ nguyên 128x128
    x = Conv_block(x, (64, 256), 1) 
    x = Res_id_block(x, (64, 256)) 
    x = Res_id_block(x, (64, 256)) 
    block_2_out = x

    # Khối 3: Bóp nhỏ 2 lần -> 64x64. (TỔNG BÓP: STRIDE 8) -> P3
    x = Conv_block(x, (128, 512), 2) 
    x = Res_id_block(x, (128, 512)) 
    x = Res_id_block(x, (128, 512)) 
    x = Res_id_block(x, (128, 512)) 
    block_3_out = x

    # Khối 4: Bóp nhỏ 2 lần -> 32x32. (TỔNG BÓP: STRIDE 16) -> P4
    x = Conv_block(x, (256, 1024), 2) 
    x = Res_id_block(x, (256, 1024)) 
    x = Res_id_block(x, (256, 1024)) 
    x = Res_id_block(x, (256, 1024)) 
    x = Res_id_block(x, (256, 1024)) 
    x = Res_id_block(x, (256, 1024)) 
    block_4_out = x

    # Khối 5: Bóp nhỏ 2 lần -> 16x16. (TỔNG BÓP: STRIDE 32) -> P5
    x = Conv_block(x, (512, 2048), 2) 
    x = Res_id_block(x, (512, 2048)) 
    x = Res_id_block(x, (512, 2048)) 
    block_5_out = x

    model = tf.keras.models.Model(inputs=input_dim, outputs=[block_3_out, block_4_out, block_5_out], name='resnet_50')
    return model


# ==========================================
# MẢNH GHÉP 3: BỘ NÃO CHỐT HẠ (DETECTION HEAD)
# ==========================================
def head_of_model(output_filters, bias_init, num_convs=4, filters=256, name=None):
    head = tf.keras.Sequential(name=name)
    head.add(tf.keras.layers.Input(shape=[None, None, filters]))
    
    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    
    for i in range(num_convs):
        head.add(tf.keras.layers.Conv2D(
            filters=filters, kernel_size=3, padding='same',
            kernel_initializer=kernel_init, name=f'head_conv_{i+1}'
        ))
        head.add(tf.keras.layers.BatchNormalization(name=f'head_bn_{i+1}'))
        head.add(tf.keras.layers.ReLU(name=f'head_relu_{i+1}'))
        
    head.add(tf.keras.layers.Conv2D(
        filters=output_filters, kernel_size=3, strides=1, padding='same',
        kernel_initializer=kernel_init, bias_initializer=bias_init,
        name='head_output'
    ))
    return head

# =========================================================
# KIỂM THỬ: ĐƯA ẢNH 512x512 VÀO "MÁY ÉP KIM TỰ THÁP"
# =========================================================
if __name__ == '__main__':
    print("="*50)
    print("🚀 BƯỚC 1: KHỞI ĐỘNG DÂY CHUYỀN RESNET-50")
    print("="*50)
    
    # 1. Gọi cỗ máy
    backbone = resnet_50_backbone()
    
    # 2. Tạo một bức ảnh giả lập 512x512 (Giống hệt ảnh xuất ra từ Bài 3)
    # Shape: (1, 512, 512, 3) -> 1 bức ảnh, cao 512, rộng 512, 3 kênh màu RGB
    dummy_img = tf.random.normal([1, 512, 512, 3])
    
    # 3. Ép ảnh qua dây chuyền
    p3, p4, p5 = backbone(dummy_img)
    
    print("✅ ĐÃ ÉP ẢNH THÀNH CÔNG! KẾT QUẢ CÁC BẢN ĐỒ ĐẶC TRƯNG:")
    print(f"👉 Khối 3 (P3 - Stride 8) : Kích thước {p3.shape} (Chứa 64x64 ô, dùng soi vật nhỏ)")
    print(f"👉 Khối 4 (P4 - Stride 16): Kích thước {p4.shape} (Chứa 32x32 ô, dùng soi vật vừa)")
    print(f"👉 Khối 5 (P5 - Stride 32): Kích thước {p5.shape} (Chứa 16x16 ô, dùng soi vật to)")