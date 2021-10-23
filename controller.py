import cv2
import numpy as np

def find_lane_lines(img):
    """Phát hiện vạch kẻ đường
    Hàm này sẽ nhận vào một hình ảnh màu, ở hệ màu BGR,
    trả ra hình ảnh các vạch kẻ đường đã được lọc
    """

    # Chuyển ảnh đã đọc sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Gaussian để loại bỏ bớt nhiễu.
    # Các bạn có thể thử nghiệm các bộ lọc khác tại đây,
    # như bộ lọc Median hoặc Bilateral.
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)

    # Áp dụng bộ lọc Canny với 2 ngưỡng.
    # Các bạn có thể điều chỉnh 2 ngưỡng này và xem sự thay đổi
    # trong ảnh kết quả
    thresh_low = 150
    thresh_high = 200
    img_canny = cv2.Canny(img_gauss, thresh_low, thresh_high)

    # Trả về kết quả vạch kẻ đường
    return img_canny

def birdview_transform(img):
    """Áp dụng chuyển đổi birdview
    """
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img


def find_left_right_points(image, draw=False):
    """Tìm điểm trái/phải
    Sau khi có được ảnh birdview, một phương pháp đơn giản giúp xác định vị trí xe
    so với làn đường là tìm 2 điểm, một điểm thuộc vạch kẻ đường bên trái và một điểm
    thuộc vạch kẻ đường bên phải, sau cùng xét vị trí của điểm giữa xe so với hai điểm đó.
    """

    im_height, im_width = image.shape[:2]
    if draw: viz_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Vạch kẻ sử dụng để xác định tâm đường
    interested_line_y = int(im_height * 0.9)
    if draw: cv2.line(viz_img, (0, interested_line_y), (im_width, interested_line_y), (0, 0, 255), 2) 
    interested_line = image[interested_line_y, :]

    # Xác định điểm bên trái và bên phải
    left_point = -1
    right_point = -1
    lane_width = 100
    center = im_width // 2

    # Tìm điểm bên trái và bên phải bằng cách duyệt từ tâm ra
    for x in range(center, 0, -1):
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] > 0:
            right_point = x
            break

    # Dự đoán điểm bên phải khi chỉ nhìn thấy điểm bên trái
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width

    # Dự đoán điểm bên trái khi chỉ thấy điểm bên phải
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width

    # Vẽ hai điểm trái / phải lên ảnh
    if draw: 
        if left_point != -1:
            viz_img = cv2.circle(viz_img, (left_point, interested_line_y), 7, (255,255,0), -1)
        if right_point != -1:
            viz_img = cv2.circle(viz_img, (right_point, interested_line_y), 7, (0,255,0), -1)

    if draw: 
        return left_point, right_point, viz_img
    else:
        return left_point, right_point


def calculate_control_signal(img):
    """Tính toán để tìm tốc độ và góc lái cho xe từ ảnh đầu vào
    """

    # Xử lý ảnh để tìm các điểm trái / phải
    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform(img_lines)
    left_point, right_point, viz_img = find_left_right_points(img_birdview, draw=True)

    # Hiển thị kết quả xử lý ảnh
    cv2.imshow("Result", viz_img)
    cv2.waitKey(1)

    # Tính toán tốc độ và góc lái
    throttle = 0.5 # Tốc độ đang được đặt là 50% tốc độ cao nhất
    steering_angle = 0
    im_center = img.shape[1] // 2
    # Nếu tìm thấy điểm trái và điểm phải,
    # các điểm này sẽ có giá trị khác -1
    if left_point != -1 and right_point != -1:

        # Tính toán độ lệch giữa điểm giữa xe và làn đường
        center_point = (right_point + left_point) // 2
        center_diff =  im_center - center_point

        # Tính toán góc lái tỷ lệ với độ lệch
        steering_angle = - float(center_diff * 0.01)

    return throttle, steering_angle

