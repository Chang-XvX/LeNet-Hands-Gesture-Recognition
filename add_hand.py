import cv2
import mediapipe as mp
import os
import random

# 改变保存图片的参数 确保数据的多样以及模型鲁棒性
def img_aug(img, light=1, bias=0):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(0, width):
        for j in range(0, height):
            for k in range(3):
                tmp = int(img[j, i, k]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,k] = tmp
    return img

#定义某类数字的总数
number_size = 500
# 选择一下要录入的数字
num = input("请输入要录入的数字(0-9):")
print('----------------------------------------------------------------')

# 初始化摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 初始化MediaPipe手部识别模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
print('请确保您已经比好该手势并处于视觉范围内')

# 创建一个目录用于保存裁剪出的手部区域图像
data_root = os.path.join("hand_images", num)
if not os.path.exists(data_root):
    os.mkdir(data_root)
print('文件夹创建成功！')

# 检查目录下共有多少张图
num_of_data = len(os.listdir(data_root))
print(data_root+'目录下共有数字{}张'.format(num_of_data))
print('正在采集....')



cnt = int(num_of_data)
while True:
    if cnt >= number_size:
        print('录入结束， 您已成功录入%d张图!数字%d录入完成！' % (cnt, int(num)))
        break
    # 读取摄像头帧
    ret, frame = cap.read()

    # 将帧转换为RGB格式并输入到MediaPipe模型中进行手部识别
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 如果检测到了手部，则框出手部并裁剪出手部区域图像保存到文件夹中
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 提取手部区域的边界框
            x_min = int(min(l.x for l in hand_landmarks.landmark) * frame.shape[1])
            y_min = int(min(l.y for l in hand_landmarks.landmark) * frame.shape[0])
            x_max = int(max(l.x for l in hand_landmarks.landmark) * frame.shape[1])
            y_max = int(max(l.y for l in hand_landmarks.landmark) * frame.shape[0])
            real_w, real_h = x_max - x_min, y_max - y_min
            now_w = now_h = max(real_w, real_h)
            now_w //= 2
            now_h //= 2
            center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            # 裁剪出手部区域图像并保存到文件夹中
            if now_h * 2 <= 100:
                delta = 120 - now_h * 2
                delta //= 2
            else:delta = 10
            now_w += delta
            now_h += delta
            hand_image = frame[center_y - now_h:center_y + now_h, center_x - now_w:center_x+now_w]
            random_number = random.random()
            if random_number > 0.5:
                aug_hand_image = img_aug(hand_image, random.uniform(0.5, 1.5), random.randint(-50, 50))
                try:
                    aug_hand_image = cv2.resize(aug_hand_image, (100, 100))
                except:
                    # # print("\rThis frame failed to resize", end = "")
                    # print("\r请确保手势在区域内", end = "")
                    # flag = False
                    continue
                cv2.imwrite(os.path.join(data_root, num + ' (' + str(cnt) + ').JPG'), aug_hand_image)
            else:
                try:
                    hand_image = cv2.resize(hand_image, (100, 100))
                except:
                    # # print("\rThis frame failed to resize", end="")
                    # print("\r请确保手势在区域内", end="")
                    # flag = False
                    continue
                cv2.imwrite(os.path.join(data_root, num + ' (' + str(cnt) + ').JPG'), hand_image)
            cnt += 1
            # 框出手部
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # 调用cv.putText()添加文字
            text = "Number {} you are recording".format(num)
            cv2.putText(frame, text, (int(x_min + y_min) // 2, int(y_min + 5)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,255), 2)

    print('\r<已采集总图像数: %d>'%(cnt), end = "")
    # 显示框出手部后的帧
    cv2.imshow("Hand Detection", frame)

    # 按下q键退出程序
    if cv2.waitKey(1) == ord('q'):
        print()
        print('录入结束， 您已成功录入%d张图!距离总数还有%d张' % (cnt, number_size - cnt))
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
