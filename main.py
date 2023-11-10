import sys
import cv2
import torch
import mediapipe as mp
import numpy as np
from QCandyUi.CandyWindow import createWindow
from torchvision import transforms
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget,QApplication,QPushButton,QToolTip,QLabel,QFileDialog,QFrame
from PyQt5.QtGui import QFont,QPixmap,QImage
from network import Model

run_num = 3
name = 'final'

"""
拍照识别bug
把去除背景识别设置成单独功能
"""
class Win(QWidget):
    def __init__(self):
        super().__init__()
        self.path = [] # 待预测的图片
        self.timer = QTimer() # 计时器
        self.timer_photo = QTimer() # 计时器2
        self.timer_bg = QTimer() # 计时器3 # 背景
        self.count = 0 # 计时完成次数-每5*40ms截一次图
        self.init_UI() # 初始化UI
        self.load_model() # 载入模型
        # 手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        # grabcut背景分割
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        self.fgcut = False
        # 背景录入
        self.bg_count = 0
    def init_UI(self):
        self.init_0()
        self.init_1()
    def init_0(self):
        # 初始化窗口0
        # ---绝对布局--- #
        # ---Label--- #
        # label_00
        self.lb_00 = QLabel(self)
        self.lb_00.setScaledContents(True)  # 图片自适应大小
        self.lb_00.setFrameShape(QFrame.Box) # 画边界
        self.lb_00.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # label_01
        self.lb_01 = QLabel(self)
        self.lb_01.setWordWrap(True)  # 自动换行
        self.lb_01.setFrameShape(QFrame.Box)  # 画边界
        self.lb_01.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # label_02
        self.lb_02 = QLabel(self)
        self.lb_02.setScaledContents(True)
        self.lb_02.setFrameShape(QFrame.Box)  # 画边界
        self.lb_02.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # label_03
        self.lb_03 = QLabel(self)
        self.lb_03.setScaledContents(True)
        self.lb_03.setFrameShape(QFrame.Box)  # 画边界
        self.lb_03.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # ---按钮---- #
        QToolTip.setFont(QFont("楷体", 15))  # 设置按钮说明字体和大小
        self.btn_00 = QPushButton("打开图片", self)  # 按钮
        self.btn_00.setToolTip("选择待预测的手势图片")  # 按钮说明
        self.btn_00.clicked.connect(lambda: self.open())
        # 切换窗口1
        self.btn_01 = QPushButton("摄像头识别", self)  # 按钮
        self.btn_01.setToolTip("切换到窗口1")  # 按钮说明
        self.btn_01.clicked.connect(lambda : self.turn_1())
        # 预测
        self.btn_02 = QPushButton("识别",self)
        self.btn_02.setToolTip("载入模型预测结果")
        self.btn_02.clicked.connect(lambda : self.predict())
        # 拍照识别
        self.btn_03 = QPushButton("拍照识别",self)
        self.btn_03.clicked.connect(lambda: self.photo_on())
        # 拍照
        self.btn_04 = QPushButton("拍照",self)
        self.btn_04.clicked.connect(lambda: self.photo_off())
        self.timer_photo.timeout.connect(lambda: self.photo_show())  # 计时结束执行
        # 窗口参数
        self.resize(800, 600)  # 设定窗口大小
        # ---绝对布局--- #
        self.lb_00.setGeometry(20,20,481,321) # 图片
        self.lb_01.setGeometry(520,420,150,100) # 框
        self.lb_02.setGeometry(20,420,100,100) # 裁剪后图片
        self.lb_03.setGeometry(140,420,100,100) # 前景图片
        self.btn_00.setGeometry(520,20,150,90) # 打开图片
        self.btn_01.setGeometry(520,220,150,90) # 切换到实时摄像头识别
        self.btn_02.setGeometry(520,320,150,90) # 识别
        self.btn_03.setGeometry(520,120,150,90) # 拍照识别
        self.btn_04.setGeometry(520,120,150,90) # 拍照
        # 隐藏控件
        self.btn_04.hide()
        # 字体参数
        self.lb_01.setFont(QFont("Roman times",15))
    def init_1(self):
        # 初始化窗口1
        # Labeld
        # 摄像框
        self.lb_10 = QLabel(self)
        self.lb_10.setFrameShape(QFrame.Box)  # 画边界
        self.lb_10.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # 文本框
        self.lb_11 = QLabel(self)
        self.lb_11.setWordWrap(True) # 自动换行
        self.lb_11.setFrameShape(QFrame.Box)  # 画边界
        self.lb_11.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # 裁剪框
        self.lb_12 = QLabel(self)
        self.lb_12.setFrameShape(QFrame.Box)  # 画边界
        self.lb_12.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # 背景分割框
        self.lb_13 = QLabel(self)
        self.lb_13.setFrameShape(QFrame.Box)  # 画边界
        self.lb_13.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # 测试框
        self.lb_14 = QLabel(self)
        self.lb_14.setFrameShape(QFrame.Box)  # 画边界
        self.lb_14.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(127, 255, 212);")
        # 按钮
        QToolTip.setFont(QFont("楷体", 15))  # 设置按钮说明字体和大小
        self.btn_10 = QPushButton("图片识别", self)  # 按钮
        self.btn_10.setToolTip("切换到窗口0")  # 按钮说明
        self.btn_10.clicked.connect(lambda : self.turn_0())
        self.btn_11 = QPushButton("开启摄像头",self)
        self.btn_11.clicked.connect(lambda: self.camera_on())
        self.btn_12 = QPushButton("关闭摄像头",self)
        self.btn_12.clicked.connect(lambda: self.camera_off())
        self.btn_13 = QPushButton("录入背景",self)
        self.btn_13.clicked.connect(lambda: self.gain_background())
        self.timer_bg.timeout.connect(lambda: self.timer_background())
        self.timer.timeout.connect(lambda : self.show_camera()) # 计时结束执行
        # 默认隐藏控件
        self.lb_10.hide()
        self.lb_11.hide()
        self.lb_12.hide()
        self.lb_13.hide()
        self.lb_14.hide()
        self.btn_10.hide()
        self.btn_11.hide()
        self.btn_12.hide()
        self.btn_13.hide()
        # 绝对布局
        self.lb_10.setGeometry(20,20,481,321) # 摄像框
        self.lb_11.setGeometry(520,320,150,100) # 文本框
        self.lb_12.setGeometry(20,420,100,100) # 手部裁剪图
        self.lb_13.setGeometry(140, 420, 100, 100)  # 手部裁剪图
        self.lb_14.setGeometry(260, 420, 100, 100) # 测试
        self.btn_10.setGeometry(520,220,150,90) # 回到窗口0
        self.btn_11.setGeometry(520,20,150,90) # 打开摄像头
        self.btn_12.setGeometry(520,20,150,90) # 关闭摄像头
        self.btn_13.setGeometry(520,120,150,90) # 录入背景
        # 字体参数
        self.lb_11.setFont(QFont("Roman times", 15))
    def load_model(self):
        # 模型加载
        self.model = Model().eval()
        state_dict = torch.load('train/run'+str(run_num) +'/'+ name + '.pth', map_location='cpu')
        print("模型加载...")
        self.model.load_state_dict(state_dict)
        print("模型加载成功！")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 显示
    def open(self):
        # 打开图片
        name,_= QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;*.jpeg;;*.bmp")
        if len(name)!=0:
            self.jpg = QPixmap(name).scaled(self.lb_00.width(), self.lb_00.height())
            self.lb_00.setPixmap(self.jpg) # 显示图片
            self.path.clear()
            self.path.append(name)
    def turn_0(self):
        # 切换到窗口0
        # 先关闭摄像头
        flag = self.timer.isActive()
        if flag == True:
            self.lb_11.setText("请先关闭摄像头")
        else:
            # 窗口1控件
            self.lb_10.hide()
            self.lb_11.hide()
            self.lb_12.hide()
            self.lb_13.hide()
            self.lb_14.hide()
            self.btn_10.hide()
            self.btn_11.hide()
            self.btn_12.hide()
            self.btn_13.hide()
            # 窗口0控件
            self.lb_00.show()
            self.lb_01.show()
            self.lb_02.show()
            self.lb_03.show()
            self.btn_00.show()
            self.btn_01.show()
            self.btn_02.show()
            self.btn_03.show()
    def turn_1(self):
        flag = self.timer_photo.isActive()
        if flag == True:
            self.lb_01.setText("请先关闭摄像头")
        else:
            # 切换到窗口1
            # 窗口0控件
            self.lb_00.hide()
            self.lb_01.hide()
            self.lb_02.hide()
            self.lb_03.hide()
            self.btn_00.hide()
            self.btn_01.hide()
            self.btn_02.hide()
            self.btn_03.hide()
            # 窗口1控件
            self.lb_10.show()
            self.lb_11.show()
            self.lb_12.show()
            self.lb_13.show()
            self.lb_14.show()
            self.btn_10.show()
            self.btn_11.show()
            self.btn_12.hide()
            self.btn_13.show()
    def camera_on(self):
        self.btn_11.hide()
        self.btn_12.show()
        # 打开摄像头
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 内置摄像头
        self.timer.start(40) # 计时器启动 40ms
    def camera_off(self):
        self.btn_11.show()
        self.btn_12.hide()
        # 关闭摄像头
        self.cap.release()
        self.timer.stop()
        self.lb_10.clear()
        self.count=0
    def predict(self):
        if(len(self.path)==0):
            self.lb_01.setText("请先载入图片")
        else:
            self.fgcut =True
            # 裁剪
            frame = cv2.imread(self.path[0])
            frame,img,fg_img,mbool = self.hand_catch(frame)
            img = cv2.resize(img,(100, 100))
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转为RGB形式
            Qimg = QImage(img_RGB.data, img_RGB.shape[1], img_RGB.shape[0], QImage.Format_RGB888)
            fg_img = cv2.resize(fg_img, (100, 100))
            fg_RGB = cv2.cvtColor(fg_img, cv2.COLOR_BGR2RGB)  # 转为RGB形式
            fgImage = QImage(fg_RGB.data, fg_RGB.shape[1], fg_RGB.shape[0], QImage.Format.Format_RGB888)
            transform = transforms.ToTensor()
            fg_img = transform(fg_img)
            input = fg_img.unsqueeze(0) # 升维
            # 预测当前图片w
            output = self.model.forward(input)
            out = torch.max(output, dim=1)[1]  # 预测标签
            out_label = str(out.cpu().numpy()[0])
            # 显示
            img_show = QPixmap(Qimg)
            self.lb_01.setText(out_label)
            self.lb_02.setPixmap(img_show)
            self.lb_03.setPixmap(QPixmap.fromImage(fgImage))
    def show_camera(self):
        # 更新一次显示的摄像头图片
        flag, image = self.cap.read()
        # 提取前景
        if(self.count == 4):
            self.fgcut = True
        else:
            self.fgcut = False
        show = cv2.resize(image, (480, 320))
        # 画框
        show,cut_frame,fg_img,mbool = self.hand_catch(show) # 对show画框、裁剪
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format.Format_RGB888)
        self.lb_10.clear()
        self.lb_10.setPixmap(QPixmap.fromImage(showImage))
        # 保存图片
        self.count += 1
        if(self.count==5):
            self.count=0
            if mbool:
                cv2.imwrite("save.JPG",cut_frame) #保存
                fg_img = cv2.resize(fg_img, (100, 100))
                fg_RGB = cv2.cvtColor(fg_img, cv2.COLOR_BGR2RGB)  # 转为RGB形式
                fgImage = QImage(fg_RGB.data, fg_RGB.shape[1], fg_RGB.shape[0], QImage.Format.Format_RGB888)
                # 预测
                hand_img = cv2.resize(cut_frame, (100, 100))
                transform = transforms.ToTensor()
                fg_img = transform(fg_img)
                input = fg_img.unsqueeze(0)  # 升维
                output = self.model.forward(input)
                out = torch.max(output, dim=1)[1]  # 预测标签
                out_label = str(out.cpu().numpy()[0])
                # 显示手部裁剪图
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                cutImage = QImage(hand_img.data, hand_img.shape[1], hand_img.shape[0], QImage.Format.Format_RGB888)
                self.lb_12.clear()
                self.lb_12.setPixmap(QPixmap.fromImage(cutImage))
                self.lb_11.clear()
                self.lb_11.setText(out_label)
                self.lb_13.setPixmap(QPixmap.fromImage(fgImage))
    def hand_catch(self,frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        # 如果检测到了手部，则框出手部并裁剪出手部区域图像保存到文件夹中
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取手部区域的边界框
                height, width, _ = frame.shape
                x, y, w, h = cv2.boundingRect(
                    np.array([[l.x * width, l.y * height] for l in hand_landmarks.landmark]).astype(np.float32))
                # 扩框
                x_cen = x + w * 0.5
                y_cen = y + h * 0.5
                bar = max(w, h) * 0.5
                up = int(y_cen - 1.1 * bar)
                down = int(y_cen + 1.1 * bar)
                left = int(x_cen - 1.1 * bar)
                right = int(x_cen + 1.1 * bar)
                if up < 0:
                    up = 0
                if left < 0:
                    left = 0
                if down > height:
                    down = height
                if right > width:
                    right = width
                # 框出手部
                original = frame.copy()
                cv2.rectangle(frame, (left, up), (right, down), (0, 255, 0), 2)
                cut = original[up:down,left:right]
                mbool = True
                fg_img = original.copy()
                if self.fgcut:
                    fg_img = self.gain_fg(fg_img,left,up,right-left,down-up)
                    self.fgcut = False
                break
        else:
            mbool = False
            cut = frame
            fg_img = frame.copy()
        return frame,cut,fg_img,mbool
    def photo_on(self):
        self.btn_03.hide()
        self.btn_04.show()
        # # 打开摄像头
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 内置摄像头
        self.timer_photo.start(40)  # 计时器2启动 40ms
    def photo_off(self):
        self.btn_03.show()
        self.btn_04.hide()
        flag, image = self.cap.read()
        cv2.imwrite("photo.JPG",image)
        # 关闭摄像头
        self.cap.release()
        self.timer_photo.stop()
        self.path.clear()
        self.path.append("photo.JPG")
    def photo_show(self):
        flag, image = self.cap.read()
        show = cv2.resize(image, (480, 320))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format.Format_RGB888)
        self.lb_00.setPixmap(QPixmap.fromImage(showImage))
    def gain_fg(self,img,x,y,w,h):
        # 等比resize
        re_shape = 100
        height,width,t = img.shape
        img = cv2.resize(img,(re_shape,re_shape))
        h_rate = re_shape/height
        w_rate = re_shape/width
        x = int(x*w_rate+0.5)
        w = int(w*w_rate+0.5)
        y = int(y*h_rate+0.5)
        h = int(h*h_rate+0.5)
        # 测试
        # if(self.timer.isActive()==True):
        #     test_img = img[y:y+h,x:x+w]
        #     test_img = cv2.resize(test_img,(100,100))
        #     test_RGB = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  # 转为RGB形式
        #     testImage = QImage(test_RGB.data, test_RGB.shape[1], test_RGB.shape[0], QImage.Format.Format_RGB888)
        #     self.lb_14.setPixmap(QPixmap.fromImage(testImage))
        # 使用grabcut进行前景提取
        mask = np.zeros(img.shape[:2], np.uint8)
        cmask = self.gain_cmask(img,x,y,w,h)
        # for i  in range(0,100):
        #      for j in range(0,100):
        # 前景提取（手部范围）
        rect = (x,y,w,h) # x,y,w,h
        # print(cmask.shape)
        # 初始前背景设定
        for i  in range(0,100):
            for j in range(0,100):
                # 框内
                if i>x and i<x+w and j>y and j<y+h:
                    if cmask[j-y][i-x] == 1:
                        mask[j][i] = 1 # 前景
                    else:
                        mask[j][i] = 3 # 可能的前景
        cv2.grabCut(img, mask, rect, self.bgdModel, self.fgdModel, 10, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        cut_img = img[y:y+h,x:x+w]
        return cut_img
    def gain_cmask(self,img,x,y,w,h):
        # 裁图
        cut_img = img[y:y+h,x:x+w]
        cut_bg = self.bg[y:y+h,x:x+w]
        # 灰度
        bgGray = cv2.cvtColor(cut_bg, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
        # 相减
        sub = imgGray.astype("int32") - bgGray.astype("int32")
        sub = np.absolute(sub).astype("uint8")
        # 二值化 0/255
        ret, thresh = cv2.threshold(sub, cv2.mean(sub)[0]+5, 255, cv2.THRESH_BINARY)
        # 膨胀和腐蚀
        # thresh2 = cv2.erode(thresh, None, iterations=1) # 先腐蚀后膨胀
        # thresh3 = cv2.dilate(thresh2, None, iterations=1)
        thresh2 = cv2.dilate(thresh, None, iterations=1)# 先膨胀后腐蚀
        thresh3 = cv2.erode(thresh2, None, iterations=1)
        ret2, thresh4 = cv2.threshold(thresh3, 127, 1, cv2.THRESH_BINARY)  # 归一
        thresh3 = cv2.resize(thresh3,(100,100))
        thresh3 = cv2.cvtColor(thresh3, cv2.COLOR_BGR2RGB)
        testImage = QImage(thresh3.data, thresh3.shape[1], thresh3.shape[0], QImage.Format.Format_RGB888)
        self.lb_14.setPixmap(QPixmap.fromImage(testImage))
        return thresh4
    def gain_background(self):
        if self.timer.isActive():
            self.lb_11.setText("请先关闭摄像头")
        else:
            # # 打开摄像头
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 内置摄像头
            self.timer_bg.start(40)  # 计时器3启动 40*50 = 2000ms = 2s
            self.bg_count = 0
            self.lb_11.setText("倒计时2s")
    def timer_background(self):
        # 展示
        flag, image = self.cap.read()
        show = cv2.resize(image, (480, 320))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format.Format_RGB888)
        self.lb_10.setPixmap(QPixmap.fromImage(showImage))
        self.bg_count+=1
        if self.bg_count==50:
            # 保存
            cv2.imwrite("camera.JPG",image)
            # 更改
            self.bg = image
            self.bg = cv2.resize(self.bg, (100, 100))
            self.lb_11.setText("完成")
            # 关闭摄像头
            self.cap.release()
            self.timer_bg.stop()

# 主函数
if __name__ == "__main__":
    app = QApplication(sys.argv) # 应用对象
    windows = Win()
    windows = createWindow(windows,'blueDeep') # 风格 blue / blueGreen / blueDeep / blueLight
    windows.setWindowTitle("手势识别")
    windows.show()
    sys.exit(app.exec())