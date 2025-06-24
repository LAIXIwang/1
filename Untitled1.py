#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import cv2
import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QFileDialog, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class CustomNet(nn.Module):
    """自定义神经网络模型。
    请完成对__init__、forward方法的实现，以完成CustomNet类的定义。
    """

    def __init__(self):
        """初始化方法。
        在本方法中，请完成神经网络的各个模块/层的定义。
        请确保每层的输出维度与下一层的输入维度匹配。
        """
        super(CustomNet, self).__init__()

        # START----------------------------------------------------------
        self.cnn_layers = nn.Sequential(
            # 输入: 3通道, 64x64
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16x32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 32x16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64x8x8
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 假设有10个类别
        )  # 修复了这里的括号
        # END------------------------------------------------------------

    def forward(self, x):
        """前向传播过程。
        在本方法中，请完成对神经网络前向传播计算的定义。
        """
        # START----------------------------------------------------------
        x = self.cnn_layers(x)
        x = self.fc_layers(x)
        return x
        # END------------------------------------------------------------





class GestureRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势识别应用")
        self.setGeometry(100, 100, 900, 700)
        
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.load('./models/model.pkl', map_location=self.device,weights_only=False)
            self.model.to(self.device)
            self.model.eval()
            print("模型加载成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            print(f"模型加载失败: {str(e)}")
            self.model = None
        
        # 创建主选项卡
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # 创建上传图片识别标签页
        self.upload_tab = QWidget()
        self.tabs.addTab(self.upload_tab, "图片识别")
        self.setup_upload_tab()
        
        # 创建实时摄像头识别标签页
        self.camera_tab = QWidget()
        self.tabs.addTab(self.camera_tab, "实时识别")
        self.setup_camera_tab()
        
        # 摄像头相关变量
        self.camera_active = False
        self.camera = None
        
        # 手势类别标签（根据实际模型输出调整）
        self.class_labels = [
            "手掌", "握拳", "OK手势", "点赞", "比心", 
            "数字1", "数字2", "数字3", "数字4", "数字5"
        ]
        
    def setup_upload_tab(self):
        """设置图片识别标签页"""
        layout = QVBoxLayout()
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setFixedHeight(40)
        btn_layout.addWidget(self.upload_btn)
        
        self.recognize_btn = QPushButton("识别手势")
        self.recognize_btn.clicked.connect(self.recognize_image)
        self.recognize_btn.setFixedHeight(40)
        self.recognize_btn.setEnabled(False)
        btn_layout.addWidget(self.recognize_btn)
        
        layout.addLayout(btn_layout)
        
        # 图片显示区域
        img_group = QGroupBox("图片预览")
        img_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setText("等待上传图片...")
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        img_layout.addWidget(self.image_label)
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)
        
        # 结果区域
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("等待识别...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        self.result_label.setMinimumHeight(80)
        result_layout.addWidget(self.result_label)
        
        # 置信度显示
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 16px; color: #555;")
        result_layout.addWidget(self.confidence_label)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        self.upload_tab.setLayout(layout)
    
    def setup_camera_tab(self):
        """设置实时摄像头识别标签页"""
        layout = QVBoxLayout()
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        self.start_camera_btn = QPushButton("启动摄像头")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        self.start_camera_btn.setFixedHeight(40)
        btn_layout.addWidget(self.start_camera_btn)
        
        self.capture_btn = QPushButton("捕获并识别")
        self.capture_btn.clicked.connect(self.capture_and_recognize)
        self.capture_btn.setFixedHeight(40)
        self.capture_btn.setEnabled(False)
        btn_layout.addWidget(self.capture_btn)
        
        layout.addLayout(btn_layout)
        
        # 摄像头区域
        camera_group = QGroupBox("摄像头画面")
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setText("摄像头未启动")
        self.camera_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        camera_layout.addWidget(self.camera_label)
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # 结果区域
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout()
        self.camera_result_label = QLabel("等待识别...")
        self.camera_result_label.setAlignment(Qt.AlignCenter)
        self.camera_result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        self.camera_result_label.setMinimumHeight(80)
        result_layout.addWidget(self.camera_result_label)
        
        # 置信度显示
        self.camera_confidence_label = QLabel("")
        self.camera_confidence_label.setAlignment(Qt.AlignCenter)
        self.camera_confidence_label.setStyleSheet("font-size: 16px; color: #555;")
        result_layout.addWidget(self.camera_confidence_label)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        self.camera_tab.setLayout(layout)
        
        # 设置定时器用于更新摄像头画面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
    
    def upload_image(self):
        """上传图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # 调整大小以适应显示区域
                self.image_label.setPixmap(
                    pixmap.scaled(
                        self.image_label.width(), 
                        self.image_label.height(),
                        Qt.KeepAspectRatio
                    )
                )
                self.recognize_btn.setEnabled(True)
                self.result_label.setText("图片已上传，点击识别手势")
                self.confidence_label.setText("")
    
    def recognize_image(self):
        """识别上传的图片"""
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(self, "警告", "请先上传图片")
            return
        
        if self.model is None:
            QMessageBox.critical(self, "错误", "模型未正确加载")
            return
        
        try:
            # 图像预处理
            transform = Compose([
                Resize((64, 64)),  # 调整到模型需要的尺寸
                ToTensor()
            ])
            
            # 加载并预处理图像
            image = Image.open(self.current_image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)  # 增加batch维度
            
            # 推理
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 获取结果
            class_idx = predicted.item()
            confidence_value = confidence.item()
            class_name = self.class_labels[class_idx] if class_idx < len(self.class_labels) else f"类别{class_idx}"
            
            # 更新UI
            self.result_label.setText(f"识别结果: {class_name}")
            self.confidence_label.setText(f"置信度: {confidence_value:.4f}")
            
            # 在图片上显示结果
            pixmap = QPixmap(self.current_image_path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.width(), 
                    self.image_label.height(),
                    Qt.KeepAspectRatio
                )
            )
            
        except Exception as e:
            QMessageBox.critical(self, "识别错误", f"识别过程中发生错误: {str(e)}")
    
    def toggle_camera(self):
        """启动/停止摄像头"""
        if not self.camera_active:
            # 尝试打开摄像头
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头")
                return
            
            self.camera_active = True
            self.start_camera_btn.setText("停止摄像头")
            self.capture_btn.setEnabled(True)
            self.timer.start(30)  # 30ms更新一次画面
        else:
            # 停止摄像头
            self.camera_active = False
            self.start_camera_btn.setText("启动摄像头")
            self.capture_btn.setEnabled(False)
            self.timer.stop()
            self.camera.release()
            self.camera_label.clear()
            self.camera_label.setText("摄像头已停止")
            self.camera_result_label.setText("等待识别...")
            self.camera_confidence_label.setText("")
    
    def update_camera_frame(self):
        """更新摄像头画面"""
        if self.camera_active and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # 转换为RGB格式
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                # 转换为Qt图像
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                # 调整大小以适应显示区域
                self.camera_label.setPixmap(
                    pixmap.scaled(
                        self.camera_label.width(), 
                        self.camera_label.height(),
                        Qt.KeepAspectRatio
                    )
                )
    
    def capture_and_recognize(self):
        """捕获当前帧并识别手势"""
        if not self.camera_active or not self.camera.isOpened():
            QMessageBox.warning(self, "警告", "请先启动摄像头")
            return
        
        if self.model is None:
            QMessageBox.critical(self, "错误", "模型未正确加载")
            return
        
        try:
            # 捕获当前帧
            ret, frame = self.camera.read()
            if not ret:
                QMessageBox.warning(self, "警告", "无法捕获摄像头画面")
                return
            
            # 转换为PIL图像
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # 保存临时图像用于显示
            self.temp_image = pil_image.copy()
            
            # 图像预处理
            transform = Compose([
                Resize((64, 64)),  # 调整到模型需要的尺寸
                ToTensor()
            ])
            
            # 预处理图像
            image_tensor = transform(pil_image).unsqueeze(0).to(self.device)  # 增加batch维度
            
            # 推理
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 获取结果
            class_idx = predicted.item()
            confidence_value = confidence.item()
            class_name = self.class_labels[class_idx] if class_idx < len(self.class_labels) else f"类别{class_idx}"
            
            # 更新UI
            self.camera_result_label.setText(f"识别结果: {class_name}")
            self.camera_confidence_label.setText(f"置信度: {confidence_value:.4f}")
            
            # 在摄像头上显示当前帧（保持不变）
            self.update_camera_frame()
            
        except Exception as e:
            QMessageBox.critical(self, "识别错误", f"识别过程中发生错误: {str(e)}")
    
    def closeEvent(self, event):
        """关闭应用时释放资源"""
        if self.camera_active and self.camera.isOpened():
            self.camera.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-size: 14px;
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 1ex;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            font-size: 14px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #888888;
        }
        QLabel {
            font-size: 14px;
        }
    """)
    
    window = GestureRecognitionApp()
    window.show()
    sys.exit(app.exec_())


# In[ ]:




