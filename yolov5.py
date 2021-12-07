'''
YoloV5 简化版
------------------------
作者: 阿凯爱玩机器人
微信: xingshunkai
邮箱: xingshunkai@qq.com
更新时间: 2021/12/07
'''
# 导入依赖
import math
import yaml
import argparse
import os
import time
import numpy as np
import cv2
# PyTorch
import torch
import torch.backends.cudnn as cudnn
# YoloV5-PyTorch
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import (
	check_img_size, non_max_suppression, apply_classifier, scale_coords,
	xyxy2xywh, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_sync


class YoloV5:
	def __init__(self, yolov5_yaml_path='config/yolov5.yaml'):
		'''初始化'''
		# 载入配置文件
		with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
			self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
		# 随机生成每个类别的颜色
		self.colors = [[np.random.randint(0, 255) for _ in range(3)]  for class_id in range(self.yolov5['class_num'])]
		# 模型初始化
		self.init_model()
	
	@torch.no_grad()
	def init_model(self):
		'''模型初始化'''
		# 设置日志输出
		set_logging()
		# 选择计算设备
		device = select_device(self.yolov5['device'])
		# 如果是GPU则使用半精度浮点数 F16
		is_half = device.type != 'cpu'
		# 载入模型
		model = attempt_load(self.yolov5['weight'], map_location=device)  # 载入全精度浮点数的模型
		input_size = check_img_size(self.yolov5['input_size'], s=model.stride.max())   # 检查模型的尺寸
		if is_half:
			model.half()  # 将模型转换为半精度
		# 设置BenchMark，加速固定图像的尺寸的推理
		cudnn.benchmark = True  # set True to speed up constant image size inference
		# 图像缓冲区初始化
		img_torch = torch.zeros((1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)  # init img
		# 创建模型
		_ = model(img_torch.half() if is_half else img) if device.type != 'cpu' else None  # run once
		self.is_half = is_half # 是否开启半精度
		self.device = device # 计算设备
		self.model = model # Yolov5模型
		self.img_torch = img_torch # 图像缓冲区
	
	def preprocessing(self, img):
		'''图像预处理'''
		# 图像缩放
		# 注: auto一定要设置为False -> 图像的宽高不同
		img_resize = letterbox(img, new_shape=(self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
		# print("img resize shape: {}".format(img_resize.shape))
		# 增加一个维度
		img_arr = np.stack([img_resize], 0)
		# 图像转换 (Convert) BGR格式转换为RGB
		# 转换为 bs x 3 x 416 x
		# 0(图像i), 1(row行), 2(列), 3(RGB三通道)
		# ---> 0, 3, 1, 2
		img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
		# 数值归一化
		# img_arr =  img_arr.astype(np.float32) / 255.0
		# 将数组在内存的存放地址变成连续的(一维)， 行优先
		# 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
		# https://zhuanlan.zhihu.com/p/59767914
		img_arr = np.ascontiguousarray(img_arr)
		return img_arr

	@torch.no_grad()
	def detect(self, img, canvas=None, view_img=True):
		'''模型预测'''
		# 图像预处理
		img_resize = self.preprocessing(img) # 图像缩放
		self.img_torch = torch.from_numpy(img_resize).to(self.device) # 图像格式转换
		self.img_torch = self.img_torch.half() if self.is_half else self.img_torch.float() # 格式转换 uint8-> 浮点数
		self.img_torch /=  255.0 # 图像归一化
		if self.img_torch.ndimension() == 3:
			self.img_torch = self.img_torch.unsqueeze(0)
		# 模型推理
		t1 = time_sync()
		pred = self.model(self.img_torch, augment=False)[0]
		# pred = self.model_trt(self.img_torch, augment=False)[0]
		# NMS 非极大值抑制
		pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'], \
			self.yolov5['threshold']['iou'], classes=None, agnostic=False)
		t2 = time_sync()
		# print("推理时间: inference period = {}".format(t2 - t1))
		# 获取检测结果
		det = pred[0]
		gain_whwh = torch.tensor(img.shape)[[1, 0, 1, 0]] # [w, h, w, h]
	

		if view_img and canvas is None:
			canvas = np.copy(img)
		xyxy_list = []
		conf_list = []
		class_id_list = []
		if det is not None and len(det):
			# 画面中存在目标对象
			# 将坐标信息恢复到原始图像的尺寸
			det[:, :4] =  scale_coords(img_resize.shape[2:], det[:, :4], img.shape).round()
			for *xyxy, conf, class_id in reversed(det):
					class_id = int(class_id)
					xyxy_list.append(xyxy)
					conf_list.append(conf)
					class_id_list.append(class_id)
					if  view_img:
						# 绘制矩形框与标签
						label = '%s %.2f' % (self.yolov5['class_name'][class_id], conf)
						self.plot_one_box(xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)
		return canvas, class_id_list, xyxy_list, conf_list
	
	def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
		''''绘制矩形框+标签'''
		tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
		color = color or [random.randint(0, 255) for _ in range(3)]
		c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
		cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
		if label:
			tf = max(tl - 1, 1)  # font thickness
			t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
			c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
			cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
			cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
	from camera import Camera
	print("[INFO] YoloV5目标检测-程序启动")
	# 创建摄像头对象
	cam = Camera()
	cap =cam.get_video_capture()
	# 创建窗口
	cv2.namedWindow('canvas', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
 	# 加载模型
	print("[INFO] 开始YoloV5模型加载")
	# YOLOV5模型配置文件(YAML格式)的路径 yolov5_yaml_path
	model = YoloV5(yolov5_yaml_path='config/yolov5.yaml')
	print("[INFO] 完成YoloV5模型加载")
	
	while True:
		# 清空缓冲区
		ret, img = cap.read()
		# 获取工作台的图像
		if not ret:
			print("[Error] USB摄像头获取失败")
			break
		t_start = time.time() # 开始计时
		# YoloV5 目标检测
		canvas, class_id_list, xyxy_list, conf_list  = model.detect(img)
		t_end = time.time() # 结束计时
		# 添加fps显示
		fps = int(1.0/(t_end - t_start))
		cv2.putText(canvas, text="FPS: {}".format(fps), org=(50, 50), \
      		fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1,\
            lineType=cv2.LINE_AA, color=(0, 0, 255))
		# 可视化
		cv2.imshow("canvas", canvas)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break
	cv2.destroyAllWindows()
	cap.release()