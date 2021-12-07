'''摄像头 Camera
功能列表
1. 相机拍摄画面预览
1. 图像采集并保存在特定的路径
2. UVC相机参数的可视化调参

备注:
1. 不同摄像头型号的可设置的参数及取值范围各不相同.
------------------------
作者: 阿凯爱玩机器人
微信: xingshunkai
邮箱: xingshunkai@qq.com
更新时间: 2021/12/07'''
import time
import cv2
import numpy as np
import pickle
import subprocess
import math
import yaml
import logging

class Camera:
	'''Camera全局曝光全局快门UVC免驱摄像头'''
	# CAM_PORT_NAME = 0 # '/dev/video1'
	# IMG_WIDTH = 640 # 图像宽度
	# IMG_HEIGHT = 480 # 图像高度
	# CAM_FPS = 30 # 帧率 (实际上达不到)
	# CAM_BUFFERSIZE = 2 # 缓冲区的尺寸
	
	IS_DEBUG = True
	def __init__(self):
		with open('config/camera.yaml', 'r', encoding='utf-8') as f:
			self.config = yaml.load(f.read(), Loader=yaml.SafeLoader)
		self.init_camera()

	def init_camera(self):
		'''UVC摄像头初始化'''
		pass
	
	def get_video_capture(self):
		'''生成Capture对象'''
		capture = None
		# capture = cv2.VideoCapture(int(self.config['device']), cv2.CAP_V4L2)
		capture = cv2.VideoCapture(int(self.config['device']))
  		# try:
		# 	# Linux
		# 	capture = cv2.VideoCapture(int(self.device[-1]), cv2.CAP_V4L2)
		# except TypeError as e:
		# 	# Windows
		# 	capture = cv2.VideoCapture(int(self.device[-1]))
		# 设置分辨率
		capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config['img_height']))#设置图像高度
		capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config['img_width'])) #设置图像宽度
		# self.init_camera()
		capture.set(cv2.CAP_PROP_FPS,  int(self.config['fps']))
		capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 设置编码方式	
		# 缓冲区设置为1结果就是帧率只有 15fps
		# 缓冲区设置为2之后帧率提升到40FPS
		capture.set(cv2.CAP_PROP_BUFFERSIZE, int(self.config['buffer_size']) )#设置视频缓冲区为3
		self.capture = capture
		return capture
	
	def load_cam_calib_data(self, file_path='config/camera_info.bin'):
		'''载入相机标定数据'''
		# 读取标定参数
		with open(file_path, 'rb') as f:
			camera_info = pickle.load(f)
			# 获取摄像头内参
			self.intrinsic = camera_info['intrinsic']
			# 获取摄像头的畸变系数
			self.distortion = camera_info['distortion']
			# x轴的映射
			self.remap_x = camera_info['remap_x']
			# y轴映射
			self.remap_y = camera_info['remap_y']
			# 根据相机标定参数
			# 提取图像中心(cx, cy)与焦距f(单位：像素)
			self.f = (self.intrinsic[0, 0] + self.intrinsic[1, 1])/2
			# 图像中心的坐标
			self.cx = self.intrinsic[0, 2]
			self.cy = self.intrinsic[1, 2]
			# 生成视场角等相关参数
			self.alpha1 = np.arctan(self.cy/self.f)
			self.alpha2 = np.arctan((self.config['img_height']-self.cy)/self.f)
			self.beta1 = np.arctan(self.cx/self.f)
			self.beta2 = np.arctan((self.config['img_width']-self.cx)/self.f)

	def remove_distortion(self, image):
		'''图像去除畸变'''
		return cv2.remap(image, self.remap_x, self.remap_y, cv2.INTER_LINEAR)

	def empty_cache(self):
		'''清空摄像头的缓冲区'''
		for i_frame in range(self.config['buffer_size']):
			ret, frame = self.capture.read()

def main(argv):
	'''调整相机参数, 预览图像'''
	img_cnt = FLAGS.img_cnt
	# 创建相机对象
	camera = Camera()
	# 初始相机
	camera.init_camera()
	capture = camera.get_video_capture()
	
	if FLAGS.rm_distortion:
		# 载入标定数据
		camera.load_cam_calib_data()
	# 创建一个名字叫做 “image_win” 的窗口
	win_name = 'image_win'
	cv2.namedWindow(win_name,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

	fps = 40 # 设定一个初始值
	while True:
		start = time.time()
		ret, image = capture.read()
		
		if not ret:
			logging.error('图像获取失败')
			break
		if FLAGS.rm_distortion:
			# 图像去除畸变
			image = camera.remove_distortion(image)
		
		# 创建画布
		canvas = np.copy(image)
		# 添加帮助信息
		cv2.putText(canvas, text='S:Save Image',\
			 	org=(50, camera.config['img_height']-100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
				fontScale=1, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))
		cv2.putText(canvas, text='Q: Quit',\
			 	org=(50, camera.config['img_height']-50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
				fontScale=1, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))

		# 在画布上添加帧率的信息
		cv2.putText(canvas, text='FPS: {}'.format(fps),\
			 	org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
				fontScale=1, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))
		# 更新窗口“image_win”中的图片
		cv2.imshow('image_win', canvas)

		end = time.time()
		fps = int(0.6*fps +  0.4*1/(end-start))
		# fps = 1
		key = cv2.waitKey(1)
		
		if key == ord('q'):
			# 如果按键为q 代表quit 退出程序
			break
		elif key == ord('s'):
			# s键代表保存数据
			cv2.imwrite('{}/{}.png'.format(FLAGS.img_path, img_cnt), image)
			logging.info("截图，并保存在  {}/{}.png".format(FLAGS.img_path, img_cnt))
			img_cnt += 1
	
	# 关闭摄像头
	capture.release()
	# 销毁所有的窗口
	cv2.destroyAllWindows()

if __name__ == '__main__':
	import logging
	import sys
	from absl import app
	from absl import flags

	# 设置日志等级
	logging.basicConfig(level=logging.INFO)

	# 定义参数
	FLAGS = flags.FLAGS
	# flags.DEFINE_integer('device', 0, '摄像头的设备号')
	flags.DEFINE_integer('img_cnt', 0, '图像计数的起始数值')
	flags.DEFINE_string('img_path', 'data/image_raw', '图像的保存地址')
	flags.DEFINE_boolean('rm_distortion', False, '载入相机标定数据, 去除图像畸变')
	
	# 运行主程序
	app.run(main)
	
