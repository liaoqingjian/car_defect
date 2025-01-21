from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class CarDefectDetectionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("汽车缺陷检测系统")
        self.root.geometry("1200x600")
        
        # 加载模型
        self.model = YOLO("model/car_defect.pt")
        
        # 存储当前图片
        self.current_image = None
        self.processed_image = None
        
        # 创建界面布局
        self.create_widgets()
        
        # 绑定窗口大小改变事件
        self.root.bind('<Configure>', self.on_window_resize)
        
    def create_widgets(self):
        # 创建左右框架
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        # 左侧布局
        tk.Label(self.left_frame, text="原始图片", font=('Arial', 12, 'bold')).pack()
        self.original_canvas = tk.Canvas(self.left_frame, bg='gray')
        self.original_canvas.pack(expand=True, fill=tk.BOTH)
        
        # 左侧按钮
        tk.Button(self.left_frame, text="选择图片", command=self.load_image,
                 width=15, height=2).pack(pady=10)
        
        # 右侧布局
        tk.Label(self.right_frame, text="检测结果", font=('Arial', 12, 'bold')).pack()
        self.result_canvas = tk.Canvas(self.right_frame, bg='gray')
        self.result_canvas.pack(expand=True, fill=tk.BOTH)
        
        # 右侧按钮
        tk.Button(self.right_frame, text="开始检测", command=self.detect_defects,
                 width=15, height=2).pack(pady=10)

    def on_window_resize(self, event):
        # 当窗口大小改变时，重新显示图片
        if self.current_image is not None:
            self.show_image(self.current_image, self.original_canvas)
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            self.show_image(self.processed_image, self.result_canvas)

    def show_image(self, cv_image, canvas):
        # 获取画布当前大小
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return  # 等待窗口真正加载完成
            
        # 计算缩放比例
        image_height, image_width = cv_image.shape[:2]
        width_ratio = canvas_width / image_width
        height_ratio = canvas_height / image_height
        scale = min(width_ratio, height_ratio)
        
        # 缩放图片
        new_width = int(image_width * scale)
        new_height = int(image_height * scale)
        resized_image = cv2.resize(cv_image, (new_width, new_height))
        
        # 转换颜色空间
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # 转换为PhotoImage
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_image))
        
        # 更新画布
        canvas.delete("all")
        
        # 计算居中位置
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        
        # 在画布中央显示图片
        canvas.create_image(x_center, y_center, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # 保持引用

    def detect_defects(self):
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择图片")
            return
            
        # 使用模型进行检测
        results = self.model(self.current_image, conf=0.5)
        
        # 在图片上绘制检测结果
        self.processed_image = self.current_image.copy()
        
        for result in results[0].boxes:
            # 提取边界框坐标
            box = result.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            # 提取类别和置信度
            class_id = int(result.cls[0])
            confidence = float(result.conf[0])
            class_name = self.model.names[class_id]
            
            # 获取类别颜色
            color = self.get_class_color(class_id)
            
            # 绘制边界框
            cv2.rectangle(self.processed_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制类别和置信度
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                self.processed_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
        # 显示处理后的图片
        self.show_image(self.processed_image, self.result_canvas)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            # 读取图片
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("错误", "无法加载图片")
                return
                
            # 显示原始图片
            self.show_image(self.current_image, self.original_canvas)
            # 清除之前的检测结果
            self.processed_image = None
            self.result_canvas.delete("all")

    @staticmethod
    def get_class_color(class_id):
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

if __name__ == "__main__":
    root = tk.Tk()
    app = CarDefectDetectionSystem(root)
    root.mainloop()

