import cv2
import os
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

# 删除现有的输出目录
if os.path.exists("images") and os.path.isdir("images"):
    shutil.rmtree("images")
if os.path.exists("labels") and os.path.isdir("labels"):
    shutil.rmtree("labels")

def convert_xml_label_to_yolov_bbox_label():
    xml_path = "Annotations"  # XML文件夹路径
    img_base_path = "JPEGImages"  # 图片文件夹路径
    xml_files = glob.glob(xml_path + "/*.xml")
    
    # 定义输出文件夹
    output_img_dir = "images"  # 存放图像的路径
    output_label_dir = "labels"  # 存放标签的路径
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 定义类别列表
    classes = ['fire']

    # 划分数据集
    train_xml_files, val_xml_files = train_test_split(xml_files, test_size=0.1, random_state=42)

    # 处理训练集和验证集
    for xml_file in train_xml_files + val_xml_files:
        print(f"Processing XML file: {xml_file}")

        # 解析XML文件
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_file}: {e}")
            continue

        root = tree.getroot()

        # 获取文件名（不含扩展名）
        xml_basename = os.path.basename(xml_file).replace(".xml", "")
        # 在图片文件夹中查找同名图片（支持多种扩展名）
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(img_base_path, xml_basename + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        # 检查是否找到对应的图片
        if not img_path:
            print(f"No matching image found for XML file {xml_file}. Skipping.")
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}. Skipping XML file {xml_file}.")
            continue
        height, width, _ = img.shape

        # 处理输出文件路径
        if xml_file in train_xml_files:
            img_output_path = os.path.join(output_img_dir, 'train', os.path.basename(img_path))
            txt_output_path = os.path.join(output_label_dir, 'train', xml_basename + ".txt")
        else:
            img_output_path = os.path.join(output_img_dir, 'val', os.path.basename(img_path))
            txt_output_path = os.path.join(output_label_dir, 'val', xml_basename + ".txt")

        # 确保输出文件夹存在
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

        # 保存图像
        cv2.imwrite(img_output_path, img)

        # 生成TXT文件
        with open(txt_output_path, "w") as f:
            for obj in root.findall('object'):
                cls = obj.find('name').text
                if cls in classes:
                    cls_id = classes.index(cls)

                    # 获取边界框坐标
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    # 转换为YOLO格式（归一化后的中心点坐标和宽高）
                    x_center = (xmin + xmax) / 2.0 / width
                    y_center = (ymin + ymax) / 2.0 / height
                    bbox_width = (xmax - xmin) / width
                    bbox_height = (ymax - ymin) / height

                    # 写入TXT
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    print("Finished processing all XML files.")

convert_xml_label_to_yolov_bbox_label()
