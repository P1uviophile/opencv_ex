import os
import zipfile

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory

from API_process import run

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_image():
    directory = "static/tmp"
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            # 删除文件
            os.remove(file_path)
    if 'file' in request.files:
        image = request.files['file']
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        original_img, thumbnail_img = run(img=img)

        # 将原图和缩略图保存在服务器上的临时文件中
        temp_original_file = "static/tmp/original.jpg"  # 保存原图
        temp_thumbnail_file = "static/tmp/thumbnail.jpg"  # 保存缩略图
        cv2.imwrite(temp_original_file, original_img)
        cv2.imwrite(temp_thumbnail_file, thumbnail_img)

        # 发送缩略图给客户端
        return send_file(temp_thumbnail_file, mimetype='image/jpeg', as_attachment=False)
    else:
        return jsonify({'error': 'no file'}), 400


@app.route('/download_original')
def download_original():
    # 从服务器上的临时文件中发送原图
    return send_from_directory('static/tmp', 'original.jpg', as_attachment=True, mimetype='image/jpeg')


@app.route('/download_Symbol')
def download_Symbol():
    return send_from_directory('camera_parameters', 'symbol.png', as_attachment=True, mimetype='image/png')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/single_img')
def single_img():
    return render_template("single_img.html")


@app.route('/multiple_img')
def multiple_img():
    return render_template('multiple_img.html')


@app.route('/show_img')
def show_img():
    return render_template('show.html')


@app.route('/upload_files', methods=['POST'])
def upload_files():
    directory = "static/tmp"
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            # 删除文件
            os.remove(file_path)
    if 'files' in request.files:
        files = request.files.getlist('files')
        processed_images = []
        for image in files:
            img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
            original_img, thumbnail_img = run(img=img)

            # 将处理后的图片保存在列表中
            processed_images.append(original_img)

        # 保存所有处理后的图片到临时文件夹
        temp_files = save_processed_images(processed_images)

        # 创建压缩包
        zip_file = create_zip(temp_files)

        # 发送压缩包给客户端
        return send_file(zip_file, mimetype='application/zip', as_attachment=True,
                         download_name='processed_images.zip')
    else:
        return jsonify({'error': 'no files'}), 400


def save_processed_images(images):
    # 保存所有处理后的图片到临时文件夹
    temp_files = []
    for i, image in enumerate(images):
        temp_file = f"static/tmp/processed_{i}.jpg"
        cv2.imwrite(temp_file, image)
        temp_files.append(temp_file)
    return temp_files


def create_zip(file_names):
    # 创建压缩包
    zip_file = 'static/tmp/processed_images.zip'
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        for file_name in file_names:
            zipf.write(file_name, os.path.basename(file_name))
    return zip_file


@app.route('/download_zip')
def download_zip():
    return send_from_directory('static/tmp', 'processed_images.zip', as_attachment=True, mimetype='application/zip')


if __name__ == "__main__":

    directory = "static/tmp"
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            # 删除文件
            os.remove(file_path)

    app.run(debug=True)
