import os
import subprocess
import re
from PIL import Image
import cv2
import numpy as np


def get_dominant_rgb(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.resize((1, 1), resample=0)
    dominant_color = np.asarray(img.getpixel((0, 0)), "uint8")
    return dominant_color


def extract_audio_from_video(input_file_path, output_file_path, overwrite=False):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError("File not found")
    if os.path.exists(output_file_path) and not overwrite:
        return None
    print("Begin converting audio")
    p = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        output_file_path
    ])  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    p_status = p.wait()
    if p_status == 0:
        r = re.findall('(ERROR)+', str(output), flags=re.IGNORECASE)
        if r:
            print('An ERROR occurred!')
        else:
            print("End")


def add_audio_to_video(input_vid, input_audio, output_vid):
    print("Begin adding audio")
    if not os.path.exists(input_vid):
        raise FileNotFoundError("Video file not found")
    if not os.path.exists(input_vid):
        raise FileNotFoundError("Audio file not found")
    p = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-i",
        input_vid,
        "-i",
        input_audio,
        # "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        output_vid
    ])  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    p_status = p.wait()
    if p_status == 0:
        r = re.findall('(ERROR)+', str(output), flags=re.IGNORECASE)
        if r:
            print('An ERROR occurred!')
        else:
            print("End")


def generate_video(input_rgb, output_file, fps, width, height):
    size = (width, height)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    last_update = -1
    for i, rgb in enumerate(input_rgb):
        frame = np.full((size[1], size[0], 3), rgb, dtype="uint8")
        out.write(frame)
        progress = int((i / len(input_rgb)) * 100)
        if progress != last_update:
            last_update = progress
            print(f"progress: {progress}%")
    out.release()


class Utils:
    pass
