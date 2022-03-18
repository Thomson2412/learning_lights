import json
import os
import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from sklearn.preprocessing import normalize
import Utils
from Model import Model
import pandas as pd
from matplotlib import pyplot as plt


def extract_data(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            input_file_path = os.path.join(root, filename)
            filename_split = os.path.splitext(filename)
            output_dir_path = os.path.join(output_path, filename_split[0])
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)

            audio_file_path = os.path.join(output_dir_path, f"{filename_split[0]}.wav")
            Utils.extract_audio_from_video(input_file_path, audio_file_path)
            samplerate, audio_data = wavfile.read(audio_file_path)
            audio_data = normalize(audio_data)
            if audio_data.shape[1] == 2:
                audio_data = audio_data.sum(axis=1) / 2
            duration = audio_data.shape[0] / samplerate

            freq_range_split = 800

            frame_count = 0
            cap = cv2.VideoCapture(input_file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            samples_per_frame = int(samplerate / fps)

            frame_data_list = list()

            while cap.isOpened():
                rtn, frame = cap.read()
                if rtn:
                    start_sample = samples_per_frame * frame_count
                    end_sample = min((samples_per_frame * frame_count) + samples_per_frame, audio_data.shape[0])
                    audio_frame = audio_data[start_sample:end_sample]
                    fft = np.abs(rfft(audio_frame))
                    fft_split = list()

                    split_place = np.floor(
                        np.flip(np.abs(np.logspace(0, np.log10(len(fft)), freq_range_split) - len(fft))))
                    for i in range(freq_range_split):
                        split_start = int(split_place[i])
                        if i == freq_range_split - 1:
                            split_end = len(fft)
                        else:
                            split_end = int(split_place[i + 1])
                        fft_split.append(np.sum(fft[split_start:split_end]))

                    dominant_color = Utils.get_dominant_rgb(frame)
                    # row = [dominant_color[0], dominant_color[1], dominant_color[2]]
                    row = [dominant_color[0], dominant_color[1], dominant_color[2]]
                    row.extend(fft_split)
                    frame_data_list.append(row)
                    frame_count = frame_count + 1
                else:
                    break
            cap.release()
            # fft_freq = rfftfreq(samples_per_frame, 1 / samplerate)
            columns = ["d_r", "d_g", "d_b"]
            # columns = ["d_rgb"]
            columns.extend(range(freq_range_split))
            dataframe = pd.DataFrame(frame_data_list, columns=columns)
            dataframe.to_json(os.path.join(output_dir_path, f"{filename_split[0]}.json"), indent=4)


def create_prediction_result(trained_model, fps, input_file, output_path):
    samplerate, audio_data = wavfile.read(input_file)
    audio_data = normalize(audio_data)
    if audio_data.shape[1] == 2:
        audio_data = audio_data.sum(axis=1) / 2

    fft_result = list()
    last_update = -1

    samples_per_frame = int(samplerate / fps)
    steps = int(audio_data.shape[0] / samples_per_frame)
    for step in range(steps):
        start_sample = samples_per_frame * step
        end_sample = min((samples_per_frame * step) + samples_per_frame, audio_data.shape[0])
        audio_frame = audio_data[start_sample:end_sample]
        fft = np.abs(rfft(audio_frame))
        freq_range_split = 800
        fft_split = list()
        split_place = np.floor(
            np.flip(np.abs(np.logspace(0, np.log10(len(fft)), freq_range_split) - len(fft))))
        for split in range(freq_range_split):
            split_start = int(split_place[split])
            if split == freq_range_split - 1:
                split_end = len(fft)
            else:
                split_end = int(split_place[split + 1])
            fft_split.append(np.sum(fft[split_start:split_end]))

        progress = int((step / steps) * 100)
        if progress != last_update:
            last_update = progress
            print(f"progress: {progress}%")
        fft_result.append(fft_split)

    pred = trained_model.predict(fft_result).astype("uint8")
    print(f"progress: 100%")

    with open(os.path.join(output_path, f"{os.path.splitext(os.path.basename(input_file))[0]}_result.json"),
              "w") as outfile:
        json.dump(pred.tolist(), outfile)


def create_video(input_result, input_audio, output_path, fps):
    with open(input_result, "r") as infile:
        result = json.load(infile)
        intermediate_filename = f"{os.path.splitext(output_path)[0]}.avi"
        Utils.generate_video(result, intermediate_filename, fps, 1280, 720)
        Utils.add_audio_to_video(intermediate_filename, input_audio, output_path)


if __name__ == "__main__":
    extract_data("data/training/input_video", "data/training/output_data")
    model = Model("data/training/output_data/lazerface_cut_10m/lazerface_cut_10m.json", 0.2)
    model.train_model(1000, 10)
    # model.predict_plot()
    create_prediction_result(model, 30, "data/result/input_audio/beepboop.wav", "data/result/output_rgb")
    create_video(
        "data/result/output_rgb/beepboop_result.json",
        "data/result/input_audio/beepboop.wav",
        "data/result/output_video/beepboop.mp4",
        30
    )

