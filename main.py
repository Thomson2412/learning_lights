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
            if "mp4" in filename_split[1]:
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

                freq_range_split = 10

                last_update = -1
                frame_count = 0
                cap = cv2.VideoCapture(input_file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames_ish = cap.get(cv2.CAP_PROP_FRAME_COUNT)
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
                        if np.sum(dominant_color) < (255 * 3) / 2:
                            dominant_color = [0, 0, 0]

                        row = [dominant_color[0], dominant_color[1], dominant_color[2]]
                        row.extend(fft_split)
                        frame_data_list.append(row)
                        frame_count = frame_count + 1

                        progress = int((frame_count / total_frames_ish) * 100)
                        if progress != last_update:
                            last_update = progress
                            print(f"{filename_split[0]} progress: {progress}%")
                    else:
                        break
                cap.release()
                # fft_freq = rfftfreq(samples_per_frame, 1 / samplerate)
                columns = ["d_r", "d_g", "d_b"]
                # columns = ["d_rgb"]
                columns.extend(range(freq_range_split))
                dataframe = pd.DataFrame(frame_data_list, columns=columns)
                dataframe.to_json(os.path.join(output_dir_path, f"{filename_split[0]}.json"), indent=4)


def create_prediction_result(trained_model, fps, input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            filename_split = os.path.splitext(filename)
            if "wav" in filename_split[1]:
                input_wav = os.path.join(root, filename)
                samplerate, audio_data = wavfile.read(input_wav)
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
                    freq_range_split = 10
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
                        print(f"Prediction progress: {progress}%")
                    fft_result.append(fft_split)

                pred = trained_model.predict(fft_result).astype("uint8")
                print(f"Prediction progress: 100%")

                with open(os.path.join(output_path, f"{filename_split[0]}.json"),
                          "w") as outfile:
                    json.dump(pred.tolist(), outfile)


def create_result_videos(input_path, input_audio_path, output_path, fps):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            filename_split = os.path.splitext(filename)
            if "json" in filename_split[1]:
                input_json = os.path.join(root, filename)
                with open(input_json, "r") as infile:
                    result = json.load(infile)
                    intermediate_file_path = f"{os.path.splitext(output_path)[0]}.avi"
                    Utils.generate_result_video(result, intermediate_file_path, fps, 1280, 720)
                    Utils.add_audio_to_video(
                        intermediate_file_path,
                        os.path.join(input_audio_path, f"{filename_split[0]}.wav"),
                        os.path.join(output_path, f"{filename_split[0]}.mp4")
                    )
                    os.remove(intermediate_file_path)


def create_training_videos(input_json_wav_path, input_video_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for root, dirs, files in os.walk(input_json_wav_path):
        for filename in files:
            filename_split = os.path.splitext(filename)
            if "json" in filename_split[1]:
                input_json = os.path.join(root, filename)
                with open(input_json, "r") as infile:
                    training_data = json.load(infile)
                    intermediate_file_path = os.path.join(output_path, f"{filename_split[0]}.avi")

                    cap = cv2.VideoCapture(os.path.join(input_video_path, f"{filename_split[0]}.mp4"))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    Utils.generate_training_video(training_data, intermediate_file_path, fps, 1280, 720)
                    Utils.add_audio_to_video(
                        intermediate_file_path,
                        os.path.join(os.path.join(input_json_wav_path, filename_split[0]), f"{filename_split[0]}.wav"),
                        os.path.join(output_path, f"{filename_split[0]}.mp4")
                    )
                    os.remove(intermediate_file_path)


if __name__ == "__main__":
    # extract_data("data/training/input_video", "data/training/output_data")
    model = Model("data/training/output_data/lazerface_cut_20m/lazerface_cut_20m.json", 0.2)
    model.train_model(1000, 10)
    # # model.predict_plot()

    set_fps = 30
    create_prediction_result(model, set_fps, "data/result/input_audio/", "data/result/output_rgb")
    create_result_videos(
        "data/result/output_rgb/",
        "data/result/input_audio/",
        "data/result/output_video/",
        set_fps
    )
    # create_training_videos(
    #     "data/training/output_data/",
    #     "data/training/input_video/",
    #     "data/training/output_video/"
    # )

