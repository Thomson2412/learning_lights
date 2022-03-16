import os
import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from sklearn.preprocessing import normalize
import Utils
from Model import Model
import pandas as pd


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
            duration = audio_data.shape[0] * samplerate

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

                    # fft_list = list()
                    # for index, freq in enumerate(fft_freq):
                    #     fft_list.append((freq, fft[index]))

                    dominant_color = Utils.get_dominant_rgb(frame)
                    row = [dominant_color[0], dominant_color[1], dominant_color[2]]
                    row.extend(fft)
                    frame_data_list.append(row)
                    frame_count = frame_count + 1
                else:
                    break
            cap.release()
            fft_freq = rfftfreq(samples_per_frame, 1 / samplerate)
            columns = ["d_r", "d_g", "d_b"]
            columns.extend(fft_freq)
            dataframe = pd.DataFrame(frame_data_list, columns=columns)
            dataframe.to_json(os.path.join(output_dir_path, f"{filename_split[0]}.json"), indent=4)


if __name__ == "__main__":
    # extract_data("data/input_video", "data/output_data")
    model = Model("data/output_data/panpot_cut/panpot_cut.json", 0.2)
    model.train_model(1000, 10)
    model.predict()
