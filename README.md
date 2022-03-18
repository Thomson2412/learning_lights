# learning_lights

## Install

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Training

Two directories need to be manually created, and the corresponding video and audio files need to be placed there:
* Training video: `data/training/input_video`
* Result audio: `data/result/input_audio`

Start the training and generate the light video from the result audio in one go:
```bash
python main.py
```

## Parameters

Something that can be adjusted to increase the number of training features:
* `freq_range_split = 10`