# learning_lights

<p float="left">
    <img src="pix/output.gif" width="800" height="200"/>
</p>

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

## Try
* Create consistent training data to see how the model reacts
* Longer features and labels (FFT, RGB) than frame time
* Label classification [R, G, B]
* GAN
* Try with whole images instead of only dominant colors
