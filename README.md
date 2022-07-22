# OpenVINO Noise Supression Demo (Mic Input/Speaker Output Support Version)

This program is a customized version of [Noise Suppression Python* Demo](https://docs.openvino.ai/latest/omz_demos_noise_suppression_demo_python.html) in OpenVINO Open Model Zoo Demos. This version supports to input sound from mic device and output noise suppressed sound to speaker device.

## Prerequisites
* Ubuntu 20.04
* Python 3.8
* OpenVINO 2022.1

## Setup
### Install modules
```
sudo apt install portaudio19-dev
pip install -r requirements.txt
```

### Download Pre-trained Model
```
pip install openvino-dev
omz_downloader --name noise-suppression-poconetlike-0001 
```

## Test
### Case 1: Use audio file.
```
wget https://assets.amazon.science/ef/0b/234f82204da385f4893a150d7e34/sample01-orig.wav
python noise_suppression_demo_mic.py -m intel/noise-suppression-poconetlike-0001/FP32/noise-suppression-poconetlike-0001.xml -i sample01-orig.wav -o sample01-output.wav
aplay -D plughw:2,0 sample01-output.wav
```

### Case 2: Use sound input/output devices.  
1. Dump index of audio device by using -l option.
```
python noise_suppression_demo_mic.py -m intel/noise-suppression-poconetlike-0001/FP32/noise-suppression-poconetlike-0001.xml -l 
```
2. Pick up index numbers for mic and speaker devices in output message of the above command.
```
{'index': 10, 'structVersion': 2, 'name': 'HDA Intel PCH: HDMI 10 (hw:0,16)', 'hostApi': 0, 'maxInputChannels': 0, 'maxOutputChannels': 8, 'defaultLowInputLatency': -1.0, 'defaultLowOutputLatency': 0.008707482993197279, 'defaultHighInputLatency': -1.0, 'defaultHighOutputLatency': 0.034829931972789115, 'defaultSampleRate': 44100.0}
{'index': 11, 'structVersion': 2, 'name': 'USB PnP Sound Device: Audio (hw:1,0)', 'hostApi': 0, 'maxInputChannels': 1, 'maxOutputChannels': 0, 'defaultLowInputLatency': 0.008707482993197279, 'defaultLowOutputLatency': -1.0, 'defaultHighInputLatency': 0.034829931972789115, 'defaultHighOutputLatency': -1.0, 'defaultSampleRate': 44100.0}
{'index': 12, 'structVersion': 2, 'name': 'Logitech USB Headset: Audio (hw:2,0)', 'hostApi': 0, 'maxInputChannels': 1, 'maxOutputChannels': 2, 'defaultLowInputLatency': 0.008707482993197279, 'defaultLowOutputLatency': 0.008707482993197279, 'defaultHighInputLatency': 0.034829931972789115, 'defaultHighOutputLatency': 0.034829931972789115, 'defaultSampleRate': 44100.0}
```
3. Run demo by specifying index numbers above. 
```
python noise_suppression_demo_mic.py -m intel/noise-suppression-poconetlike-0001/FP32/noise-suppression-poconetlike-0001.xml --device_input 11 --device_output 12 -i test_input.wav -o test_output.wav -t 50
```

```
usage: noise_suppression_demo_mic.py [-h] -m MODEL [-i INPUT] [-o OUTPUT] [-d DEVICE] [--device_input DEVICE_INPUT] [--device_output DEVICE_OUTPUT] [--skip_infer] [-t RECORD_TIME] [-l]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -i INPUT, --input INPUT
                        Optional. Path to a 16kHz wav file with speech+noise
  -o OUTPUT, --output OUTPUT
                        Optional. Path to output wav file for cleaned speech
  -d DEVICE, --device DEVICE
                        Optional. Target device to perform inference on. Default value is CPU
  --device_input DEVICE_INPUT
                        Optional. device id for input
  --device_output DEVICE_OUTPUT
                        Optional. device id for output
  --skip_infer          Optional. skip inference
  -t RECORD_TIME, --record_time RECORD_TIME
                        Optional. Recording time [sec]
  -l, --list_device     Optional. show a list of audio device
```

> **Note**
> * OpenVINO API 2.0 is used. Inference Engine API is not supported. 
> * Not support noise-suppression-denseunet-ll-0001


