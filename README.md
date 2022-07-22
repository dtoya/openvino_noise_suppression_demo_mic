

'''
(openvino-dev) $ omz_downloader --name noise-suppression-poconetlike-0001 -o /data/intel/open_model_zoo/models
'''

'''
$ python3 -m venv venv
$ . venv/bin/activate
(venv) $ pip install -U pip
(venv) $ pip install openvino==2022.1.0
(venv) $ sudo apt install portaudio19-dev
(venv) $ pip install pyaudio
wget https://assets.amazon.science/ef/0b/234f82204da385f4893a150d7e34/sample01-orig.wav -O test_input.wav
'''

'''
python noise_suppression_demo_mic.py -m /data/intel/open_model_zoo/models/intel/noise-suppression-poconetlike-0001/FP32/noise-suppression-poconetlike-0001.xml -i test_input.wav -o test_output.wav
python noise_suppression_demo_mic.py -m /data/intel/open_model_zoo/models/intel/noise-suppression-poconetlike-0001/FP32/noise-suppression-poconetlike-0001.xml --device_input 7 --device_output 6 -i test_input.wav -o test_output.wav
'''
