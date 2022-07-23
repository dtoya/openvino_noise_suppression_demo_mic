#!/usr/bin/env python3

"""
 Copyright (c) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import copy
import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import numpy as np
import wave
import pyaudio
import os

#from openvino.inference_engine import IECore, Blob
from openvino.runtime import Core, get_version 

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model", required=False, type=Path)
    args.add_argument("-i", "--input", help="Optional. Path to a 16kHz wav file with speech+noise", required=False, type=str)
    args.add_argument("-o", "--output", help="Optional. Path to output wav file for cleaned speech", required=False, type=str)
    args.add_argument("-d", "--device", help="Optional. Target device to perform inference on. Default value is CPU", default="CPU", type=str)
    args.add_argument("--device_input", required=False, type=int, help="Optional. device id for input")
    args.add_argument("--device_output", required=False, type=int, help="Optional. device id for output")
    args.add_argument("--skip_infer", required=False, action='store_true', default=False, help="Optional. skip inference")
    args.add_argument("-t", "--record_time", required=False, default=100, help="Optional. Recording time [sec]")
    args.add_argument("-l", "--list_device", required=False, action='store_true', default=False, help="Optional. show a list of audio device")
    return parser

def wav_read(wav_name):
    with wave.open(wav_name, "rb") as wav:
        if wav.getsampwidth() != 2:
            raise RuntimeError("wav file {} does not have int16 format".format(wav_name))
        if wav.getframerate() != 16000:
            raise RuntimeError("wav file {} does not have 16kHz sampling rate".format(wav_name))

        data = wav.readframes( wav.getnframes() )
        x = np.frombuffer(data, dtype=np.int16)
        x = x.astype(np.float32) * (1.0 / np.iinfo(np.int16).max)
        if wav.getnchannels() > 1:
            x = x.reshape(-1, wav.getnchannels())
            x = x.mean(1)
    return x

def wav_write(wav_name, x):
    x = (x*np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(wav_name, "wb") as wav:
        wav.setnchannels(1)
        wav.setframerate(16000)
        wav.setsampwidth(2)
        wav.writeframes(x.tobytes())

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    audio = pyaudio.PyAudio()
    if args.list_device:
        for i in range(audio.get_device_count()):
            print("[Audio Device List]")
            print(audio.get_device_info_by_index(i))
        sys.exit(0)

    log.info("Initializing Inference Engine")
    core = Core() 
    version = core.get_versions(args.device)[args.device]
    version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
    log.info("Plugin version is {}".format(version_str))

    # read IR
    model_xml = args.model
    model_bin = model_xml.with_suffix(".bin")
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    ov_encoder = core.read_model(model=model_xml, weights=model_bin)

    inp_shapes = {name: obj.shape for obj in ov_encoder.inputs for name in obj.get_names()}
    out_shapes = {name: obj.shape for obj in ov_encoder.outputs for name in obj.get_names()}

    state_out_names = [n for n in out_shapes.keys() if "state" in n]
    state_inp_names = [n for n in inp_shapes.keys() if "state" in n]
    if len(state_inp_names) != len(state_out_names):
        raise RuntimeError(
            "Number of input states of the model ({}) is not equal to number of output states({})".
                format(len(state_inp_names), len(state_out_names)))

    state_param_num = sum(np.prod(inp_shapes[n]) for n in state_inp_names)
    log.debug("State_param_num = {} ({:.1f}Mb)".format(state_param_num, state_param_num*4e-6))

    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    compiled_model = core.compile_model(ov_encoder, args.device)
    
    infer_request = compiled_model.create_infer_request()

    input_size = inp_shapes["input"][1]
    res = None

    index_in = None
    index_out = None
    if args.device_input: 
        index_in = args.device_input 
        print(audio.get_device_info_by_index(index_in))  
        if args.input:
            samples_in = []
    else:
        sample_inp = wav_read(args.input)

    if args.device_output:
        index_out = args.device_output 
        print(audio.get_device_info_by_index(index_out))
    else:
        if not args.output:
            args.output = "noise_suppression_demo_out.wav"

    fmt = pyaudio.paInt16  
    ch = 1              
    sampling_rate = 16000 
    chunk = 2**11   # 2048
    if args.device_input:  
        stream = audio.open(format=fmt, channels=ch, rate=sampling_rate, input=True, input_device_index = index_in, frames_per_buffer=chunk)
    if args.device_output:
        player = audio.open(format=fmt, channels=ch, rate=sampling_rate, output=True, output_device_index = index_out, frames_per_buffer=chunk)

    start_time = time.perf_counter()
    samples_out = []
    samples_times = []
    print('Start. Record {} sec.'.format(args.record_time))
    for i in range(int(args.record_time)):
        if args.device_input:
            data = stream.read(chunk)
            x = np.frombuffer(data, dtype=np.int16)
            x = x.astype(np.float32) * (1.0 / np.iinfo(np.int16).max)
            if ch > 1:
                x = x.reshape(-1, ch)
                x = x.mean(1)
            input = x
            if args.input:
                samples_in.append(input)
        else:
            if sample_inp is None or sample_inp.shape[0] == 0:
                break
            if sample_inp.shape[0] > input_size:
                input = sample_inp[:input_size]
                sample_inp = sample_inp[input_size:]
            else:
                input = np.pad(sample_inp, ((0, input_size - sample_inp.shape[0]), ), mode='constant')
                sample_inp = None
           
        #forms input
        inputs = {"input": input[None, :]}

        #add states to input
        for n in state_inp_names:
            if res:
                inputs[n] = infer_request.get_tensor(n.replace('inp', 'out')).data
            else:
                #on the first iteration fill states by zeros
                inputs[n] = np.zeros(inp_shapes[n], dtype=np.float32)

        t0 = time.perf_counter()
        if args.skip_infer:
            data_out = input
        else:
            infer_request.infer(inputs)
            res = infer_request.get_tensor("output")
            data_out = copy.deepcopy(res.data).squeeze(0) 
            
        t1 = time.perf_counter()
        if args.device_output:
            player.write((data_out*np.iinfo(np.int16).max).astype(np.int16), chunk)
        samples_out.append(data_out)
        samples_times.append(t1-t0)

    end_time = time.perf_counter()

    log.info("Sequence of length {:0.2f}s is processed by {:0.2f}s".format(
        sum(s.shape[0] for s in samples_out)/16000,
        sum(samples_times),
    ))
    total_time = end_time - start_time    
    log.info("Total time: {:0.2f}s".format(total_time))

    if args.device_input and args.input:
        samples_in = np.concatenate(samples_in, 0)
        basepath, ext = os.path.splitext(args.input) 
        wav_write(basepath+ext, samples_in)
   
    if args.output:
        sample_out = np.concatenate(samples_out, 0)
        wav_write(args.output, sample_out)

    if args.device_input or args.device_output:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == '__main__':
    sys.exit(main() or 0)
