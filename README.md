# reTerminal Virtual Mouse Demo Application

<p align="center">
  <img alt="Light" src="https://files.seeedstudio.com/wiki/reTerminal_ML/virtual_mouse.gif" width="100%">
</p>

Sample application for using hand landmarks MediaPipe model for controlling the mouse on Seeed Studio reTerminal. Will work on Linux PC and other SBCs as well, provided necessary requirements are met.

## Installation

Install additional dependencies:

```bash
pip3 install -r requirements.txt
```

[Install MediaPipe package](https://wiki.seeedstudio.com/reTerminal_ML_MediaPipe/)  - for faster inference install 64 bit version.

[Download the models](https://files.seeedstudio.com/ml/hand_lm_models/hand_lm_models.zip), extract and place .tflite files inside of models directory.

## Usage

Run 
```bash
DISPLAY=:0 python3 virtual_mouse.py
```
from the project main folder. 

Use index finger to control mouse cursor, close the palm to press and hold left mouse button for dragging, open the palm to release the left mouse button. 

## Acknowledgements and license

This project is based on other two projects from Github:

- https://github.com/ravigithub19/ai-virtual-mouse for mouse control, however for cross-platform support autopy was replaced with pyautogui. Additionally, instead of hard-coding the gesture classification algorithm, a set of two models from below project was used
- https://github.com/kinivi/hand-gesture-recognition-mediapipe Gesture classification project. Published under Apache 2.0 license, the models used still need some work and will be improved in the future.

## TO-DO

- [ ] Pre-trained models published under MIT License
- [ ] Better models, with increased accuracy for most important gestures
- [ ] On-device data collection and training to support user-defined gestures
- [ ] Support for Raspberry Pi camera