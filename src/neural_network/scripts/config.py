#!/usr/bin/env python3
### CONFIG FILE ###

## Tkinter Config
windowColor = '#5c31ad'
windowSize = ''


## .csv Data Config
header = ['filename', 'type', 'threshold', 'epoch', 'expectation', 'success', 'failure']


## Detection config
modelPath = "pretrained_weights/mask_rcnn_object_0010.h5"
classNumber = 1 + 1

filterValue = 0.5
maxMasks = 1

detectImageBrightness = 30
detectImageContrast = 0.5

detectImageROIyMin = 200
detectImageROIyMax = 870
detectImageROIxMin = 520
detectImageROIxMax = 1200

## OpenCV config
kernelSizeBlur = 5
kernelSizeMorphology = 5

drawContourHull = False
drawContourRaw = True

drawCenterPoint = True
drawReferencePoint = True

showMask = False