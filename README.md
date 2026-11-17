# Vision-Based People Counting System

A real-time computer vision system for detecting, tracking and counting people using YOLOv8 and OpenCV.

## Overview

This project explores the use of computer vision for automated people counting and occupancy monitoring.

The system uses a webcam to capture video, a pretrained YOLOv8 model to detect people, and object tracking to monitor their movement across a virtual counting line.

It maintains separate counts for people entering and exiting a monitored area.

## Features

- Real-time person detection using YOLOv8.
- Object tracking with persistent IDs.
- Entry and exit detection using virtual line crossing.
- Current occupancy calculation.
- Peak occupancy tracking.
- Smoothed detection counts to reduce fluctuations.

## Technologies

- Python
- OpenCV
- NumPy
- Ultralytics YOLOv8

## How It Works

1. The webcam captures live video.
2. YOLOv8 detects people in each frame.
3. The tracking algorithm assigns IDs to detected individuals.
4. A virtual line determines the direction of movement.
5. The system updates entry, exit and occupancy counts.
6. Results are displayed in real time.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download and run the project using:

```bash
python src/people_counter.py
```

Press `q` to close the camera window.

## Current Development Status

The initial software prototype implements real-time person detection, tracking and entry/exit counting.

### Planned Improvements

- Improve tracking reliability.
- Integrate an Arduino Uno.
- Display occupancy on an LCD.
- Add LED-based capacity indicators.
- Integrate RFID attendance logging.
- Store attendance records on an SD card.
- Evaluate counting accuracy under different conditions.

## Project Goal

To develop an affordable vision-based people counting and occupancy monitoring system for training centres, events and other shared spaces.
