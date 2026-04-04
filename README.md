# Sign Language Dataset Builder

## Overview

This project is used to **collect hand sign data** for training a sign language recognition model.

It uses:

* Hand tracking from **Mediapipe**
* Camera input from **OpenCV**
* Data storage using **Pandas**

---

## Requirements

Make sure you install the following dependencies:

* **Python**: 3.11.9 or 3.12.x
* **Mediapipe**: 0.10.21
* **OpenCV**: 4.8.0.74
* **Pandas**: 3.0.1
## How to Record Data

### General Instructions

* Follow instructions shown in the terminal
* Only **one hand** should appear in the frame
* Keep the correct hand sign during recording

---

### Recording Process

* Each session records **600 frames per letter**
* The program pauses every **100 frames**

  * During pause: rotate your hand slightly
  * Keep the same sign → increase data diversity

---

### Handling Issues

* If tracking is laggy:

  * Continue until 600 frames
  * Press **R** to replay
* If data is incorrect:

  * Replay and record again

---

## Notes

* Letters **J** and **Z** are NOT included
  → Because they require **motion (sequence of frames)**
  → These will be handled using deep learning models (e.g. TensorFlow)

---

## Output

* 600 frames per letter
* Total dataset: **14,400 frames**

---

## Goal

Create a **high-quality dataset** for training an accurate sign language recognition model.

---

## Future Improvements

* Add support for dynamic signs (J, Z)
* Train sequence models (LSTM / CNN + RNN)
* Improve real-time prediction

---
