Project goal, at first, figure out how to work Siamese trackers in general and Nanotrack especially + Kalman filter.
Almost all code was taken from https://github.com/HonglinChu/SiamTrackers, but I figured out how it works, figured out what kind of Siamise tracker I need and added some improvements.

<strong>Improvements:</strong>
- Kalman filter in ```nano_tracker.py```: At first - stabilization, the most common usage for tracking. At second - lost mode, when tracker is lost it will be conducted only by Kalman filter trajectory.
- Optical flow as additional source of tracking.
- Combining results from tracker, optical flow and kalman prediction.
- Re-initializing when tracker confident. Needs for successful tracking after object changing during the track.
- Resizing input video to 720p, caching channel_average.

Now tracker works in 18 fps with 720p video.

So, the main goal is track not only contrast objects but plain parts of something like: forrest, ground, grass. After adding optical-flow tracker became more stable for this goal.

Pain: As I understood, If i want to fine-tune my model to this goal, I need to create specific dataset for it, with non-object bboxes so it will take time...

Test:
```sh
python -m venv venv
pip install -r requirements.txt
source venv/bin/activate
python tracker_pipeline.py
```
