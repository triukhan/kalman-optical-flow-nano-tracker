Project goal, at first, figure out how to work Siamese trackers in general and Nanotrack especially + Kalman filter.
Almost all code was taken from https://github.com/HonglinChu/SiamTrackers, but I figured out how it works, figured out what kind of Siamise tracker I need and added some improvements.

Improvements:
- Kalman filter in nano_tracker.py: At first - stabilization, the most common usage for tracking. At second - lost mode, when tracker is lost it will be conducted only by Kalman filter trajectory.
- Re-detection in nano_tracker.py (_redetect): If tracker is lost - it will try to use search with doubled radius to find original object. 

So, the combination of these 2 improvements will provide mechanism (mostly based on assumptions) that help to re-detect objects after disappearing it behind the other objects.
