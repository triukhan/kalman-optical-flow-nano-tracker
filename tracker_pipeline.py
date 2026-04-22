from pathlib import Path

import cv2

from models.model_builder import ModelBuilder
from nano_tracker import NanoTracker
from utils import load_pretrain


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'nano-track' / 'models' / 'pretrained' / 'nanotrackv3.pth'

def draw_box(frame, box, color, name):
    x, y, w, h = box

    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))

    cv2.rectangle(frame, pt1, pt2, color, 2)
    cv2.putText(
        frame,
        name,
        (int(x), int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


def track_object(video_path: Path, stop=False):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = load_pretrain(ModelBuilder(), MODEL_PATH).eval()
    model.eval()

    tracker = NanoTracker(model)
    cv2.namedWindow('tracking', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('tracking', tracker.on_mouse)
    count = 0

    while True:
        count += 1
        ret, frame = video.read()
        if not ret:
            break
        #

        if count in range(10, 100):
            print('skip')
            continue

        if tracker.need_init:
            tracker.init(frame, tracker.bbox)
            tracker.need_init = False

        if tracker.center_pos is not None:
            res = tracker.track(frame)
            filtered = res['filtered']
            draw_box(frame, filtered, (0, 255, 0), 'filtered')

        cv2.imshow('tracking', frame)

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


vid = PROJECT_ROOT / 'nano-track' / 'data' / '8177427-uhd_3840_2160_24fps.mp4'
track_object(vid, stop=True)
