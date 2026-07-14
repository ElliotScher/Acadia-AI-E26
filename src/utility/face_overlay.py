import av
import cv2
import numpy as np
import requests
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

# ----------------------------
# Pose landmarker model
# ----------------------------

# Pose landmarks (nose/ears), not face mesh landmarks, are used to locate the
# head - the pose model estimates them from full-body context, so it keeps
# tracking a head that's turned to profile, partly occluded, or otherwise
# missing the facial features FaceLandmarker requires.
#
# The pip-installed mediapipe package only ships the C++ runtime; the model
# itself must be fetched separately, so it's cached locally on first run
# instead of shipping a multi-megabyte binary in the repo.
_MODEL_PATH = "pose_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _ensure_model() -> str:
    import os

    if not os.path.exists(_MODEL_PATH):
        response = requests.get(_MODEL_URL, timeout=30)
        response.raise_for_status()
        with open(_MODEL_PATH, "wb") as f:
            f.write(response.content)
    return _MODEL_PATH


# ----------------------------
# Load overlay image (RGBA)
# ----------------------------

overlay = cv2.imread("face.png", cv2.IMREAD_UNCHANGED)

if overlay is None:
    raise RuntimeError("Couldn't load face.png")

if overlay.ndim != 3 or overlay.shape[2] != 4:
    raise RuntimeError("Overlay image must have an alpha channel (PNG).")

overlay_rgb = overlay[:, :, :3]
overlay_alpha = overlay[:, :, 3] / 255.0

# ----------------------------
# Talking animation
# ----------------------------

# The jaw alternates shut/open every this many frames, regardless of audio.
_JAW_CYCLE_FRAMES = 2
# How far the lower third drops when open, as a fraction of the (unrotated)
# overlay height.
_JAW_DROP_RATIO = 0.18

# ----------------------------
# Setup MediaPipe
# ----------------------------

pose_landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        # Gates whether a person is detected in the frame at all; too low and
        # frames with no one in view (or no head visible) start producing
        # fabricated, garbage keypoint guesses. Left at the library default.
        min_pose_detection_confidence=0.7,
        # Gates whether an already-tracked person is kept once locked on -
        # lowered so a turned/occluded head doesn't drop tracking.
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.3,
    )
)

# ----------------------------
# Video
# ----------------------------

# Decoded via PyAV (not cv2.VideoCapture): OpenCV's bundled FFmpeg fails to
# decode AV1-encoded footage (common from phone/action-cam sources) on this
# platform, silently returning zero frames. PyAV bundles libdav1d and decodes
# it correctly.
container = av.open("input.mp4")
video_stream = container.streams.video[0]
input_audio_stream = container.streams.audio[0]

fps = float(video_stream.average_rate)
width = video_stream.codec_context.width
height = video_stream.codec_context.height

# Also written via PyAV (not cv2.VideoWriter): cv2.VideoWriter only handles
# video, so the audio track would silently disappear from the output. This
# script never touches the audio, so its packets are demuxed from the input
# and remuxed straight into the output rather than decoded/re-encoded.
output_container = av.open("output.mp4", mode="w")

output_video_stream = output_container.add_stream(
    "libx264", rate=video_stream.average_rate
)
output_video_stream.width = width
output_video_stream.height = height
output_video_stream.pix_fmt = "yuv420p"

output_audio_stream = output_container.add_stream_from_template(input_audio_stream)

# Pose landmark indices (see mediapipe.tasks.python.vision.PoseLandmark)
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8

# Pose landmarks don't include a crown/chin point the way face mesh did, so
# head height is estimated from ear-to-ear width using this average human
# head height/width ratio rather than measured directly.
HEAD_ASPECT_RATIO = 1.3

frame_idx = 0

for packet in container.demux(video_stream, input_audio_stream):
    if packet.stream.type == "audio":
        if packet.dts is None:  # flush packet at EOF - nothing to remux
            continue
        packet.stream = output_audio_stream
        output_container.mux(packet)
        continue

    for av_frame in packet.decode():
        frame = av_frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks[0]

            nose = np.array((landmarks[NOSE].x * width, landmarks[NOSE].y * height))

            # BlazePose's LEFT_EAR/RIGHT_EAR are anatomical (the subject's own
            # left/right), the opposite convention from FaceMesh's landmarks
            # (which are named from the viewer's perspective). For a normally
            # oriented, non-mirrored camera view, the subject's left ear is on
            # screen-right, so the mapping below is swapped to keep `left`/
            # `right` geometric (as seen on screen), matching the rotation math.
            left = np.array(
                (landmarks[RIGHT_EAR].x * width, landmarks[RIGHT_EAR].y * height)
            )
            right = np.array(
                (landmarks[LEFT_EAR].x * width, landmarks[LEFT_EAR].y * height)
            )

            center = (nose + left + right) / 3

            face_width = np.linalg.norm(right - left)
            face_height = face_width * HEAD_ASPECT_RATIO

            angle = np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))

            scale = 5

            new_w = int(face_width * scale)
            new_h = int(face_height * scale)

            # Guards against near-zero-confidence ear landmarks collapsing onto
            # each other, which would otherwise pass a zero/negative size to
            # cv2.resize and crash.
            if new_w > 0 and new_h > 0:

                # Alternates shut/open every _JAW_CYCLE_FRAMES frames, with no
                # regard for audio - a fixed animation cycle rather than a
                # loudness-driven one.
                jaw_open = (frame_idx // _JAW_CYCLE_FRAMES) % 2 == 0
                jaw_drop = int(_JAW_DROP_RATIO * new_h) if jaw_open else 0

                resized_rgb = cv2.resize(overlay_rgb, (new_w, new_h))
                resized_alpha = cv2.resize(overlay_alpha, (new_w, new_h))

                # South Park-style talking effect: split the overlay at the
                # boundary between its top two-thirds and its lower third
                # (roughly chin/mouth) and drop the lower third down by
                # jaw_drop, opening a gap between the pieces. Done in the
                # unrotated overlay's own local space (not screen space) so
                # the split still lines up with the head's own vertical axis
                # once rotated below.
                mid = int(new_h * 2 / 3)
                canvas_h = new_h + jaw_drop

                jaw_rgb = np.zeros((canvas_h, new_w, 3), dtype=resized_rgb.dtype)
                jaw_alpha = np.zeros((canvas_h, new_w), dtype=resized_alpha.dtype)

                jaw_rgb[:mid] = resized_rgb[:mid]
                jaw_alpha[:mid] = resized_alpha[:mid]

                # The gap opened up by the drop is filled with a black bar
                # (jaw_rgb is already zeroed) instead of transparent, so it
                # reads as a mouth cavity rather than a window onto the video
                # behind the overlay. Only the center half of the width is
                # covered, rather than the whole face width.
                bar_w = new_w // 2
                bar_x1 = (new_w - bar_w) // 2
                bar_x2 = bar_x1 + bar_w
                jaw_alpha[mid:mid + jaw_drop, bar_x1:bar_x2] = 1.0

                jaw_rgb[mid + jaw_drop:canvas_h] = resized_rgb[mid:new_h]
                jaw_alpha[mid + jaw_drop:canvas_h] = resized_alpha[mid:new_h]

                # cv2.getRotationMatrix2D treats a positive angle as counter-
                # clockwise, the opposite sign convention from atan2(dy, dx) in
                # image pixel coordinates (y grows downward) - negate here so the
                # overlay tilts the same direction as the measured head tilt
                # instead of mirrored. The rotation center stays anchored to the
                # unrotated overlay's own center (not the taller jaw-drop canvas)
                # so the tilt pivot doesn't shift as the mouth opens.
                M = cv2.getRotationMatrix2D(
                    (new_w / 2, new_h / 2),
                    -angle,
                    1.0,
                )

                rotated_rgb = cv2.warpAffine(
                    jaw_rgb,
                    M,
                    (new_w, canvas_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                rotated_alpha = cv2.warpAffine(
                    jaw_alpha,
                    M,
                    (new_w, canvas_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                x = int(center[0] - new_w / 2)
                y = int(center[1] - new_h / 2)

                x1 = max(x, 0)
                y1 = max(y, 0)
                x2 = min(x + new_w, width)
                y2 = min(y + canvas_h, height)

                ox1 = x1 - x
                oy1 = y1 - y
                ox2 = ox1 + (x2 - x1)
                oy2 = oy1 + (y2 - y1)

                if x2 > x1 and y2 > y1:

                    roi = frame[y1:y2, x1:x2]

                    alpha = rotated_alpha[oy1:oy2, ox1:ox2][..., None]
                    img = rotated_rgb[oy1:oy2, ox1:ox2]

                    roi[:] = (alpha * img + (1 - alpha) * roi).astype(np.uint8)

        output_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for out_packet in output_video_stream.encode(output_frame):
            output_container.mux(out_packet)

        frame_idx += 1

for out_packet in output_video_stream.encode():  # flush encoder
    output_container.mux(out_packet)

container.close()
output_container.close()
pose_landmarker.close()

print("Done!")
