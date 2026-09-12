import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# EXPERIMENT 7
# Motion Estimation using Optical Flow Algorithms
# Lucas-Kanade Sparse Optical Flow
# Farneback Dense Optical Flow
# ============================================================


# ------------------------------------------------------------
# PART 1: LOAD VIDEO
# ------------------------------------------------------------

video_path = Path(__file__).with_name("video.mp4")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Could not open video.")
    print(f"Make sure this file is readable: {video_path}")
    exit()

print("Video loaded successfully!")


# ------------------------------------------------------------
# PART 2: READ FIRST FRAME
# ------------------------------------------------------------

ret, first_frame = cap.read()

if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Convert first frame to grayscale
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


# ============================================================
# PART 3: LUCAS-KANADE SPARSE OPTICAL FLOW
# ============================================================

print("\nStarting Lucas-Kanade Sparse Optical Flow...")
print("Press ESC to stop.")


# Detect good feature points
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

# Mask for drawing motion trajectories
mask = np.zeros_like(first_frame)

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    if p1 is not None:

        # Select successfully tracked points
        good_new = p1[status == 1]
        good_old = p0[status == 1]

        # Draw motion trajectories
        for new, old in zip(good_new, good_old):

            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            # Draw line showing movement
            mask = cv2.line(
                mask,
                (int(x_new), int(y_new)),
                (int(x_old), int(y_old)),
                (255, 255, 255),
                2
            )

            # Draw current feature point
            frame = cv2.circle(
                frame,
                (int(x_new), int(y_new)),
                5,
                (0, 0, 0),
                -1
            )

        # Combine frame and trajectories
        output = cv2.add(frame, mask)

        cv2.imshow(
            "Lucas-Kanade Sparse Optical Flow",
            output
        )

        # Update previous frame
        old_gray = frame_gray.copy()

        # Update feature points
        p0 = good_new.reshape(-1, 1, 2)

    else:
        print("No feature points found.")
        break

    # Press ESC to exit
    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break


# Close Lucas-Kanade window
cv2.destroyAllWindows()


# ============================================================
# PART 4: RESET VIDEO FOR FARNEBACK
# ============================================================

print("\nStarting Farneback Dense Optical Flow...")
print("Press ESC to stop.")

# Reopen video
cap.release()
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Could not reopen video.")
    exit()


# Read first frame
ret, frame1 = cap.read()

if not ret:
    print("Error: Could not read video.")
    cap.release()
    exit()


# Convert first frame to grayscale
prev_gray = cv2.cvtColor(
    frame1,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# PART 5: FARNEBACK DENSE OPTICAL FLOW
# ============================================================

while True:

    # Read next frame
    ret, frame2 = cap.read()

    if not ret:
        break

    # Convert next frame to grayscale
    next_gray = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        0.5,       # pyramid scale
        3,         # number of pyramid levels
        15,        # window size
        3,         # iterations
        5,         # polynomial neighborhood
        1.2,       # polynomial sigma
        0          # flags
    )

    # --------------------------------------------------------
    # Convert optical flow from Cartesian to Polar coordinates
    # --------------------------------------------------------

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # --------------------------------------------------------
    # Create HSV image for visualization
    # --------------------------------------------------------

    hsv = np.zeros_like(frame2)

    # Direction of motion
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Full saturation
    hsv[..., 1] = 255

    # Motion magnitude
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert HSV to BGR
    optical_flow = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    # Display dense optical flow
    cv2.imshow(
        "Farneback Dense Optical Flow",
        optical_flow
    )

    # Update previous frame
    prev_gray = next_gray

    # Press ESC to exit
    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break


# ============================================================
# PART 6: RELEASE RESOURCES
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nExperiment 7 completed successfully!")