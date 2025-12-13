def analyze_eye_movements(video_path: str):
    # 🔹 Lazy imports (VERY IMPORTANT for cloud deploy)
    import cv2
    import mediapipe as mp
    import numpy as np

    mp_face_mesh = mp.solutions.face_mesh

    """
    Returns:
      - fixation_count
      - avg_fixation_duration
      - regressions_count
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback safety

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    last_gaze_x = None
    fixation_frames = 0
    fixation_count = 0
    regressions = 0
    fixation_durations = []

    FIXATION_THRESHOLD = 0.01
    REGRESSION_THRESHOLD = -0.01

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            continue

        face = results.multi_face_landmarks[0]

        # Left eye landmarks (MediaPipe)
        left_eye = np.mean(
            [(lm.x, lm.y) for lm in face.landmark[33:133]],
            axis=0
        )

        gaze_x = left_eye[0]

        if last_gaze_x is not None:
            movement = gaze_x - last_gaze_x

            if abs(movement) < FIXATION_THRESHOLD:
                fixation_frames += 1
            else:
                if fixation_frames > 0:
                    fixation_count += 1
                    fixation_durations.append(fixation_frames / fps)
                fixation_frames = 0

            if movement < REGRESSION_THRESHOLD:
                regressions += 1

        last_gaze_x = gaze_x

    cap.release()
    face_mesh.close()

    avg_fixation = (
        sum(fixation_durations) / len(fixation_durations)
        if fixation_durations else 0
    )

    return {
        "fixation_count": fixation_count,
        "avg_fixation_duration": round(avg_fixation, 3),
        "regressions_count": regressions
    }
