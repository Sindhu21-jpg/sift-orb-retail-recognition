import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import tempfile

st.title("🛒 SIFT Product Matching using FLANN")

run_webcam = st.checkbox(
    "Enable Webcam Recognition"
)

img1_file = st.file_uploader(
    "Upload Reference Image",
    type=["jpg", "png", "jpeg"]
)

img2_file = st.file_uploader(
    "Upload Test Image",
    type=["jpg", "png", "jpeg"]
)

if img1_file is not None and img2_file is not None:

    start_time = time.time()

    file_bytes1 = np.asarray(
        bytearray(img1_file.read()),
        dtype=np.uint8
    )

    file_bytes2 = np.asarray(
        bytearray(img2_file.read()),
        dtype=np.uint8
    )

    img1 = cv2.imdecode(
        file_bytes1,
        cv2.IMREAD_GRAYSCALE
    )

    img2 = cv2.imdecode(
        file_bytes2,
        cv2.IMREAD_GRAYSCALE
    )

    # =========================
    # SIFT + ORB
    # =========================

    sift = cv2.SIFT_create()

    orb = cv2.ORB_create()

    kp1, des1 = sift.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        img2,
        None
    )

    orb_kp1, orb_des1 = orb.detectAndCompute(
        img1,
        None
    )

    orb_kp2, orb_des2 = orb.detectAndCompute(
        img2,
        None
    )

    # =========================
    # FLANN MATCHER
    # =========================

    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = flann.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # =========================
    # DRAW MATCHES
    # =========================

    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches[:50],
        None,
        flags=2
    )

    st.image(
        result,
        caption="FLANN Matching Result"
    )

    # =========================
    # ORB KEYPOINTS
    # =========================

    orb_result = cv2.drawKeypoints(
        img1,
        orb_kp1,
        None,
        color=(0, 255, 0)
    )

    st.image(
        orb_result,
        caption="ORB Keypoints"
    )

    st.write(
        "ORB Keypoints:",
        len(orb_kp1)
    )

    # =========================
    # ACCURACY
    # =========================

    accuracy = (
        len(good_matches) / len(matches)
    ) * 100

    st.write(
        "Total Matches:",
        len(matches)
    )

    st.write(
        "Good Matches:",
        len(good_matches)
    )

    st.write(
        f"Matching Accuracy: {accuracy:.2f}%"
    )

    st.progress(
        int(accuracy)
    )

    # =========================
    # STATUS
    # =========================

    if accuracy > 70:

        st.success(
            "✅ Excellent Product Match"
        )

    elif accuracy > 40:

        st.warning(
            "⚠ Moderate Product Match"
        )

    else:

        st.error(
            "❌ Poor Product Match"
        )

    # =========================
    # GRAPH
    # =========================

    methods = [
        "SIFT-FLANN",
        "ORB"
    ]

    accuracies = [
        accuracy,
        len(orb_kp1) / 10
    ]

    fig, ax = plt.subplots()

    ax.bar(
        methods,
        accuracies
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "Feature Matching Comparison"
    )

    st.pyplot(fig)

    # =========================
    # OBJECT LOCALIZATION
    # =========================

    MIN_MATCH_COUNT = 10

    if len(good_matches) > MIN_MATCH_COUNT:

        src_pts = np.float32([
            kp1[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp2[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        h, w = img1.shape

        pts = np.float32([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(
            pts,
            M
        )

        img2_color = cv2.cvtColor(
            img2,
            cv2.COLOR_GRAY2BGR
        )

        localized = cv2.polylines(
            img2_color,
            [np.int32(dst)],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        st.image(
            localized,
            caption="Detected Object Localization"
        )

    # =========================
    # PROCESSING TIME
    # =========================

    end_time = time.time()

    processing_time = (
        end_time - start_time
    )

    st.write(
        f"Processing Time: {processing_time:.2f} seconds"
    )

# =========================
# WEBCAM
# =========================

if run_webcam:

    camera = cv2.VideoCapture(0)

    frame_window = st.image([])

    st.write(
        "Press STOP in terminal to close webcam"
    )

    while camera.isOpened():

        success, frame = camera.read()

        if not success:

            st.error(
                "Failed to access webcam"
            )

            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        sift = cv2.SIFT_create()

        webcam_kp, webcam_des = (
            sift.detectAndCompute(
                gray,
                None
            )
        )

        webcam_result = cv2.drawKeypoints(
            gray,
            webcam_kp,
            None
        )

        frame_window.image(
            webcam_result,
            channels="GRAY"
        )