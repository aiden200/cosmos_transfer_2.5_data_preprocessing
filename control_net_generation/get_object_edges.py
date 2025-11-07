import cv2, os


def generate_edges(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    assert cap.isOpened(), "Could not open input video."
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bright = 50
    contrast = 1.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # use "avc1" if you prefer H.264 and it's available
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)
        edges = cv2.Canny(blurred, 10, 50)
        out.write(edges)
    cap.release()
    out.release()


if __name__ == "__main__":
    in_path = "input_video.mp4"
    out_path = "edges.mp4"
    generate_edges(in_path, out_path)