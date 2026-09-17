import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        print("Video not found or video ended.")
        break

    # Detect objects
    results = model(frame, verbose=False)

    vehicle_count = 0

    # Count vehicles
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            # Vehicle classes:
            # 2 = car
            # 3 = motorcycle
            # 5 = bus
            # 7 = truck
            if class_id in [2, 3, 5, 7]:
                vehicle_count += 1

    # Decide traffic level and signal timing
    if vehicle_count > 20:
        traffic_status = "HIGH TRAFFIC"
        signal_time = 60

    elif vehicle_count > 10:
        traffic_status = "MEDIUM TRAFFIC"
        signal_time = 40

    else:
        traffic_status = "LOW TRAFFIC"
        signal_time = 20

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Display information
    cv2.putText(
        annotated_frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Traffic: {traffic_status}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Green Signal: {signal_time} sec",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Show output
    cv2.imshow(
        "AI Smart Traffic Management System",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()