import cv2
from ultralytics import YOLO

# تحميل نموذج التصنيف
model = YOLO("yolo11n-cls.pt")

# فتح الفيديو
video_path = "small.mp4"
cap = cv2.VideoCapture(video_path)

# تحديد حجم النافذة
window_width, window_height = 800, 600  # يمكنك التعديل حسب رغبتك
cv2.namedWindow("Classification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Classification", window_width, window_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # إعادة تحجيم الفيديو ليتناسب مع النافذة
    frame = cv2.resize(frame, (window_width, window_height))

    # تنفيذ التصنيف
    results = model(frame)

    # استخراج أفضل تصنيف
    for result in results:
        class_id = int(result.probs.top1)
        label = f"{model.names[class_id]} ({result.probs.top1conf:.2f})"

        # إضافة النص إلى الصورة
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # عرض الإطار
    cv2.imshow("Classification", frame)

    # للخروج من الفيديو عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إغلاق الفيديو
cap.release()
cv2.destroyAllWindows()
