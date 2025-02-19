# 📁 **تصنيف الفيديو باستخدام YOLO**

---

## 🌟 **نبذة عن المشروع**
هذا المشروع هو تطبيق بسيط يستخدم مكتبة **YOLO (You Only Look Once)** لتصنيف الفيديوهات في الزمن الحقيقي. يتم استخدام نموذج التصنيف المُدرّب مسبقًا `yolo11n-cls.pt` لتحديد الفئة الرئيسية لكل إطار من الفيديو، ويتم عرض النتائج مباشرة على الشاشة مع إضافة تسميات تصنيفية.

---

## 🛠️ **المتطلبات الأساسية**
قبل تشغيل المشروع، تأكد من توفر المتطلبات التالية:

### 1. **المكتبات المطلوبة**
- Python 3.x
- OpenCV (`cv2`)
- Ultralytics YOLO

```bash
pip install opencv-python ultralytics
```

### 2. **الملفات المطلوبة**
- ملف الفيديو: `small.mp4` (يجب أن يكون موجودًا في نفس المسار أو تحديد مساره الصحيح).
- نموذج التصنيف: `yolo11n-cls.pt` (يتم تحميله تلقائيًا إذا لم يكن موجودًا).

---

## 🚀 **كيف يعمل المشروع؟**
### 1. **تحميل النموذج**
```python
model = YOLO("yolo11n-cls.pt")
```
- يتم تحميل نموذج التصنيف المُدرّب مسبقًا باستخدام مكتبة **Ultralytics**.
- النموذج قادر على تصنيف الصور إلى فئات متعددة بناءً على البيانات التي تم تدريبه عليها.

### 2. **فتح الفيديو**
```python
cap = cv2.VideoCapture(video_path)
```
- يتم فتح ملف الفيديو باستخدام مكتبة **OpenCV**.
- يمكن تعديل مسار الفيديو عن طريق تغيير قيمة `video_path`.

### 3. **إعداد نافذة العرض**
```python
cv2.namedWindow("Classification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Classification", window_width, window_height)
```
- يتم إنشاء نافذة للعرض بأبعاد قابلة للتخصيص (`window_width` و `window_height`).

### 4. **معالجة الإطارات**
- يتم قراءة كل إطار من الفيديو باستخدام حلقة `while`.
- يتم إعادة تحجيم الإطار ليتناسب مع حجم النافذة.
- يتم تنفيذ التصنيف باستخدام النموذج:
  ```python
  results = model(frame)
  ```
- يتم استخراج أفضل تصنيف وإضافة تسمية تحتوي على اسم الفئة ودرجة الثقة:
  ```python
  label = f"{model.names[class_id]} ({result.probs.top1conf:.2f})"
  ```

### 5. **عرض النتائج**
- يتم عرض الإطار المُصنَّف مع التسمية باستخدام:
  ```python
  cv2.imshow("Classification", frame)
  ```
- يمكن الخروج من البرنامج بالضغط على زر `'q'`.

---

## 🔧 **كيفية التشغيل**
1. قم بتنزيل المشروع وتأكد من وجود جميع الملفات المطلوبة.
2. شغل البرنامج باستخدام الأمر التالي:
   ```bash
   python script_name.py
   ```
3. سيتم فتح نافذة تعرض الفيديو مع التصنيف المباشر لكل إطار.
4. اضغط على زر `'q'` للخروج من البرنامج.

---

## 🎨 **التخصيص**
### 1. **تعديل حجم النافذة**
يمكنك تعديل حجم نافذة العرض عن طريق تغيير القيم التالية:
```python
window_width, window_height = 800, 600
```

### 2. **استخدام فيديو مختلف**
قم بتغيير مسار الفيديو:
```python
video_path = "your_video.mp4"
```

### 3. **نموذج تصنيف مختلف**
إذا كنت تريد استخدام نموذج تصنيف آخر، قم بتغيير اسم الملف:
```python
model = YOLO("your_model.pt")
```

---

## 📸 **عينة من النتائج**
عند تشغيل البرنامج، ستظهر نافذة تعرض الفيديو مع تسميات مثل:
- **Cat (0.95)**: يشير إلى أن الفئة هي "قطة" مع درجة ثقة 95%.
- **Dog (0.87)**: يشير إلى أن الفئة هي "كلب" مع درجة ثقة 87%.

---

## 📝 **ملاحظات**
- تأكد من أن ملف الفيديو والنموذج في المسار الصحيح.
- إذا كنت تستخدم كاميرا ويب بدلاً من ملف فيديو، يمكنك تغيير الكود ليصبح:
  ```python
  cap = cv2.VideoCapture(0)
  ```
- قد تحتاج إلى تعديل حجم النافذة بناءً على دقة الفيديو الخاص بك.

---

## 🙏 **شكرًا لك!**
نأمل أن تستمتع باستخدام هذا المشروع البسيط والمفيد! إذا كان لديك أي أسئلة أو اقتراحات، فلا تتردد في التواصل معنا. ✨

--- 

🌟 **Happy Coding!** 🌟
