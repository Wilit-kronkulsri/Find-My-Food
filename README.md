# 🍽️ Find My Food (Thai Food & Object Image Classification)

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.13.0-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io)
[![Gemini](https://img.shields.io/badge/Google_GenAI-Gemini_2.5-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)

**Find My Food** เป็นระบบปัญญาประดิษฐ์เพื่อการจำแนกและระบุประเภทอาหารไทยรวมถึงวัตถุต่าง ๆ ผ่านภาพถ่าย พัฒนาขึ้นเพื่อช่วยแยกแยะประเภทอาหารไทยยอดนิยมและตรวจสอบการจำแนกวัตถุเบื้องต้น โดยโปรเจกต์นี้ได้รับ **"รางวัลชมเชย"** จากโครงการประกวดผลงาน **"ผู้กล้าท้า AI" (KU AI Pioneers: Forward Challenge)** มหาวิทยาลัยเกษตรศาสตร์ เมื่อวันที่ 15 มิถุนายน 2026

---

## 👨‍💻 My Role & Responsibilities (ตำแหน่งและความรับผิดชอบ)

ในโปรเจกต์นี้ ผมรับหน้าที่เป็น **AI / Machine Learning Engineer** โดยรับผิดชอบระบบประมวลผลและการเทรนโมเดล Deep Learning ทั้งหมดในส่วนของ Core AI:

1.  **Model Architecture Design & Implementation:** ออกแบบและพัฒนาโมเดลจำแนกรูปภาพโดยใช้เทคนิค Transfer Learning จากสถาปัตยกรรม **MobileNetV2** บนโครงสร้าง Sequential ของ Keras โดยมีการทำ Preprocessing แบบปรับแต่งสีด้วยเลเยอร์ Rescaling และทำฐานข้อมูลให้เสถียรผ่านการประยุกต์ใช้ Batch Normalization และ Dropout เพื่อลดปัญหา Overfitting
2.  **Data Augmentation Optimization:** ออกแบบขั้นตอนการสุ่มปรับแต่งรูปภาพ (Data Augmentation) เช่น RandomFlip, RandomRotation, RandomZoom และ RandomContrast เพื่อให้โมเดลมีความทนทานและจดจำอาหารในสภาพแสงและมุมมองที่แตกต่างกันได้ดีขึ้น
3.  **Two-Stage Model Training (Fine-Tuning):** 
    *   *Stage 1:* ล็อค Layer ส่วนล่างของ MobileNetV2 และเทรนส่วนล่าง (Top Classifier Layer) ด้วย Learning Rate ปกติ
    *   *Stage 2:* ทำการปลดล็อค Layer (ตั้งแต่เลเยอร์ 100 ขึ้นไป) เพื่อทำ Fine-Tuning ด้วยอัตราการเรียนรู้ระดับต่ำมาก (Learning Rate = 1e-5) ร่วมกับ Early Stopping เพื่อรีดประสิทธิภาพสูงสุดออกมา
4.  **Generative AI Integration:** ออกแบบสคริปต์เชื่อมต่อและสั่งการโครงข่ายร่วมกับ **Google Gen AI SDK (Gemini API)** เพื่อนำความสามารถของขุมพลังโมเดลภาษาขนาดใหญ่ (LLM) มาวิเคราะห์ส่วนผสมหรือระบุคุณค่าทางโภชนาการเพิ่มเติมต่อจากผลการคัดแยกภาพหลัก

---

## 📊 Dataset & Classes

โมเดลนี้ได้รับการฝึกฝนและทดสอบกับรูปภาพทั้งหมด **11 คลาสหลัก** (รวมมากกว่า 12,000 รูปภาพ) แบ่งออกเป็นประเภทอาหารไทยยอดนิยมและวัตถุทั่วไป ดังนี้:

*   🍛 Fried rice (ข้าวผัด)
*   🌶️ Pad Krapow (ผัดกะเพรา)
*   🍜 Pad Thai (ผัดไทย)
*   🦀 Phat Phong Kari (ปู/กุ้งผัดผงกะหรี่)
*   🥢 Phat See Ew (ผัดซีอิ๊ว)
*   🍳 Rice with omelet (ข้าวไข่เจียว)
*   🥗 Somtam (ส้มตำ)
*   🍲 Suki (สุกี้)
*   🧄 Thot Kratiam (ทอดกระเทียม)
*   🍤 Tom Yum (ต้มยำ)
*   📦 object (วัตถุหรือสิ่งของทั่วไปที่ไม่ใช่อาหาร)

---

## 📁 Repository Structure

ภายในคลังข้อมูล (Repository) นี้ประกอบด้วยไฟล์หลักที่ใช้ในการทำงานดังนี้:

*   `TestModel.ipynb`: ไฟล์ Jupyter Notebook หลักที่ใช้ในกระบวนการตั้งแต่การเตรียมข้อมูล, ทำ Data Augmentation, สร้าง Pipeline การเทรน, ตรวจสอบกราฟ Loss/Accuracy ไปจนถึงการทดสอบเรียกใช้โมเดล
*   `MNV2_Project.keras`: ไฟล์น้ำหนักโมเดล (Model Weights) ที่ผ่านการเทรนขั้นสุดท้าย (Fine-Tuning) เรียบร้อยแล้ว พร้อมนำไปใช้งานจัดหมวดหมู่คลาสอาหาร
*   `MNV2_Project_AI.keras`: โมเดลเวอร์ชันปรับปรุงเพิ่มเติมที่ถูกพัฒนาขึ้นเพื่อทำงานสอดประสานระบบร่วมกับคุณสมบัติของปัญญาประดิษฐ์ระดับสูง

---

## 🚀 Performance & Results

โมเดลที่ผ่านการประมวลผลสองขั้นตอน (Transfer Learning + Fine-Tuning) สามารถทำความแม่นยำ (Accuracy) ในรอบการทดสอบสุดท้ายได้สูงถึง **96%+ บนชุดข้อมูลฝึกฝน (Training Set)** และมีเสถียรภาพความแม่นยำสูงเมื่อประเมินบน Validation Set สะท้อนถึงการปรับจูนไฮเปอร์พารามิเตอร์และการทำ Augmentation ที่เหมาะสม
