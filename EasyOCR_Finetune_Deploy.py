import cv2
import numpy as np
import time
import mmap
import struct
import easyocr
import os
import base64
import socketio
import re
import requests
from difflib import get_close_matches
import subprocess

# ✅ รายชื่อจังหวัดไทย
THAI_PROVINCES = [
    "กรุงเทพมหานคร", "เชียงใหม่", "ชลบุรี", "ขอนแก่น", "ภูเก็ต", "นนทบุรี", "ปทุมธานี", 
    "สมุทรปราการ", "นครราชสีมา", "นครปฐม", "นครศรีธรรมราช", "สงขลา", "สุราษฎร์ธานี", "ระยอง",
    "ราชบุรี", "อุบลราชธานี", "อุดรธานี", "สุพรรณบุรี", "สระบุรี", "ลำปาง", "สกลนคร", "เพชรบุรี",
    "อยุธยา", "ลพบุรี", "บุรีรัมย์"
]



TOKEN = "7810240233:AAHanfpFVW7fblvVGMN1ixdYyPyAwM6JERQ"
#CHAT_ID = "7898013745"
CHAT_ID = "-4790527287"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

SHM_NAME = "/dev/shm/CroppedImageSHM"
SHM_SIZE = 1024 * 1024

#SOCKET_SERVER_URL = "ws://192.168.43.178:8765"
SOCKET_SERVER_URL = "https://rec.licenseplate.pro/socket.io/"

# ✅ สร้าง Socket.IO Client
sio = socketio.Client(reconnection=True, reconnection_attempts=10, reconnection_delay=5)

# ✅ กำหนด State Machine
STATE_WIFI = "WIFI"
STATE_SOCKET = "SOCKET"
STATE_RUNNING = "RUNNING"

current_state = STATE_WIFI


def get_jetson_temperature():
    temp_path = "/sys/devices/virtual/thermal/thermal_zone0/temp"
    try:
        with open(temp_path, "r") as file:
            temp_milli_celsius = int(file.read().strip()) 
            temp_celsius = temp_milli_celsius / 1000.0
            return temp_celsius
    except Exception as e:
        return f"Error reading temperature: {e}"

def get_wifi_ips():
    try:
        result = subprocess.check_output("ip addr show", shell=True).decode()
        interfaces = re.findall(r"(\d+):\s+(wlan\d*|wlp\d*):", result)
        wifi_ips = {}
        for _, interface in interfaces:
            match = re.search(rf"{interface}.*?\n\s*inet\s+(\d+\.\d+\.\d+\.\d+)", result, re.S)
            if match:
                wifi_ips[interface] = match.group(1)
        return wifi_ips if wifi_ips else "No WiFi connection found"
    except Exception as e:
        return f"Error: {e}"



def rotate_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # ใช้ Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    angles = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90  # แปลงเป็นองศา
            angles.append(angle)

        # ✅ คำนวณมุมหมุน
        rotation_angle = np.median(angles)

        # ✅ หมุนภาพ
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated
    else:
        print("❌ ไม่พบเส้นขอบในภาพ")
        return image  # คืนค่าเดิมถ้าไม่พบเส้น

def set_fan_speed(speed):
    """
    กำหนดความเร็วพัดลมของ Jetson Nano
    speed: ค่า PWM (0-255)
    """
    speed = max(0, min(255, speed))  # จำกัดค่าให้อยู่ในช่วง 0 - 255
    cmd = f"echo {speed} | sudo tee /sys/devices/pwm-fan/target_pwm"
    os.system(cmd)
    print(f"Fan speed set to {speed}/255")


def is_wifi_connected():
    """✅ ตรวจสอบว่า WiFi เชื่อมต่อหรือไม่"""
    try:
        result = subprocess.run(["ping", "-c", "1", "8.8.8.8"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except Exception:
        return False


def connect_websocket():
    """✅ พยายามเชื่อมต่อ WebSocket"""
    global sio
    try:
        if sio.connected:
            print("✅ WebSocket already connected")
            return True
        print("⚙️ Connecting to WebSocket...")
        sio.disconnect()  # ปิดการเชื่อมต่อล่าสุดถ้ามีปัญหา
        sio.connect(SOCKET_SERVER_URL, transports=["websocket", "polling"], wait_timeout=5)
        data = {"chat_id": CHAT_ID, "text": f"Connected to WebSocket Server:{SOCKET_SERVER_URL}"}
        requests.post(url, data=data)
        print("✅ WebSocket Connected Success")
        return True
    except Exception as e:
        data = {"chat_id": CHAT_ID, "text": "WebSocket Connection Failed"}
        requests.post(url, data=data)
        print("❌ WebSocket Connection Failed:", e)
        return False


def handle_websocket_data(data):
    """✅ จัดการข้อมูลที่รับมาจาก WebSocket"""
    print("📩 Received data from WebSocket:", data)

    if isinstance(data, dict):
        command = data.get("command", "")
        if command == "restart":
            print("🔄 Restarting system...")
            os.system("sudo reboot")

        elif command == "shutdown":
            print("⚠️ Shutting down system...")
            os.system("sudo shutdown -h now")

        elif command == "status":
            print("✅ Sending status update...")
            sio.emit("status_response", {"status": "running", "wifi": is_wifi_connected()})

        elif command == "custom_message":
            message = data.get("message", "No message received")
            print("📢 Custom Message:", message)
            requests.post(url, data={"chat_id": CHAT_ID, "text": f"Message from server: {message}"})

    elif isinstance(data, str):
        print("📩 Received text:", data)


# ✅ ฟัง WebSocket Event
@sio.on("message")
def on_message(data):
    """✅ รับข้อความจาก WebSocket"""
    handle_websocket_data(data)


@sio.on("command")
def on_command(data):
    """✅ รับคำสั่งจาก WebSocket"""
    handle_websocket_data(data)


def create_shm():
    """✅ สร้าง Shared Memory ถ้ายังไม่มี"""
    if not os.path.exists(SHM_NAME):
        print("⚙️ Creating Shared Memory...")
        with open(SHM_NAME, "wb") as f:
            f.write(b'\x00' * SHM_SIZE)
        print("✅ Shared Memory Created!")

def correct_province(province_text):
    """ ✅ แก้ไขชื่อจังหวัดที่สะกดผิด """
    province_text = province_text.strip()
    matches = get_close_matches(province_text, THAI_PROVINCES, n=1, cutoff=0.5)  # ลด cutoff เป็น 0.5 ให้ยืดหยุ่นขึ้น
    return matches[0] if matches else "Unknown"

# def extract_license_plate_info(ocr_text):
#     """ ✅ แยกเลขทะเบียนและจังหวัดจากข้อความ OCR พร้อมแก้ไขจังหวัดที่ผิด """
#     words = ocr_text.replace(" ", "")  # ✅ ลบเว้นวรรคทั้งหมดออก
#     detected_province = None
    
#     # ✅ ค้นหาชื่อจังหวัดที่ OCR อ่านออกมา
#     for province in THAI_PROVINCES:
#         if province in words:
#             detected_province = province
#             break

#     # ✅ ลองค้นหาจังหวัดโดยใช้แค่ 3-4 ตัวแรกของชื่อจังหวัด
#     # if detected_province is None:
#     #     for province in THAI_PROVINCES:
#     #         if province[:4] in words:  # ตรวจหาตัวอักษรขึ้นต้น 4 ตัวแรก
#     #             detected_province = correct_province(province)
#     #             break

#     # ✅ ใช้ fuzzy matching ตรวจสอบและแก้ไขชื่อจังหวัดที่ผิดพลาด
#     if detected_province is None:
#         detected_province = correct_province(words)

#     # ✅ ตรวจจับหมายเลขทะเบียนรถยนต์
#     pattern = r"([0-9]{1,2}[ก-ฮ]{1,2}[0-9]{1,4})|([ก-ฮ]{1,2}[0-9]{1,4})"
#     match = re.search(pattern, words)
#     license_plate = match.group(0) if match else "Unknown"

#     if detected_province and license_plate and detected_province in license_plate:
#         print("❌ ตรวจพบจังหวัดไม่ถูกต้อง! ตั้งค่าเป็น Unknown")
#         detected_province = "Unknown"

#     return license_plate, detected_province

# def extract_license_plate_info(ocr_text, text_results):
#     """ ✅ แยกเลขทะเบียนและจังหวัดจากข้อความ OCR ให้ถูกต้อง """
    
#     words = ocr_text.replace(" ", "")  # ✅ ลบเว้นวรรคทั้งหมดออก
#     detected_province = None
#     license_plate = None

#     # ✅ ตรวจหาหมายเลขทะเบียนที่ถูกต้อง
#     pattern = r"[0-9]{1,2}[ก-ฮ][0-9]{1,4}|[ก-ฮ]{1,2}[0-9]{1,4}"  
#     matches = re.findall(pattern, words)
    
#     if matches:
#         license_plate = matches[0]  # ✅ เอาทะเบียนที่พบตัวแรก

#     # ✅ ค้นหาชื่อจังหวัดจากข้อความ OCR
#     for province in THAI_PROVINCES:
#         if province in words:
#             detected_province = province
#             break

#     if detected_province is None:
#         detected_province = correct_province(words)

#     # ✅ ตรวจสอบว่าจังหวัดอยู่ก่อนทะเบียนหรือไม่
#     if detected_province and license_plate:
#         province_index = words.find(detected_province)
#         plate_index = words.find(license_plate)

#         if province_index < plate_index:
#             print("⚠️ พบจังหวัดอยู่ก่อนทะเบียน! สลับตำแหน่งให้ถูกต้อง")
#             license_plate, detected_province = detected_province, license_plate

#     # ✅ ตรวจสอบจังหวัดไม่ควรมีตัวเลข
#     if detected_province and any(char.isdigit() for char in detected_province):
#         print("❌ พบจังหวัดที่มีตัวเลขอยู่ในชื่อ! รีเซ็ตเป็น Unknown")
#         detected_province = "Unknown"

#     return license_plate or "Unknown", detected_province or "Unknown"

# def extract_license_plate_info(ocr_text):
#     """ ✅ แยกทะเบียนและจังหวัดจากข้อความ OCR ให้ถูกต้อง """
    
#     words = ocr_text.replace(" ", "")  # ✅ ลบช่องว่างทั้งหมด
#     detected_province = None
#     license_plate = None

#     # ✅ ค้นหาชื่อจังหวัดจากข้อความ OCR
#     for province in THAI_PROVINCES:
#         if province in words:
#             detected_province = province
#             break

#     if detected_province is None:
#         detected_province = correct_province(words)


#     # ✅ ตรวจหาทะเบียนที่ถูกต้อง (ตัวอักษรไทย + ตัวเลข)
#     pattern = r"[0-9]{1,2}[ก-ฮ][0-9]{1,4}|[ก-ฮ]{1,2}[0-9]{1,4}"  
#     matches = re.findall(pattern, words)
    
#     if matches:
#         license_plate = matches[0]  # ✅ เอาทะเบียนที่พบตัวแรก

#     # ✅ ถ้าทะเบียนไม่มีอักษรไทย ให้ตั้งเป็น "Unknown"
#     if license_plate and not re.search(r"[ก-ฮ]", license_plate):
#         print("❌ พบทะเบียนที่ไม่มีตัวอักษรไทย! รีเซ็ตเป็น Unknown")
#         license_plate = "Unknown"

#     if detected_province and license_plate and detected_province in license_plate:
#         print("❌ ตรวจพบจังหวัดไม่ถูกต้อง! ตั้งค่าเป็น Unknown")
#         detected_province = "Unknown"

    

#     return license_plate or "Unknown", detected_province or "Unknown"

def extract_license_plate_info(ocr_text):
    """ ✅ แยกทะเบียนและจังหวัดจากข้อความ OCR ให้ถูกต้อง """

    words_list = ocr_text.split()  # ✅ แยกข้อความเป็นรายการ (list)

    # ✅ ตรวจสอบว่าตำแหน่งแรกเป็นตัวเลข หรือ ตัวเลข+อักษรไทย (เช่น 6996 หรือ 2กท)
    if len(words_list) >= 2 and (re.match(r"^\d+[ก-ฮ]?$", words_list[0]) and words_list[1].isdigit()):
        print(f"⚠️ พบหมายเลขอยู่ก่อนตัวอักษร! สลับตำแหน่ง {words_list[0]} <-> {words_list[1]}")
        words_list[0], words_list[1] = words_list[1], words_list[0]  # ✅ สลับตำแหน่ง
    elif len(words_list) >= 2 and words_list[0].isdigit() and re.search(r"[ก-ฮ]", words_list[1]):
        print(f"⚠️ พบหมายเลขอยู่ก่อนตัวอักษร! สลับตำแหน่ง {words_list[0]} <-> {words_list[1]}")
        words_list[0], words_list[1] = words_list[1], words_list[0]  # ✅ สลับตำแหน่ง

    words = "".join(words_list)  # ✅ รวมข้อความกลับ
    
    detected_province = None
    license_plate = None

    # ✅ ค้นหาชื่อจังหวัดจากข้อความ OCR
    for province in THAI_PROVINCES:
        if province in words:
            detected_province = province
            break

    if detected_province is None:
        detected_province = correct_province(words)

    # ✅ ตรวจหาทะเบียนที่ถูกต้อง (ตัวอักษรไทย + ตัวเลข)
    # pattern = r"[0-9]{1,2}[ก-ฮ][0-9]{1,4}|[ก-ฮ]{1,2}[0-9]{1,4}" 
    pattern = r"[0-9]{1,2}[ก-ฮ]{1,2}[0-9]{1,4}|[ก-ฮ]{1,2}[0-9]{1,4}" 
    matches = re.findall(pattern, words)
    
    if matches:
        license_plate = matches[0]  # ✅ เอาทะเบียนที่พบตัวแรก

    # ✅ ถ้าจังหวัดอยู่ก่อนทะเบียน ต้องสลับลำดับ
    # if detected_province and license_plate:
    #     province_index = words.find(detected_province)
    #     plate_index = words.find(license_plate)

    #     if province_index < plate_index:
    #         print("⚠️ พบจังหวัดอยู่ก่อนทะเบียน! สลับตำแหน่งให้ถูกต้อง")
    #         license_plate, detected_province = detected_province, license_plate

    # ✅ ถ้าทะเบียนไม่มีอักษรไทย ให้ตั้งเป็น "Unknown"
    if license_plate and not re.search(r"[ก-ฮ]", license_plate):
        print("❌ พบทะเบียนที่ไม่มีตัวอักษรไทย! รีเซ็ตเป็น Unknown")
        license_plate = "Unknown"

    return license_plate or "Unknown", detected_province or "Unknown"


def read_from_shm():
    fan_speed = 125
    set_fan_speed(fan_speed)  # ปรับพัดลมให้ทำงานที่ 100%
    """✅ อ่าน Shared Memory และส่งไปยัง WebSocket"""
    create_shm()
    last_timestamp = 0
    count_check_temp = 0
    reader = easyocr.Reader(['th'], recog_network='first', gpu=True)
    print("✅ OCR Ready!")
    data = {"chat_id": CHAT_ID, "text": f"OCR Ready! fanspeed: {fan_speed} "}
    requests.post(url, data=data)

    while True:
        with open(SHM_NAME, "r+b") as f:
            shm = mmap.mmap(f.fileno(), SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            while True:
                global current_state
                if not is_wifi_connected():
                    print("❌ WiFi Disconnected! Reconnecting...")
                    current_state = STATE_WIFI
                    return
                if not sio.connected:
                    print("❌ WebSocket Disconnected! Reconnecting...")
                    current_state = STATE_SOCKET
                    return

                timestamp_bytes = shm[:8]
                timestamp = struct.unpack("<Q", timestamp_bytes)[0]
                img_data = shm[8:]
                if timestamp > last_timestamp:
                    print(f"timestamp:{timestamp},lasttimestamp:{last_timestamp}")
                    last_timestamp = timestamp
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if image is not None and image.size > 0:
                        print(f"✅ Image received at {timestamp}")
                        rotated_image = rotate_plate(image)
                        gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)  # ✅ แปลงเป็น Grayscale


                        # results = reader.readtext(gray, detail=1)  # ✅ ให้ OCR คืนค่าตำแหน่ง (bounding box)
                        # text_results = []
                        # for (bbox, text, prob) in results:
                        #     x_min = bbox[0][0]  # ✅ ตำแหน่ง x ของข้อความ
                        #     text_results.append((x_min, text))

                        # # ✅ เรียงข้อความจากซ้ายไปขวา
                        # text_results.sort()
                        # sorted_texts = [text for (_, text) in text_results]

                        # ocr_text = " ".join(sorted_texts)
                        # print("📄 OCR Ordered Result:", ocr_text)

                        # # ✅ แยกทะเบียนและจังหวัด
                        # license_plate, province = extract_license_plate_info(ocr_text)
                        # if not license_plate:
                        #     print("❌ ไม่สามารถตรวจจับหมายเลขทะเบียนได้")
                        #     license_plate = "Unknown"
                        # if not province:
                        #     print("❌ ไม่สามารถตรวจจับจังหวัดได้")
                        #     province = "Unknown"
                        # print(f"🚗 ทะเบียน: {license_plate} | 📍 จังหวัด: {province}")

                        # results = reader.readtext(gray, detail=1)
                        # text_results = []

                        # for (bbox, text, prob) in results:
                        #     (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]  # ✅ ดึงตำแหน่ง x, y
                        #     text_results.append((y_min, x_min, text))  # ✅ เก็บตำแหน่ง Y ก่อน X

                        # # ✅ จัดกลุ่มข้อความตามแนวตั้ง แล้วเรียงแนวนอน
                        # text_results.sort()  # เรียงจากบนลงล่างก่อน แล้วค่อยเรียงจากซ้ายไปขวาในแต่ละแถว

                        # # ✅ รวมข้อความใหม่หลังการเรียง
                        # sorted_texts = [text for (_, _, text) in text_results]
                        # ocr_text = " ".join(sorted_texts)
                        # print("📄 OCR Ordered Result:", ocr_text)

                        # # ✅ แยกทะเบียนและจังหวัด
                        # license_plate, province = extract_license_plate_info(ocr_text, text_results)

                        # if not license_plate:
                        #     print("❌ ไม่สามารถตรวจจับหมายเลขทะเบียนได้")
                        #     license_plate = "Unknown"
                        # if not province:
                        #     print("❌ ไม่สามารถตรวจจับจังหวัดได้")
                        #     province = "Unknown"

                        # print(f"🚗 ทะเบียน: {license_plate} | 📍 จังหวัด: {province}")

                        # results = reader.readtext(gray, detail=1)
                        # text_results = []

                        # for (bbox, text, prob) in results:
                        #     (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]  # ✅ ดึงตำแหน่ง x, y
                        #     text_results.append((y_min, x_min, text))  # ✅ เก็บตำแหน่ง Y ก่อน X

                        # # ✅ เรียงข้อความจากบนลงล่างก่อน แล้วซ้ายไปขวา
                        # text_results.sort()

                        # # ✅ รวมข้อความใหม่หลังการเรียง
                        # sorted_texts = [text for (_, _, text) in text_results]
                        # ocr_text = " ".join(sorted_texts)
                        # print("📄 OCR Ordered Result:", ocr_text)

                        # # ✅ แยกทะเบียนและจังหวัด
                        # license_plate, province = extract_license_plate_info(ocr_text)

                        # if not license_plate:
                        #     print("❌ ไม่สามารถตรวจจับหมายเลขทะเบียนได้")
                        #     license_plate = "Unknown"
                        # if not province:
                        #     print("❌ ไม่สามารถตรวจจับจังหวัดได้")
                        #     province = "Unknown"

                        # print(f"🚗 ทะเบียน: {license_plate} | 📍 จังหวัด: {province}")

                        results = reader.readtext(gray, detail=1)
                        text_results = []

                        for (bbox, text, prob) in results:
                            (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]  # ✅ ดึงตำแหน่ง x, y
                            text_results.append((y_min, x_min, text))  # ✅ จัดเก็บตำแหน่ง Y ก่อน X

                        # ✅ เรียงข้อความจากบนลงล่างก่อน แล้วซ้ายไปขวา
                        text_results.sort()

                        # ✅ รวมข้อความใหม่หลังการเรียง
                        sorted_texts = [text for (_, _, text) in text_results]
                        ocr_text = " ".join(sorted_texts)
                        print("📄 OCR Ordered Result:", ocr_text)

                        # ✅ แยกทะเบียนและจังหวัด
                        license_plate, province = extract_license_plate_info(ocr_text)

                        if not license_plate:
                            print("❌ ไม่สามารถตรวจจับหมายเลขทะเบียนได้")
                            license_plate = "Unknown"
                        if not province:
                            print("❌ ไม่สามารถตรวจจับจังหวัดได้")
                            province = "Unknown"

                        print(f"🚗 ทะเบียน: {license_plate} | 📍 จังหวัด: {province}")

                        
                        _, buffer = cv2.imencode('.jpg', image)
                        image_base64 = base64.b64encode(buffer).decode('utf-8')
                        print("📡 Sending data to server...")
                        sio.emit("sendData", {"license_plate": license_plate,"provice_plate": province, "image": image_base64,"camera_id":1})
                        data = {
                             "chat_id": CHAT_ID,
                             "text": f"liceseplate:{license_plate} | provice_plate: {province} | camera_id: {1}"
                        }
                        requests.post(url, data=data)
                        print("✅ Data sent successfully!")
                        shm.seek(0)
                        shm.write(struct.pack("<Q", 0))  # เคลียร์ Timestamp
                        shm.write(b'\x00' * (SHM_SIZE - 8))  # เคลียร์ข้อมูลภาพ
                        print("✅ Cleared old image from Shared Memory")
                
                count_check_temp += 1
                #print(count_check_temp)
                if count_check_temp == 60:
                    count_check_temp = 0
                    temperature = get_jetson_temperature()
                    print(temperature)
                    data = {
                             "chat_id": CHAT_ID,
                             "text": f"temperature : {temperature}"
                    }
                    requests.post(url, data=data)

                time.sleep(0.5)


if __name__ == "__main__":
    while True:
        if current_state == STATE_WIFI:
            print("🔍 Checking WiFi...")
            if is_wifi_connected():
                print("✅ WiFi Connected!")
                wifi_ips = get_wifi_ips()
                print("WiFi IP Addresses:", wifi_ips)
                data = {"chat_id": CHAT_ID, "text": f"WiFi : {wifi_ips}"}
                requests.post(url, data=data) 
                current_state = STATE_SOCKET
            else:
                print("❌ WiFi Not Connected! Retrying in 5 seconds...")
                time.sleep(5)

        elif current_state == STATE_SOCKET:
            if connect_websocket():
                current_state = STATE_RUNNING
            else:
                print("❌ WebSocket Not Connected! Retrying in 5 seconds...")
                time.sleep(5)

            if not is_wifi_connected():
                current_state = STATE_WIFI

        elif current_state == STATE_RUNNING:
            print("🚀 Running OCR & Listening WebSocket...")
            read_from_shm()
