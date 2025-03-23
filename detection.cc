
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <curl/curl.h>
#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <sys/resource.h> // สำหรับ getrusage
#include <thread>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

// work on camera

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// change these variables accordingly
#define INPUT_SIZE 320
#define MODEL_FILE "/home/jetson/Desktop/quickyolo/QuickYOLO/compute-engine/gen/linux_aarch64/quickyolov2_custom_add_Relu.tflite"
#define LABEL_MAP_FILE "/home/jetson/Desktop/quickyolo/QuickYOLO/compute-engine/gen/linux_aarch64/voc_names_bignew.txt"
#define NUM_THREADS 2

#define NO_PRESS 255

using namespace boost::interprocess;
using namespace tflite;
using namespace std;
using namespace cv;
using namespace std::chrono;

// ✅ ชื่อและขนาดของ Shared Memory
#define SHM_NAME "CroppedImageSHM"
#define SHM_SIZE 1024 * 1024  // 1MB

#define N_FRAME_BUFFER 2 // เก็บผลลัพธ์ 5 เฟรมล่าสุด


struct Object {
  cv::Rect rec;
  int class_id;
  float prob;
  double sharpness;
};

// global variables
int frame_counter = 0;
std::time_t time_begin = std::time(0);
std::time_t last_delta_t = 0;
vector<Object> detected_objects_buffer[N_FRAME_BUFFER]; // บัฟเฟอร์เก็บ Object
vector<Mat> cropped_images_buffer[N_FRAME_BUFFER]; // บัฟเฟอร์เก็บภาพที่ถูกครอบ
int buffer_index = 0;

std::vector<std::string> initialize_labels() {
  // get the list of labels from labelmap
  std::vector<std::string> labels;
  std::ifstream input(LABEL_MAP_FILE);
  for (std::string line; getline(input, line);) {
    labels.push_back(line);
  }
  return labels;
}

void preprocess(
  cv::VideoCapture& cam,
  cv::Mat& original_image,
  cv::Mat& resized_image
) {
  // read frame from camera
  auto success = cam.read(original_image);
  if (!success) {
    std::cerr << "cam fail" << std::endl;
    throw new exception();
  }

  // Resize the original image
  resize(original_image, resized_image, Size(INPUT_SIZE, INPUT_SIZE));

  // Convert input image to Float and normalize
  resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255, 0);
}

std::vector<Object> postprocess(
  float* output_scores_tensor,
  float* output_boxes_tensor,
  int* output_classes_tensor,
  int* output_selected_idx_tensor,
  unsigned long num_selected_idx,
  double cam_width,
  double cam_height
) {
  // filter selected_idx to get non-zeros (don't use selected_idx == 0 because output is zero-padded by tflite)
  std::vector<int> selected_idx;
  for (int i = 0; i < num_selected_idx; i++) {
    auto selected_id = output_selected_idx_tensor[i];
    if (selected_id == 0) {
      continue;
    }
    selected_idx.push_back(selected_id);
  }

  // populate the vector of objects
  std::vector<Object> objects;
  for (auto selected_id : selected_idx) {
    // get box dimensions
    auto xmin = output_boxes_tensor[selected_id * 4 + 0] * cam_width;
    auto ymin = output_boxes_tensor[selected_id * 4 + 1] * cam_height;
    auto xmax = output_boxes_tensor[selected_id * 4 + 2] * cam_width;
    auto ymax = output_boxes_tensor[selected_id * 4 + 3] * cam_height;
    auto width = xmax - xmin;
    auto height = ymax - ymin;

    // populate an object proper
    Object object;
    object.class_id = output_classes_tensor[selected_id];
    object.rec.x = xmin;
    object.rec.y = ymin;
    object.rec.width = width;
    object.rec.height = height;
    object.prob = output_scores_tensor[selected_id];
    objects.push_back(object);
  }

  return objects;
}


#if 0

void send_image_to_shm(cv::Mat &image) {
    // ✅ เปิด Shared Memory
    shared_memory_object shm(open_or_create, SHM_NAME, read_write);
    shm.truncate(SHM_SIZE);

    // ✅ Map Memory
    mapped_region region(shm, read_write);
    void *addr = region.get_address();

    // ✅ แปลงภาพเป็น JPG (Compress เพื่อลดขนาด)
    vector<uchar> buffer;
    cv::imencode(".jpg", image, buffer);

    // ✅ เขียนข้อมูลลง Shared Memory
    size_t size = buffer.size();
    if (size > SHM_SIZE) {
        cerr << "Error: Image too large for Shared Memory!" << endl;
        return;
    }
    memcpy(addr, buffer.data(), size);
}

#endif

void send_image_to_shm(cv::Mat &image) {
    // ✅ เปิด Shared Memory
    shared_memory_object shm(open_or_create, SHM_NAME, read_write);
    shm.truncate(SHM_SIZE);

    // ✅ Map Memory
    mapped_region region(shm, read_write);
    void *addr = region.get_address();

    // ✅ แปลงภาพเป็น JPG (Compress เพื่อลดขนาด)
    vector<uchar> buffer;
    cv::imencode(".jpg", image, buffer);
    size_t size = buffer.size();

    if (size > SHM_SIZE - sizeof(uint64_t)) {
        cerr << "Error: Image too large for Shared Memory!" << endl;
        return;
    }

    // ✅ เขียน Timestamp (8 bytes) ตามด้วยรูปภาพ
    uint64_t timestamp = chrono::duration_cast<chrono::milliseconds>(
                             chrono::system_clock::now().time_since_epoch())
                             .count();
    memcpy(addr, &timestamp, sizeof(uint64_t));
    memcpy(static_cast<uint8_t *>(addr) + sizeof(uint64_t), buffer.data(), size);
}

#if 0
void draw_boxes(std::vector<Object> objects, cv::Mat& original_image, std::vector<string>& labels) {
  // Loop through all Bounding Boxes
  for (int l = 0; l < objects.size(); l++) {
    Object object = objects.at(l);
    auto score = object.prob;
    auto score_rounded = ((float)((int)(score * 100 + 0.5)) / 100);

    Scalar color = Scalar(255, 0, 0);
    auto class_id = object.class_id;
    auto class_label = labels[class_id];

    std::ostringstream label_txt_stream;
    label_txt_stream << class_label << " (" << score_rounded << ")";
    std::string label_txt = label_txt_stream.str();

    // เพิ่ม Padding รอบๆ Bounding Box
    int padding = 30; // ขนาด Padding รอบกล่อง
    cv::Rect expanded_box(
      std::max(0, object.rec.x - padding),                        // เพิ่ม Padding ด้านซ้าย
      std::max(0, object.rec.y - padding),                        // เพิ่ม Padding ด้านบน
      std::min(object.rec.width + 2 * padding, original_image.cols - object.rec.x),  // ขยายความกว้าง
      std::min(object.rec.height + 2 * padding, original_image.rows - object.rec.y)  // ขยายความสูง
    );

    // วาด Bounding Box ขยาย
    cv::rectangle(original_image, expanded_box, color, 2);
    cv::putText(original_image, 
                label_txt,
                cv::Point(expanded_box.x, expanded_box.y - 5),
                cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));

    // ตัดภาพเฉพาะส่วนที่อยู่ใน Bounding Box ขยาย
    if (expanded_box.width <= 0 || expanded_box.height <= 0) {
        std::cerr << "Invalid bounding box, skipping..." << std::endl;
        continue;
    }
    cv::Mat cropped_image = original_image(expanded_box);

    // แสดงภาพ Original และ Cropped Image
    // cv::imshow("Original Image", original_image);
    cv::imshow("Cropped Image", cropped_image);

    // // Save the Cropped Image
    // std::ostringstream filename_stream;
    // filename_stream << "cropped_" << l << ".jpg";
    // std::string filename = filename_stream.str();
    // cv::imwrite(filename, cropped_image);
    // std::cout << "Saved cropped image to " << filename << std::endl;
  }
}
#endif

#if 0
void draw_boxes_and_crop(vector<Object> objects, Mat &original_image, vector<string> &labels) {
    for (const auto &object : objects) {
        if (object.prob > 0.5) { // ✅ ส่งเฉพาะ Object ที่มี prob > 60%
            Rect expanded_box(
                max(0, object.rec.x - 30),
                max(0, object.rec.y - 30),
                min(object.rec.width + 40, original_image.cols - object.rec.x),
                min(object.rec.height + 40, original_image.rows - object.rec.y)
            );

            if (expanded_box.width > 0 && expanded_box.height > 0) {
                Mat cropped_image = original_image(expanded_box);
                std::cout << "image was detected"<< std::endl;
                send_image_to_shm(cropped_image); // ✅ ส่งภาพไป Shared Memory
            }
        }
        
    }
}
#endif

// work_old
#if 0
void draw_boxes_and_crop(vector<Object> objects, Mat &original_image, vector<string> &labels) {
    if (objects.empty()) return;

    detected_objects_buffer[buffer_index] = objects;
    cropped_images_buffer[buffer_index].clear();

    for (const auto &obj : objects) {
        cv::Mat cropped = original_image(obj.rec);
        cropped_images_buffer[buffer_index].push_back(cropped);
    }

    buffer_index = (buffer_index + 1) % N_FRAME_BUFFER;

    if (buffer_index == 0) {
        int best_idx = 0;
        float best_prob = 0.0;

        for (int i = 0; i < N_FRAME_BUFFER; i++) {
            for (const auto &obj : detected_objects_buffer[i]) {
                if (obj.prob > best_prob) {
                    best_prob = obj.prob;
                    best_idx = i;
                }
            }
        }

        if (!cropped_images_buffer[best_idx].empty()) {
            send_image_to_shm(cropped_images_buffer[best_idx][0]); // ✅ เลือก Mat แรกจาก vector
            std::cout << "Send image"<< std::endl;
        }
    }
}

#endif

// ล่าสุด
#if 1

int empty_frame_count = 0; 
void draw_boxes_and_crop(vector<Object> objects, Mat &original_image, vector<string> &labels) {
    // if (objects.empty()) return;


    if (objects.empty()) {
        empty_frame_count++;  // นับจำนวนรอบที่ไม่มี object
        // std::cout << "No objects detected. Empty frame count: " << empty_frame_count << std::endl;

        // ✅ ล้าง buffer ถ้าไม่มี object 10 รอบติดกัน
        if (empty_frame_count >= 20) {
            // std::cout << "Clearing buffer due to 20 empty frames." << std::endl;
            for (int i = 0; i < N_FRAME_BUFFER; i++) {
                detected_objects_buffer[i].clear();
                cropped_images_buffer[i].clear();
            }
            empty_frame_count = 0;  // รีเซ็ตตัวนับ
        }
        return;
    }

    empty_frame_count = 0;

    detected_objects_buffer[buffer_index] = objects;
    cropped_images_buffer[buffer_index].clear();

    int padding = 20; // ปรับขนาด padding ตามต้องการ

    for (const auto &obj : objects) {

      if (obj.prob > 0.5) { 
        // ขยาย Bounding Box ด้วย padding
        cv::Rect expanded_rec = obj.rec;
        expanded_rec.x = max(expanded_rec.x - padding, 0);
        expanded_rec.y = max(expanded_rec.y - padding, 0);
        expanded_rec.width = min(expanded_rec.width + 2 * padding, original_image.cols - expanded_rec.x);
        expanded_rec.height = min(expanded_rec.height + 2 * padding, original_image.rows - expanded_rec.y);

        // Crop ภาพโดยใช้ Bounding Box ที่ขยายแล้ว
        cv::Mat cropped = original_image(expanded_rec);
        cropped_images_buffer[buffer_index].push_back(cropped);
        std::cout << "obj.prob = "<< obj.prob <<std::endl;
      }else{
       return;
      }
    }

    buffer_index = (buffer_index + 1) % N_FRAME_BUFFER;
    std::cout << "buffer index : "<< buffer_index <<std::endl;
    if (buffer_index == 0) {
        int best_idx = 0;
        float best_prob = 0.0;

        for (int i = 0; i < N_FRAME_BUFFER; i++) {
            for (const auto &obj : detected_objects_buffer[i]) {
                if (obj.prob > best_prob) {
                    best_prob = obj.prob;
                    best_idx = i;
                }
            }
        }

        if (!cropped_images_buffer[best_idx].empty()) {
            send_image_to_shm(cropped_images_buffer[best_idx][0]); // ✅ เลือก Mat แรกจาก vector
            std::cout << "Send image"<< best_prob <<std::endl;
        }
    }
}
#endif

#if 1

// void draw_boxes_and_crop(vector<Object> objects, Mat &original_image, vector<string> &labels) {
//     // if (objects.empty()) return;


//     detected_objects_buffer[buffer_index] = objects;
//     cropped_images_buffer[buffer_index].clear();

//     int padding = 20;

//     for (const auto &obj : objects) {
//         cv::Rect expanded_rec = obj.rec;
//         expanded_rec.x = max(expanded_rec.x - padding, 0);
//         expanded_rec.y = max(expanded_rec.y - padding, 0);
//         expanded_rec.width = min(expanded_rec.width + 2 * padding, original_image.cols - expanded_rec.x);
//         expanded_rec.height = min(expanded_rec.height + 2 * padding, original_image.rows - expanded_rec.y);

//         cv::Mat cropped = original_image(expanded_rec);
        
//         // คำนวณค่า sharpness (Laplacian Variance)
//         double sharpness = cv::Laplacian(cropped, CV_64F).var();
        
//         // บันทึกค่าความมั่นใจและ sharpness
//         obj.extra_data["sharpness"] = sharpness;
//         cropped_images_buffer[buffer_index].push_back(cropped);
//     }

//     buffer_index = (buffer_index + 1) % N_FRAME_BUFFER;

//     if (buffer_index == 0) {
//         int best_idx = 0;
//         float best_prob = 0.0;
//         double best_sharpness = 0.0;

//         for (int i = 0; i < N_FRAME_BUFFER; i++) {
//             for (const auto &obj : detected_objects_buffer[i]) {
//                 // พิจารณาเฟรมที่ prob สูง และ sharpness ดีที่สุด
//                 double sharpness = obj.extra_data["sharpness"];
//                 if (obj.prob > best_prob && sharpness > best_sharpness) {
//                     best_prob = obj.prob;
//                     best_sharpness = sharpness;
//                     best_idx = i;
//                 }
//             }
//         }

//         if (!cropped_images_buffer[best_idx].empty()) {
//             send_image_to_shm(cropped_images_buffer[best_idx][0]);
//             std::cout << "Send best image with prob " << best_prob << " and sharpness " << best_sharpness << std::endl;

//             for (int i = 0; i < N_FRAME_BUFFER; i++) {
//                 detected_objects_buffer[i].clear();
//                 cropped_images_buffer[i].clear();
//             }
//         }
//     }
// }

// void draw_boxes_and_crop(vector<Object> objects, Mat &original_image, vector<string> &labels) {
//     if (objects.empty()) return;

//     detected_objects_buffer[buffer_index] = objects;
//     cropped_images_buffer[buffer_index].clear();

//     int padding = 20;

//     for (auto &obj : objects) {  // ✅ เปลี่ยนจาก const auto &obj เป็น auto &obj เพื่อแก้ค่า sharpness ได้
//         cv::Rect expanded_rec = obj.rec;
//         expanded_rec.x = max(expanded_rec.x - padding, 0);
//         expanded_rec.y = max(expanded_rec.y - padding, 0);
//         expanded_rec.width = min(expanded_rec.width + 2 * padding, original_image.cols - expanded_rec.x);
//         expanded_rec.height = min(expanded_rec.height + 2 * padding, original_image.rows - expanded_rec.y);

//         cv::Mat cropped = original_image(expanded_rec);
        
//         // ✅ คำนวณค่า sharpness ใหม่
//         cv::Mat laplacian;
//         cv::Laplacian(cropped, laplacian, CV_64F);
//         cv::Scalar mean, stddev;
//         cv::meanStdDev(laplacian, mean, stddev);
//         double sharpness = stddev[0] * stddev[0];

//         std::cout << "Object detected - prob: " << obj.prob << ", sharpness: " << sharpness << std::endl;

//         obj.sharpness = sharpness;  // ✅ ใช้ field sharpness แทน extra_data
//         cropped_images_buffer[buffer_index].push_back(cropped);
//     }

//     buffer_index = (buffer_index + 1) % N_FRAME_BUFFER;

//     if (buffer_index == 0) {
//         int best_idx = 0;
//         float best_prob = 0.0;
//         double best_sharpness = 0.0;

//         for (int i = 0; i < N_FRAME_BUFFER; i++) {
//             for (auto &obj : detected_objects_buffer[i]) {
//                 if (obj.prob > best_prob && obj.sharpness > best_sharpness) { // ✅ ใช้ obj.sharpness
//                     best_prob = obj.prob;
//                     best_sharpness = obj.sharpness;
//                     best_idx = i;
//                 }
//             }
//         }

//         if (!cropped_images_buffer[best_idx].empty()) {
//             send_image_to_shm(cropped_images_buffer[best_idx][0]);
//             std::cout << "Send best image with prob " << best_prob << " and sharpness " << best_sharpness << std::endl;

//             for (int i = 0; i < N_FRAME_BUFFER; i++) {
//                 detected_objects_buffer[i].clear();
//                 cropped_images_buffer[i].clear();
//             }
//         }
//     }
// }


// int empty_frame_count = 0; 

// void draw_boxes_and_crop(vector<Object> objects, Mat &original_image, vector<string> &labels) {
//     if (objects.empty()) {
//         empty_frame_count++;  // นับจำนวนรอบที่ไม่มี object
//         // std::cout << "No objects detected. Empty frame count: " << empty_frame_count << std::endl;

//         // ✅ ล้าง buffer ถ้าไม่มี object 10 รอบติดกัน
//         if (empty_frame_count >= 20) {
//             // std::cout << "Clearing buffer due to 20 empty frames." << std::endl;
//             for (int i = 0; i < N_FRAME_BUFFER; i++) {
//                 detected_objects_buffer[i].clear();
//                 cropped_images_buffer[i].clear();
//             }
//             empty_frame_count = 0;  // รีเซ็ตตัวนับ
//         }
//         return;
//     }

//     // รีเซ็ตตัวนับเมื่อมี object ถูกตรวจจับ
//     empty_frame_count = 0;


//     detected_objects_buffer[buffer_index] = objects;
//     cropped_images_buffer[buffer_index].clear();

//     int padding = 20;

//     for (auto &obj : objects) {
//      if (obj.prob > 0.5) {
//         cv::Rect expanded_rec = obj.rec;
//         expanded_rec.x = max(expanded_rec.x - padding, 0);
//         expanded_rec.y = max(expanded_rec.y - padding, 0);
//         expanded_rec.width = min(expanded_rec.width + 2 * padding, original_image.cols - expanded_rec.x);
//         expanded_rec.height = min(expanded_rec.height + 2 * padding, original_image.rows - expanded_rec.y);

//         cv::Mat cropped = original_image(expanded_rec);

//         // คำนวณ sharpness
//         cv::Mat laplacian;
//         cv::Laplacian(cropped, laplacian, CV_64F);
//         cv::Scalar mean, stddev;
//         cv::meanStdDev(laplacian, mean, stddev);
//         double sharpness = stddev[0] * stddev[0];

//         obj.sharpness = sharpness;

//         // Debug ค่า object ที่ตรวจพบ
//         std::cout << "Object detected - prob: " << obj.prob 
//                   << ", sharpness: " << sharpness << std::endl;

//         cropped_images_buffer[buffer_index].push_back(cropped);
//       }
//     }

//     buffer_index = (buffer_index + 1) % N_FRAME_BUFFER;

//     if (buffer_index == 0) {
//         int best_idx = -1;
//         float best_prob = -1.0;
//         double best_sharpness = -1.0;

//         for (int i = 0; i < N_FRAME_BUFFER; i++) {
//             for (auto &obj : detected_objects_buffer[i]) {
//                 if (obj.prob > best_prob || (obj.prob >= best_prob && obj.sharpness > best_sharpness)) {  
//                     best_prob = obj.prob;
//                     best_sharpness = obj.sharpness;
//                     best_idx = i;
//                 }
//             }
//         }

//         // Debug: พิมพ์ค่า best ที่เลือกได้
//         std::cout << "Selected best_idx: " << best_idx 
//                   << " with prob: " << best_prob 
//                   << " and sharpness: " << best_sharpness << std::endl;

//         if (best_idx != -1 && !cropped_images_buffer[best_idx].empty()) {
//             send_image_to_shm(cropped_images_buffer[best_idx][0]);
//             std::cout << "Send best image with prob " << best_prob 
//                       << " and sharpness " << best_sharpness << std::endl;
//         } else {
//             std::cout << "No valid object found!" << std::endl;
//         }

//         // ล้าง buffer
//         for (int i = 0; i < N_FRAME_BUFFER; i++) {
//             detected_objects_buffer[i].clear();
//             cropped_images_buffer[i].clear();
//         }
//     }
// }

#endif

void profile_execution() {
  frame_counter++;
  std::time_t delta_t = std::time(0) - time_begin;
  if (delta_t % 60 == 0 && delta_t != last_delta_t) { // delta_t % 60 == 0 can be true multiple times
    std::cout << "Frames Processed in the last minute: " << frame_counter << std::endl;
    frame_counter = 0;
    last_delta_t = delta_t;
  }
}

#if 1

void test() {
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(MODEL_FILE);

  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter, NUM_THREADS);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  std::vector<std::string> labels = initialize_labels();

  std::cout << "Initialized interpreter and labels" << std::endl;

#if 0
  // Declare the camera using GStreamer pipeline
  // std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
#else
  // std::string pipeline = "nvarguscamerasrc exposuretimerange=\"100000 80000000\" ! video/x-raw(memory:NVMM), width=3264, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv ! videoflip method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
  // std::string pipeline = "nvarguscamerasrc exposuretimerange=\"100000 80000000\" ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=30/1 ! nvvidconv ! videoflip method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
  // std::string pipeline = "nvarguscamerasrc exposuretimerange=\"100000 80000000\" ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
  // std::string pipeline = "nvarguscamerasrc exposuretimerange=\"100000 80000000\" ! "
                           "video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=120/1 ! "
                           "nvvidconv ! videoflip method=2 ! "
                           "video/x-raw, format=(string)BGRx ! videoconvert ! "
                           "video/x-raw, format=(string)BGR ! appsink";

  std::string pipeline = "nvarguscamerasrc exposuretimerange=\"100000 20000000\" gainrange=\"1 16\" ispdigitalgainrange=\"1 8\" wbmode=1 ! video/x-raw(memory:NVMM), width=1800, height=1250, format=(string)NV12, framerate=27/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
  
  //std::string pipeline = "nvarguscamerasrc exposuretimerange=\"100000 16000000\" gainrange=\"1 8\" ispdigitalgainrange=\"1 4\" wbmode=1 aeantibanding=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! nvjpegenc ! appsink";

  

#endif
  auto cam = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
  if (!cam.isOpened()) {
    std::cerr << "Cannot open the camera with GStreamer!" << std::endl;
    return;
  }

  // Get camera resolution
  auto cam_width = cam.get(cv::CAP_PROP_FRAME_WIDTH);
  auto cam_height = cam.get(cv::CAP_PROP_FRAME_HEIGHT);

  std::cout << "Got the camera, see cam_width and cam_height: " << cam_width
            << ',' << cam_height << std::endl;

  // Allocate tensor before inference loop
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Start camera loop
  while (true) {
    // Declare image buffers
    cv::Mat original_image;
    cv::Mat resized_image;

    // Capture frame from camera
    cam >> original_image;
    if (original_image.empty()) {
      std::cerr << "Empty frame captured!" << std::endl;
      break;
    }

    // Preprocess the image
    preprocess(
      cam,
      original_image,
      resized_image
    );

    // Declare the input
    float* input = interpreter->typed_input_tensor<float>(0);

    // Feed input
    memcpy(input, resized_image.data,
           resized_image.total() * resized_image.elemSize());

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    // Declare the output buffers
    float* output_boxes_tensor = interpreter->typed_output_tensor<float>(0);
    float* output_scores_tensor = interpreter->typed_output_tensor<float>(1);
    int* output_classes_tensor = interpreter->typed_output_tensor<int>(2);
    int* output_selected_idx_tensor = interpreter->typed_output_tensor<int>(3);

    auto num_selected_idx = *(interpreter->output_tensor(3)->dims[0].data);

    // Get boxes from the output buffers
    std::vector<Object> objects = postprocess(
      output_scores_tensor, 
      output_boxes_tensor, 
      output_classes_tensor, 
      output_selected_idx_tensor, 
      num_selected_idx,
      cam_width,
      cam_height
    );

    // Draw the boxes on the original image
    draw_boxes_and_crop(objects, original_image, labels);

    // Profile the code whenever you can
    // profile_execution();

    // Show image on screen
    // cv::imshow("QuickYOLO", original_image);

    // // Go to next frame after 1ms if no key pressed
    // auto k = cv::waitKey(1) & 0xFF;
    // if (k != NO_PRESS) {
    //   std::cout << "See k: " << k << std::endl;
    //   break;
    // }
  }
}

#endif


#if 0
#define DEBUG_MODE
// Static variables to avoid reloading the model
static std::unique_ptr<tflite::FlatBufferModel> model;
static std::unique_ptr<tflite::Interpreter> interpreter;

// Initialize the model once
void initialize_model() {
    if (!model) {
        model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILE);
        TFLITE_MINIMAL_CHECK(model != nullptr);

        tflite::ops::builtin::BuiltinOpResolver resolver;
        compute_engine::tflite::RegisterLCECustomOps(&resolver);

        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter, NUM_THREADS);
        TFLITE_MINIMAL_CHECK(interpreter != nullptr);

        // Allocate once
        TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    }
}

void test() {
    // Load model once
    initialize_model();

    // Initialize labels once
    static std::vector<std::string> labels = initialize_labels();

    std::cout << "Initialized interpreter and labels" << std::endl;

    // Optimized GStreamer pipeline (lower resolution and frame rate)
    std::string pipeline = "nvarguscamerasrc exposuretimerange=\"100000 80000000\" ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

    cv::VideoCapture cam(pipeline, cv::CAP_GSTREAMER);
    if (!cam.isOpened()) {
        std::cerr << "Cannot open the camera with GStreamer!" << std::endl;
        return;
    }

    // Get camera resolution
    auto cam_width = cam.get(cv::CAP_PROP_FRAME_WIDTH);
    auto cam_height = cam.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Got the camera: " << cam_width << "x" << cam_height << std::endl;

    // Start camera loop
    while (true) {
        cv::Mat original_image, resized_image;

        // Capture frame from camera
        cam >> original_image;
        if (original_image.empty()) {
            std::cerr << "Empty frame captured!" << std::endl;
            break;
        }

        // Preprocess (use reference instead of copy)
        preprocess(cam, original_image, resized_image);

        // Get input tensor memory directly
        cv::Mat input_tensor(cv::Size(320, 320), CV_32F, interpreter->typed_input_tensor<float>(0));
        resized_image.convertTo(input_tensor, CV_32F);

        // Run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

        // Get output tensors
        float* output_boxes_tensor = interpreter->typed_output_tensor<float>(0);
        float* output_scores_tensor = interpreter->typed_output_tensor<float>(1);
        int* output_classes_tensor = interpreter->typed_output_tensor<int>(2);
        int* output_selected_idx_tensor = interpreter->typed_output_tensor<int>(3);

        auto num_selected_idx = *(interpreter->output_tensor(3)->dims[0].data);

        // Postprocess results
        std::vector<Object> objects = postprocess(
            output_scores_tensor, 
            output_boxes_tensor, 
            output_classes_tensor, 
            output_selected_idx_tensor, 
            num_selected_idx,
            cam_width,
            cam_height
        );

        // Draw and crop using optimized bounding box handling
        draw_boxes_and_crop(objects, original_image, labels);

        // Profile execution time (optional)
        profile_execution();

        // Display frame (only in debug mode)
       // #ifdef DEBUG_MODE
        //cv::imshow("QuickYOLO", original_image);
        if ((cv::waitKey(1) & 0xFF) != NO_PRESS) {
            break;
        }
        //#endif
    }
}

#endif
int main(int argc, char** argv) {
  test();
  return 0;
}

