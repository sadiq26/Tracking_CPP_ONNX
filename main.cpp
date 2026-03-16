
#include "L2F_Tracking_V2.h"

#include <opencv2/tracking.hpp>
#include <algorithm>
#include <opencv2/tracking/tracking_legacy.hpp> // For MOSSE





void processVideo(const std::string& modelPath,
    const std::string& videoPath,
    const std::string& classNamesPath,
    bool useGPU = false,
    float confThreshold = 0.3f,
    float iouThreshold = 0.4f)
{
    try
    {
        // Load class names
        const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
        if (classNames.empty())
        {
            std::cerr << "Error: Empty class names file." << std::endl;
            return;
        }

        // Initialize detector
        YOLODetector detector(modelPath, useGPU, cv::Size(640, 640));
        std::cout << "Model was initialized successfully." << std::endl;

        // Open video capture
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
            return;
        }

        // Get video properties
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        // Add this after opening the video to debug
        std::cout << "Total frames in video: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
        std::cout << "Video duration (seconds): " << cap.get(cv::CAP_PROP_FRAME_COUNT) / fps << std::endl;
        std::cout << "Video info: " << frameWidth << "x" << frameHeight
            << ", FPS: " << fps << std::endl;

        // Optional: Create video writer to save output
        // cv::VideoWriter writer("output_video.mp4", 
        //                       cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
        //                       fps, 
        //                       cv::Size(frameWidth, frameHeight));

        cv::Mat frame;
        int frameCount = 0;

        std::cout << "Processing video... Press 'q' or ESC to quit." << std::endl;

        while (true)
        {
            // Read frame
            cap >> frame;
            if (frame.empty())
            {
                std::cout << "End of video reached." << std::endl;
                break;
            }

            // Perform detection
            std::vector<Detection> results = detector.detect(frame, confThreshold, iouThreshold);

            // Visualize results on frame
            utils::visualizeDetection(frame, results, classNames);

            // Add frame counter text
            frameCount++;
            std::string frameText = "Frame: " + std::to_string(frameCount);
            cv::putText(frame, frameText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            // Optional: Write frame to output video
            // writer.write(frame);

            // Display frame
            cv::imshow("Video Detection", frame);

            // Check for exit key
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q' || key == 'Q' || key == 27) // 27 is ESC key
            {
                std::cout << "Processing interrupted by user." << std::endl;
                break;
            }
        }

        // Release resources
        cap.release();
        // writer.release();
        cv::destroyAllWindows();

        std::cout << "Video processing completed. Total frames processed: " << frameCount << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during video processing: " << e.what() << std::endl;
    }
}

// Alternative version with webcam support
void processWebcam(const std::string& modelPath,
    const std::string& classNamesPath,
    int cameraId = 0,
    bool useGPU = false,
    float confThreshold = 0.3f,
    float iouThreshold = 0.4f)
{
    try
    {
        // Load class names
        const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
        if (classNames.empty())
        {
            std::cerr << "Error: Empty class names file." << std::endl;
            return;
        }

        // Initialize detector
        YOLODetector detector(modelPath, useGPU, cv::Size(640, 640));
        std::cout << "Model was initialized successfully." << std::endl;

        // Open webcam
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open webcam with ID: " << cameraId << std::endl;
            return;
        }

        // Set webcam properties (optional)
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        std::cout << "Webcam started. Press 'q' or ESC to quit." << std::endl;

        cv::Mat frame;
        int frameCount = 0;

        while (true)
        {
            // Capture frame
            cap >> frame;
            if (frame.empty())
            {
                std::cerr << "Error: Empty frame captured." << std::endl;
                break;
            }

            // Perform detection
            std::vector<Detection> results = detector.detect(frame, confThreshold, iouThreshold);

            // Visualize results on frame
            utils::visualizeDetection(frame, results, classNames);

            // Add FPS counter
            frameCount++;
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(cap.get(cv::CAP_PROP_FPS)));
            cv::putText(frame, fpsText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            // Display frame
            cv::imshow("Webcam Detection", frame);

            // Check for exit key
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q' || key == 'Q' || key == 27)
            {
                std::cout << "Webcam processing stopped by user." << std::endl;
                break;
            }
        }

        // Release resources
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during webcam processing: " << e.what() << std::endl;
    }
}

// Example usage in main function:
void processWebcam_v2(const std::string& modelPath,
    const std::string& classNamesPath,
    int cameraId = 0,
    bool useGPU = false,
    float confThreshold = 0.3f,
    float iouThreshold = 0.4f)
{
    try
    {
        // Load class names
        const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
        if (classNames.empty())
        {
            std::cerr << "Error: Empty class names file." << std::endl;
            return;
        }

        // Initialize detector
        YOLODetector detector(modelPath, useGPU, cv::Size(640, 640));
        std::cout << "Model was initialized successfully." << std::endl;

        // Open webcam
        //cv::VideoCapture cap(cameraId);
        cv::VideoCapture cap(cameraId, cv::CAP_DSHOW);


        cv::Mat warmup;
        for (int i = 0; i < 10; i++) {
            cap >> warmup;
            cv::waitKey(50);
        }


        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open webcam with ID: " << cameraId << std::endl;
            return;
        }

        // Set webcam properties
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        // Create window FIRST with specific flags
        cv::namedWindow("Webcam Detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Webcam Detection", 1280, 720);

        std::cout << "Webcam started. Press 'q' or ESC to quit." << std::endl;

        // Allow time for window to initialize
        cv::waitKey(100);

        cv::Mat frame;
        int frameCount = 0;

        while (true)
        {
            // Capture frame
            cap >> frame;
            if (frame.empty())
            {
                std::cerr << "Error: Empty frame captured." << std::endl;
                break;
            }
            cout << "Frame is being processed" << endl;
            // Perform detection
            std::vector<Detection> results = detector.detect(frame, confThreshold, iouThreshold);

            // Visualize results on frame
            utils::visualizeDetection(frame, results, classNames);

            // Add FPS counter
            frameCount++;
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(cap.get(cv::CAP_PROP_FPS)));
            cv::putText(frame, fpsText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            // Display frame
            cv::imshow("Webcam Detection", frame);

            // Force window update and check for key
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27)
            {
                std::cout << "Webcam processing stopped by user." << std::endl;
                break;
            }
        }

        // Release resources
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during webcam processing: " << e.what() << std::endl;
    }
}

void testWebcamSimple(int cameraId = 0) {
    cout << "Running webcam test with DSHOW backend..." << endl;

    // Use DSHOW instead of MSMF
    cv::VideoCapture cap(cameraId, cv::CAP_DSHOW);

    if (!cap.isOpened()) {
        std::cout << "Failed to open camera with DSHOW" << std::endl;

        // Try MSMF as fallback
        std::cout << "Trying MSMF..." << std::endl;
        cap.open(cameraId, cv::CAP_MSMF);

        if (!cap.isOpened()) {
            std::cout << "Failed to open camera with any backend" << std::endl;
            return;
        }
    }

    // Set resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::cout << "Camera opened successfully!" << std::endl;
    std::cout << "Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    cv::namedWindow("Test", cv::WINDOW_NORMAL);

    cv::Mat frame;
    int frameCount = 0;

    // Try to grab frames
    for (int i = 0; i < 100; i++) {
        if (cap.grab()) {  // Use grab() instead of >>
            cap.retrieve(frame);

            if (!frame.empty()) {
                cv::imshow("Test", frame);
                std::cout << "Frame " << frameCount++ << " captured" << std::endl;
            }
        }
        else {
            std::cout << "Failed to grab frame " << i << std::endl;
        }

        int key = cv::waitKey(30);
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
}


// Helper function to calculate IoU
float calculateIoU(const cv::Rect2d& box1, const cv::Rect2d& box2) {
    double interArea = (box1 & box2).area();
    double unionArea = box1.area() + box2.area() - interArea;
    return (unionArea > 0) ? static_cast<float>(interArea / unionArea) : 0.0f;
}


// Helper function to calculate IoU
float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    double interArea = (box1 & box2).area();
    double unionArea = box1.area() + box2.area() - interArea;
    return (unionArea > 0) ? static_cast<float>(interArea / unionArea) : 0.0f;
}





void processWebcam_with_MOSSE(const string& modelPath,
    const string& classNamesPath,
    const string& outputVideoPath = "output.mp4", // Add output path parameter
    int cameraId = 0,
    bool useGPU = false,
    float confThreshold = 0.3f,
    float iouThreshold = 0.4f)
{
    try
    {
        // Load class names
        const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
        if (classNames.empty())
        {
            std::cerr << "Error: Empty class names file." << std::endl;
            return;
        }

        // Initialize detector
        YOLODetector detector(modelPath, useGPU, cv::Size(640, 640));
        std::cout << "Model was initialized successfully." << std::endl;

        // Open webcam
        cv::VideoCapture cap(cameraId, cv::CAP_DSHOW);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open webcam." << std::endl;
            return;
        }

        // Set resolution
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        // Get video properties for saving
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0; // Default to 30 if can't get FPS

        cv::waitKey(500);

        // ===== VIDEO WRITER INITIALIZATION =====
        // Create video writer object
        cv::VideoWriter videoWriter;

        // Define codec and create video file
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // MP4 codec
        // Alternative codecs:
        // int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D'); // AVI
        // int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // Motion JPEG

        bool videoOpened = videoWriter.open(outputVideoPath, fourcc, fps, cv::Size(frameWidth, frameHeight));

        if (!videoOpened)
        {
            std::cerr << "Error: Could not create output video file: " << outputVideoPath << std::endl;
            std::cerr << "Trying alternative codec..." << std::endl;

            // Try alternative codec
            fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            videoOpened = videoWriter.open(outputVideoPath, fourcc, fps, cv::Size(frameWidth, frameHeight));
        }

        if (videoOpened)
        {
            std::cout << "Output video will be saved to: " << outputVideoPath << std::endl;
            std::cout << "Video properties: " << frameWidth << "x" << frameHeight << " @ " << fps << " fps" << std::endl;
        }
        else
        {
            std::cerr << "Warning: Could not create output video file. Will proceed without saving." << std::endl;
        }
        // ===== END VIDEO WRITER INITIALIZATION =====

        cv::namedWindow("MOSSE + YOLO Tracking", cv::WINDOW_NORMAL);
        cv::resizeWindow("MOSSE + YOLO Tracking", 640, 480);

        // Tracking structures
        struct TrackedObject {
            cv::Ptr<cv::legacy::TrackerMOSSE> tracker;
            cv::Rect bbox;
            int classId;
            float confidence;
            int age;
            int maxAge;
            int missedDetections;

            TrackedObject() : age(0), maxAge(30), missedDetections(0), confidence(0.0f) {}
        };

        std::vector<TrackedObject> trackedObjects;

        int detectionInterval = 45;
        int frameCount = 0;
        float iouMatchingThreshold = 0.3f;
        cv::Mat frame;

        std::cout << "MOSSE tracking started. Press 'q' to quit, 'd' for manual detection." << std::endl;
        std::cout << "Saving video to: " << outputVideoPath << std::endl;

        while (true)
        {
            if (!cap.grab()) continue;
            cap.retrieve(frame);
            if (frame.empty()) continue;

            frameCount++;
            bool runDetection = (frameCount % detectionInterval == 0) || trackedObjects.empty();

            // Update existing trackers (same as before)
            std::vector<bool> activeTrackers(trackedObjects.size(), false);

            for (size_t i = 0; i < trackedObjects.size(); i++)
            {
                auto& obj = trackedObjects[i];
                cv::Rect2d newBbox;

                bool ok = obj.tracker->update(frame, newBbox);

                if (ok && newBbox.width > 0 && newBbox.height > 0)
                {
                    obj.bbox = newBbox;
                    obj.age++;
                    obj.missedDetections = 0;
                    activeTrackers[i] = true;
                }
                else
                {
                    obj.missedDetections++;
                    if (obj.missedDetections < 8)
                    {
                        activeTrackers[i] = true;
                    }
                }
            }

            // Remove lost trackers
            std::vector<TrackedObject> newTrackedObjects;
            for (size_t i = 0; i < trackedObjects.size(); i++)
            {
                if (activeTrackers[i])
                {
                    newTrackedObjects.push_back(trackedObjects[i]);
                }
            }
            trackedObjects = newTrackedObjects;

            // Run detection if needed
            if (runDetection)
            {
                std::vector<Detection> detections = detector.detect(frame, confThreshold, iouThreshold);

                // Match detections with existing trackers
                std::vector<bool> matchedDetections(detections.size(), false);
                std::vector<bool> matchedTrackers(trackedObjects.size(), false);

                for (size_t i = 0; i < trackedObjects.size(); i++)
                {
                    for (size_t j = 0; j < detections.size(); j++)
                    {
                        float iou = calculateIoU(trackedObjects[i].bbox, detections[j].box);

                        if (iou > iouMatchingThreshold &&
                            trackedObjects[i].classId == detections[j].classId)
                        {
                            matchedTrackers[i] = true;
                            matchedDetections[j] = true;
                            trackedObjects[i].tracker->init(frame, detections[j].box);
                            trackedObjects[i].bbox = detections[j].box;
                            trackedObjects[i].confidence = detections[j].conf;
                            break;
                        }
                    }
                }

                // Create new trackers for unmatched detections
                for (size_t j = 0; j < detections.size(); j++)
                {
                    if (!matchedDetections[j])
                    {
                        TrackedObject newObj;
                        newObj.bbox = detections[j].box;
                        newObj.classId = detections[j].classId;
                        newObj.confidence = detections[j].conf;

                        newObj.tracker = cv::legacy::TrackerMOSSE::create();
                        newObj.tracker->init(frame, newObj.bbox);

                        trackedObjects.push_back(newObj);
                    }
                }

                std::cout << "Detection: " << detections.size()
                    << " objects, Trackers: " << trackedObjects.size() << std::endl;
            }

            // Create display frame with annotations
            cv::Mat displayFrame = frame.clone();

            for (const auto& obj : trackedObjects)
            {
                cv::Scalar color = cv::Scalar(0, 255, 0);
                if (obj.classId == 0) color = cv::Scalar(0, 0, 255);
                else if (obj.classId == 2) color = cv::Scalar(255, 0, 0);
                else if (obj.classId == 5) color = cv::Scalar(255, 255, 0);
                else if (obj.classId == 7) color = cv::Scalar(255, 0, 255);

                cv::rectangle(displayFrame, obj.bbox, color, 2);

                std::string label = classNames[obj.classId] + " (" +
                    std::to_string(static_cast<int>(obj.confidence * 100)) + "%)";

                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

                cv::rectangle(displayFrame,
                    cv::Point(obj.bbox.x, obj.bbox.y - labelSize.height - 5),
                    cv::Point(obj.bbox.x + labelSize.width, obj.bbox.y),
                    color, cv::FILLED);

                cv::putText(displayFrame, label,
                    cv::Point(obj.bbox.x, obj.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            }

            // Add info text
            std::string infoText = "Frame: " + std::to_string(frameCount) +
                " | Trackers: " + std::to_string(trackedObjects.size());
            cv::putText(displayFrame, infoText, cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

            // Add recording indicator
            if (videoWriter.isOpened())
            {
                cv::circle(displayFrame, cv::Point(30, 50), 8, cv::Scalar(0, 0, 255), -1);
                cv::putText(displayFrame, "REC", cv::Point(45, 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }

            // ===== WRITE FRAME TO VIDEO =====
            if (videoWriter.isOpened())
            {
                videoWriter.write(displayFrame);
            }
            // ===== END VIDEO WRITING =====

            cv::imshow("MOSSE + YOLO Tracking", displayFrame);

            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) break;
            else if (key == 'd' || key == 'D') frameCount = 0;
        }

        // ===== RELEASE VIDEO WRITER =====
        if (videoWriter.isOpened())
        {
            videoWriter.release();
            std::cout << "Video saved successfully to: " << outputVideoPath << std::endl;
        }
        // ===== END VIDEO WRITER RELEASE =====

        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during tracking: " << e.what() << std::endl;
    }
}













//int main()
 //{
 //string model_path = "E:/Coding_Space/New folder/L2F_Tracking_V2/models/yolov5n.onnx";
 //string class_names_path = "E:/Coding_Space/New folder/L2F_Tracking_V2/models/classes.txt";

                    // Use KCF + YOLO tracking
 //processWebcam_with_KCF_advanced(model_path, class_names_path, 0, false, 0.3f, 0.4f);

                    // Or use the advanced version
                    // processWebcam_with_KCF_advanced(model_path, class_names_path, 0, false, 0.3f, 0.4f);

 //return 0;
 //}

int main()
{
    cout << "PROGREMA RUNNING" << endl;

    string model_path = "E:/Coding_Space/New folder/L2F_Tracking_V2/models/yolov5n.onnx";
    string class_names_path = "E:/Coding_Space/New folder/L2F_Tracking_V2/models/classes.txt";
    string video_path_path = "E:/Coding_Space/New folder/L2F_Tracking_V2/media/longer_Tracker.mp4";
    string outputVideoPath = "E:/Coding_Space/New folder/L2F_Tracking_V2/media/output.mp4";


        processWebcam_with_MOSSE(model_path, class_names_path, outputVideoPath, 0, false, 0.3f, 0.4f);
                 
    // For webcam processing
  //  processWebcam_v2(model_path, class_names_path, 0, false, 0.3f, 0.4f);
    //(model_path, class_names_path, 0, false, 0.3f, 0.4f);

    //testWebcamSimple();
    return 0;
}



