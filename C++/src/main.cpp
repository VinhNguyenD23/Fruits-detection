#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

// Đọc ảnh và đảm bảo ảnh có 3 kênh (BGR)
Mat readImage(const string &imagePath)
{
    Mat image = imread(imagePath);
    if (image.empty())
    {
        cerr << "Không thể đọc ảnh từ đường dẫn: " << imagePath << endl;
        exit(-1);
    }
    // Nếu ảnh chỉ có 1 kênh (grayscale), chuyển đổi sang BGR (3 kênh)
    if (image.channels() == 1)
    {
        cvtColor(image, image, COLOR_GRAY2BGR);
    }
    Mat resized;
    resize(image, resized, Size(800, 800), INTER_LINEAR);
    return resized;
}

// Hàm nạp mô hình ONNX và thực hiện suy luận
void runOnnxModel(const string &modelPath, const Mat &image)
{
    // Tải mô hình ONNX với OpenCV dnn
    Net net = readNetFromONNX(modelPath);

    // Tạo blob từ ảnh với định dạng NCHW
    Mat blob = blobFromImage(image, 1.0 / 255.0, Size(224, 224), Scalar(0, 0, 0), true, false);

    // Chuyển đổi từ NCHW sang NHWC nếu mô hình yêu cầu NHWC
    Mat nhwcBlob;
    Mat channels[3];
    for (int i = 0; i < 3; ++i)
    {
        channels[i] = Mat(blob.size[2], blob.size[3], CV_32F, blob.ptr(0, i)); // Kênh từng phần của blob
    }
    cv::merge(channels, 3, nhwcBlob); // Kết hợp lại thành NHWC (H, W, C)

    // Thêm batch dimension ở đầu bằng cách reshape lại thành [N, H, W, C]
    nhwcBlob = nhwcBlob.reshape(1, {blob.size[0], blob.size[2], blob.size[3], blob.size[1]});

    // Đưa blob với định dạng NHWC vào mạng nếu mô hình yêu cầu NHWC
    net.setInput(nhwcBlob);

    // Thực hiện suy luận
    Mat output = net.forward();
    cout << output.size << endl;
    // Giả sử đầu ra bao gồm các tọa độ của khung chữ nhật và chỉ mục của nhãn
    float *data = (float *)output.data;
    // vector<string> dataLabel = {"Apple", "Banana", "Cherry", "Chickoo", "Grapes", "Kiwi", "Mango", "Orange", "Strawberry"};
    vector<string> dataLabel = {"Apple", "Banana", "Cabbage", "Carrot", "Cherry", "Chickoo", "Cucumber", "Eggplant", "grapes", "Kiwi", "Mango", "Orange", "Pear", "Strawberry", "Zucchini"};
    // vector<string> dataLabel = {"Tomato", "Orange", "Banana", "Watermelon", "Pear", "Grape", "Tangerine", "Apple", "Pineapple", "Mango"};
    for (int i = 0; i < output.total(); ++i)
    {
        cout << fixed << setprecision(12) << "Confidence['" << dataLabel[i] << "']: " << data[i] << endl;
    }
    int width = image.cols;
    int height = image.rows;
    cout << "width: " << width << endl;
    cout << "height: " << height << endl;
    // Tính toán khung chữ nhật từ đầu ra
    int labelIndex = static_cast<int>(data[0]); // Chỉ mục của nhãn đối tượng
    float maxScore = data[0];

    for (int i = 1; i < output.total(); ++i)
    {
        if (data[i] > maxScore)
        {
            maxScore = data[i];
            labelIndex = i;
        }
    }
    Rect boundingBox(
        static_cast<int>(width * 0.1),                 // x1
        static_cast<int>(height * 0.1),                // y1
        static_cast<int>((data[2] - data[0]) * width), // width
        static_cast<int>((data[3] - data[1]) * height) // height
    );
    // Vẽ khung chữ nhật và nhãn lên ảnh
    // rectangle(image, boundingBox, Scalar(0, 255, 0), 2);
    putText(image, "Object: " + dataLabel[labelIndex] + " " + format("%.2f", data[labelIndex] * 100) + "%", boundingBox.tl(), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

    // Hiển thị ảnh kết quả
    imshow("Object Detection", image);
    waitKey(0);
}

int main(int argc, char **argv)
{
    // Kiểm tra số lượng tham số
    if (argc < 3)
    {
        cerr << "Sử dụng: " << argv[0] << " <đường_dẫn_model.onnx> <đường_dẫn_ảnh>" << endl;
        return -1;
    }

    // Lấy đường dẫn mô hình và ảnh từ tham số dòng lệnh
    string modelPath = argv[1];
    string imagePath = argv[2];
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    Mat image = readImage(imagePath);
    runOnnxModel(modelPath, image);

    return 0;
}
