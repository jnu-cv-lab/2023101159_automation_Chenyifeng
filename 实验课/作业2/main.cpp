#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // --------------------------
    // 任务1：读取测试图片
    // --------------------------
    if (argc != 2) {
        cout << "用法: ./opencv_demo <图片路径>" << endl;
        return -1;
    }
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        cout << "❌ 无法读取图片，请检查路径！" << endl;
        return -1;
    }

    // --------------------------
    // 任务2：输出图像基本信息
    // --------------------------
    cout << "=== 图像基本信息 ===" << endl;
    cout << "宽度: " << img.cols << " 像素" << endl;
    cout << "高度: " << img.rows << " 像素" << endl;
    cout << "通道数: " << img.channels() << endl;
    
    string type_str;
    int type = img.type();
    if (type == CV_8UC3) type_str = "8位无符号，3通道（彩色）";
    else if (type == CV_8UC1) type_str = "8位无符号，1通道（灰度）";
    else type_str = "其他类型";
    cout << "像素数据类型: " << type_str << endl;

    // --------------------------
    // 任务3：显示原图
    // --------------------------
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", img);

    // --------------------------
    // 任务4：转换为灰度图并显示
    // --------------------------
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    namedWindow("灰度图", WINDOW_AUTOSIZE);
    imshow("灰度图", gray_img);

    // --------------------------
    // 任务5：保存灰度图
    // --------------------------
    imwrite("gray_output.jpg", gray_img);
    cout << "✅ 灰度图已保存为: gray_output.jpg" << endl;

    // --------------------------
    // 任务6：NumPy 风格简单操作（C++ 实现）
    // --------------------------
    // 1. 输出指定像素值（以(100,100)为例）
    if (img.rows > 100 && img.cols > 100) {
        Vec3b pixel = img.at<Vec3b>(100, 100);
        cout << "像素(100,100)的 BGR 值: " 
             << (int)pixel[0] << ", " 
             << (int)pixel[1] << ", " 
             << (int)pixel[2] << endl;
    }

    // 2. 裁剪左上角 200x200 区域并保存
    if (img.rows >= 200 && img.cols >= 200) {
        Rect roi(0, 0, 200, 200);
        Mat cropped_img = img(roi);
        imwrite("cropped_output.jpg", cropped_img);
        cout << "✅ 左上角 200x200 区域已保存为: cropped_output.jpg" << endl;
    }

    // 等待按键后关闭窗口
    waitKey(0);
    destroyAllWindows();
    return 0;
}