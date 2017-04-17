#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void handle_image() {
    Mat img;
    img = imread("../images/sample.png", 1);

    imshow("image", img);

    waitKey(0);
    destroyAllWindows();
}

void handle_image2() {
    Mat img;
    string imgfile = "../images/sample.png";
    img = imread(imgfile, IMREAD_GRAYSCALE);

    namedWindow("image", WINDOW_NORMAL);
    imshow("image", img);

    int key = waitKey(0);

    if (key == 27) {
        destroyAllWindows();
    } else if (key == 's') {
        imwrite("grayscale.png", img);
        destroyAllWindows();
    }
}

void contour() {
    Mat img, imgGray, edge;

    vector<vector<Point>> contours;

    string imgfile = "../images/contour.jpg";
    img = imread(imgfile);
    cvtColor(img, imgGray, CV_BGR2GRAY);

    Canny(imgGray, edge, 100, 200);

    findContours(edge, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    imshow("edge", edge);
    drawContours(img, contours, -1, Scalar(0, 255, 0), 1);
    imshow("contour", img);

    waitKey(0);
    destroyAllWindows();
}

void contour_approx() {
    Mat img, img2, imgGray, edge;

    vector<Point> contour, approx;
    vector<vector<Point>> contours, contours2, approxs;

    string imgfile = "../images/contour2.png";
    img = imread(imgfile);
    cvtColor(img, imgGray, CV_BGR2GRAY);
    img.copyTo(img2);

    Canny(imgGray, edge, 100, 200);

    findContours(edge, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    contour = contours[0];
    contours2.push_back(contour);
    drawContours(img, contours2, 0, Scalar(0, 255, 0), 3);

    double epsilon = 0.1 * arcLength(contours[0], true);
    approxPolyDP(contour, approx, epsilon, true);

    approxs.push_back(approx);

    for ( int i = 0; i < approx.size(); i++ ) {
        cout << approx[0] << " : ";
    }

    drawContours(img2, approxs, 0, Scalar(0, 255, 0), 3);
    imshow("Contour", img);
    imshow("Approx", img2);

    waitKey(0);
    destroyAllWindows();
}

void warp_affine() {

    Point2f pts1[3];
    Point2f pts2[3];

    Mat img;
    Mat warp_dst, warp_mat;

    img = imread("../images/transform.png");

    pts1[0] = Point2f( 50, 50 );
    pts1[1] = Point2f( 200, 50 );
    pts1[2] = Point2f( 20, 200 );

    pts2[0] = Point2f( 70, 100 );
    pts2[1] = Point2f( 220, 50 );
    pts2[2] = Point2f( 150, 250 );

    warp_mat = getAffineTransform( pts1, pts2 );

    warpAffine( img, warp_dst, warp_mat, Size(350, 300) );

    imshow("original", img);
    imshow("Affine Transform", warp_dst);

    waitKey(0);
    destroyAllWindows();
}

void warp_perspective() {

    Point2f pts1[4], pts2[4];
    Point2f topLeft, topRight, bottomLeft, bottomRight;
    float w1, w2, h1, h2, minWidth, minHeight;

    Mat img;
    Mat warp_mat, warp_dst;

    img = imread("../images/transform.jpg");

    topLeft     = Point2f( 127, 157 );
    topRight    = Point2f( 440, 152 );
    bottomRight = Point2f( 578, 526 );
    bottomLeft  = Point2f( 54, 549 );

    pts1[0] = topLeft;
    pts1[1] = topRight;
    pts1[2] = bottomRight;
    pts1[3] = bottomLeft;

    w1 = abs(bottomRight.x - bottomLeft.x);
    w2 = abs(topRight.x - topLeft.x);
    h1 = abs(topRight.y - bottomRight.y);
    h2 = abs(topLeft.y - bottomLeft.y);
    minWidth = min(w1, w2);
    minHeight = min(h1, h2);

    pts2[0] = Point2f(0, 0);
    pts2[1] = Point2f(minWidth - 1, 0);
    pts2[2] = Point2f(minWidth - 1, minHeight - 1);
    pts2[3] = Point2f(0, minHeight - 1);

    warp_mat = getPerspectiveTransform(pts1, pts2);

    warpPerspective(img, warp_dst, warp_mat, Size2f(minWidth, minHeight));

    imshow("original", img);
    imshow("Warp Transform", warp_dst);

    waitKey(0);
    destroyAllWindows();
}

void global_threshold() {

    Mat img, th;

    String imgfile = "../images/document.jpg";
    img = imread(imgfile, IMREAD_GRAYSCALE);

    // Resize image
    float r = (float) (600.0 / img.rows);
    Size2f dim = Size2f(img.cols * r, 600);
    resize(img, img, dim, INTER_AREA);

    String windowName = "Window";
    String trackbarName = "Threshold";

    int lowThreshold = 70;
    // Make Window and Trackbar
    namedWindow(windowName);
    createTrackbar(trackbarName, windowName, &lowThreshold, 255);

    // Loop for get trackbar pos and process it
    while (true) {
        // Allocate destination image
        th = Mat::zeros(img.size(), CV_32FC1);
        // Get Position in trackbar
        int pos = getTrackbarPos(trackbarName, windowName);
        cout << "trackbarpos : " << pos << endl;
        // Apply Threshold
        threshold(img, th, pos, 255, CV_THRESH_BINARY);
        // Show in window
        imshow(windowName, th);

        // wait for ESC key to exit
        int k = waitKey(0);
        if ( k == 27 ) {
            destroyAllWindows();
            break;
        }
    }
}

void adaptive_threshold() {

    Mat img, blur, resultWithoutBlur, resultWithBlur;

    String imgfile = "../images/document.jpg";
    img = imread(imgfile, IMREAD_GRAYSCALE);

    // Resize image
    float r = (float) (600.0 / img.rows);
    Size2f dim = Size2f(img.cols * r, 600);
    resize(img, img, dim, CV_INTER_AREA);

    // Blur image and apply adaptive threshold
    GaussianBlur(img, blur, Size(5, 5), 0);
    adaptiveThreshold(img, resultWithoutBlur, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 10);
    adaptiveThreshold(blur, resultWithBlur, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 10);
    imshow("Without Blur", resultWithoutBlur);
    imshow("With Blur", resultWithBlur);

    waitKey(0);
    destroyAllWindows();
}

bool compareContourAreas(vector<Point> contour1, vector<Point> contour2) {
    double i = fabs( contourArea(Mat(contour1)) );
    double j = fabs( contourArea(Mat(contour2)) );
    return ( i < j );
}

vector<Point> compute(vector<Point> points, float (*func)(float, float)) {

    vector<Point> result;

    float min, max;
    Point minPoint, maxPoint;
    min = 0; max = 0;
    for ( auto point : points ) {
        float sum = func(point.x, point.y);
        if ( min == 0 || min > sum ) {
            minPoint = point;
            min = sum;
        }
        if ( max == 0 || max < sum ) {
            maxPoint = point;
            max = sum;
        }
    }
    cout << "min : " << minPoint << " max : " << maxPoint << endl;

    result.emplace_back(minPoint);
    result.emplace_back(maxPoint);

    return result;
}

vector<Point> order_points(vector<Point> points) {

    vector<Point> rect;

    // initialzie a list of coordinates that will be ordered
    // such that the first entry in the list is the top-left,
    // the second entry is the top-right, the third is the
    // bottom-right, and the fourth is the bottom-left

    // the top-left point will have the smallest sum, whereas
    // the bottom-right point will have the largest sum
    vector<Point> sum = compute(points, [](float a, float b) { return a + b; });

    // now, compute the difference between the points, the
    // top-right point will have the smallest difference,
    // whereas the bottom-left will have the largest difference
    vector<Point> diff = compute(points, [](float a, float b) { return a - b; });

    rect.emplace_back(sum[0]);
    rect.emplace_back(diff[0]);
    rect.emplace_back(sum[1]);
    rect.emplace_back(diff[1]);

    cout << "rect : " << rect << endl;
    // return the ordered coordinates
    return rect;
}

void auto_scan_image() {
    Mat image, orig, gray, edged;
    Mat warp_mat, warp_dst;
    vector<vector<Point>> contours, contours2;
    vector<Point> screenContour, rect;
    Point topLeft, topRight, bottomRight, bottomLeft;
    Point2f pts1[4], pts2[4];

    float w1, w2, h1, h2, maxWidth, maxHeight;

    // load the image and compute the ratio of the old height
    // to the new height, clone it, and resize it
    // document.jpg ~ document7.jpg
    image = imread("../images/document.jpg");
    image.copyTo(orig);
    float r = (float) (800.0 / image.rows);
    Size2f dim = Size2f(image.cols * r, 800);
    resize(image, image, dim, INTER_AREA);

    // convert the image to grayscale, blur it, and find edges
    // in the image
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(3, 3), 0);
    Canny(gray, edged, 75, 200);

    // show the original image and the edge detected image
    cout << "STEP 1: Edge Detection" << endl;
    imshow("Image", image);
    imshow("Edged", edged);

    waitKey(0);
    destroyAllWindows();

    // find the contours in the edged image, keeping only the
    // largest ones, and initialize the screen contour
    findContours(edged, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    sort(contours.rbegin(), contours.rend(), compareContourAreas);

    // loop over the contours
    for ( auto contour : contours ) {
        // approximate the contour
        vector<Point> approx;
        double peri = arcLength(contour, true);
        approxPolyDP(contour, approx, 0.02 * peri, true);

        // if our approximated contour has four points, then we
        // can assume that we have found our screen
        if ( approx.size() == 4 ) {
            screenContour = approx;
            break;
        }
    }

    // show the contour (outline) of the piece of paper
    cout << "STEP 2: Find contours of paper" << endl;
    contours2.emplace_back(screenContour);
    drawContours(image, contours2, -1, Scalar(0, 255, 0), 2);
    imshow("Outline", image);

    waitKey(0);
    destroyAllWindows();

    // apply the four point transform to obtain a top-down
    // view of the original image
    rect = order_points(screenContour);
    topLeft     = rect[0];
    topRight    = rect[1];
    bottomRight = rect[2];
    bottomLeft  = rect[3];

    pts1[0] = rect[0];
    pts1[1] = rect[1];
    pts1[2] = rect[2];
    pts1[3] = rect[3];

    w1 = abs(bottomRight.x - bottomLeft.x);
    w2 = abs(topRight.x - topLeft.x);
    h1 = abs(topRight.y - bottomRight.y);
    h2 = abs(topLeft.y - bottomLeft.y);
    maxWidth = max(w1, w2);
    maxHeight = max(h1, h2);

    cout << w1 << ":" << w2 << ":" << h1 << ":" << h2 << endl;

    pts2[0] = Point2f(0, 0);
    pts2[1] = Point2f(maxWidth - 1, 0);
    pts2[2] = Point2f(maxWidth - 1, maxHeight - 1);
    pts2[3] = Point2f(0, maxHeight - 1);

    warp_mat = getPerspectiveTransform(pts1, pts2);

    warpPerspective(orig, warp_dst, warp_mat, Size2f(maxWidth, maxHeight));

    // show the original and scanned images
    cout << "STEP 3: Apply perspective transform" << endl;
    imshow("Wraped", warp_dst);

    waitKey(0);
    destroyAllWindows();


}



int main() {

//    handle_image();
//    handle_image2();
//    contour();
//    contour_approx();
//    warp_affine();
//    warp_perspective();
//    global_threshold();
//    adaptive_threshold();
    auto_scan_image();
    return 0;
}