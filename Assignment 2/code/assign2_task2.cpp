#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

vector<Mat> surf(Mat img1, Mat img2){
    
    Ptr<SURF> detector = SURF::create();

    int minHessian = 400;
    detector->setHessianThreshold(minHessian);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, Mat(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, Mat(), keypoints2, descriptors2 );

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
     
     
    const float ratio_thresh = 0.6f;
    vector<DMatch> good_matches;
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < knn_matches.size(); i++){

        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
            pts2.push_back(keypoints2[knn_matches[i][0].trainIdx].pt);
            pts1.push_back(keypoints1[knn_matches[i][0].queryIdx].pt);
        }
    }
    
    // Convert points to integer
    Mat pts1Mat(pts1), pts2Mat(pts2);
    pts1Mat.convertTo(pts1Mat, CV_32S);
    pts2Mat.convertTo(pts2Mat, CV_32S);
    
    // Fundamental Matrix estimation using the 8-point algorithm
    Mat F_8pt = findFundamentalMat(pts1Mat, pts2Mat, FM_8POINT);
    
    // Fundamental Matrix estimation using the 8-point algorithm with RANSAC
    Mat F_8pt_ransac = findFundamentalMat(pts1Mat, pts2Mat, FM_RANSAC);
    
    // Essential Matrix estimation using the 5-point algorithm with RANSAC
    Mat E, mask;
    E = findEssentialMat(pts1Mat, pts2Mat, 1.0, Point2f(0, 0), RANSAC, 0.999, 1.0, mask);
    
    // Enable epipolar lines to have color
    cvtColor(img1, img1, COLOR_GRAY2BGR);
    cvtColor(img2, img2, COLOR_GRAY2BGR);
    
    // Draw epipolar lines for all three methods
    vector<Vec3f> lines, lines_ransac, lines_5pt;
    computeCorrespondEpilines(pts2Mat, 2, F_8pt, lines);
    computeCorrespondEpilines(pts2Mat, 2, F_8pt_ransac, lines_ransac);
    computeCorrespondEpilines(pts2Mat, 2, E, lines_5pt);
    
    Mat img5 = img1.clone(), img6 = img2.clone(), img7 = img1.clone(), img8 = img2.clone(), img9 = img1.clone(), img10 = img2.clone();
    
    // Select random color and random feature match
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 255);
    uniform_int_distribution<> dis1(0, lines.size());
    uniform_int_distribution<> dis2(0, lines_ransac.size());
    uniform_int_distribution<> dis3(0, lines_5pt.size());
    
    for (int i = 0; i < 20; ++i) {
    	int k = dis3(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines_5pt[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img1.cols, y1 = -(r[2] + r[0] * img1.cols) / r[1];
        line(img9, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img9, pts1[k], 5, color, -1);
    }
    
    // Compute epipolar lines for the right image as well
    computeCorrespondEpilines(pts1Mat, 1, F_8pt, lines);
    computeCorrespondEpilines(pts1Mat, 1, F_8pt_ransac, lines_ransac);
    
    // For the right image
    for (int i = 0; i < 20; ++i) {
    	int k = dis1(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img2.cols, y1 = -(r[2] + r[0] * img2.cols) / r[1];
        line(img6, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img6, pts2[k], 5, color, -1);
    }
    
    for (int i = 0; i < 20; ++i) {
    	int k = dis2(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines_ransac[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img2.cols, y1 = -(r[2] + r[0] * img2.cols) / r[1];
        line(img8, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img8, pts2[k], 5, color, -1);
    }

    vector<Mat> output;
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img1(img5.rows, img5.cols + img6.cols, img6.type());
    
    // Copy img5 to the left side of merged_img
    Mat left_roi1 = merged_img1(Rect(0, 0, img5.cols, img5.rows));
    img5.copyTo(left_roi1);

    // Copy img6 to the right side of merged_img
    Mat right_roi1 = merged_img1(Rect(img5.cols, 0, img6.cols, img6.rows));
    img6.copyTo(right_roi1);

    // Display or save the merged image
    imwrite("epipolar_8point_FUNDAMENTAL_alg_surf.png", merged_img1);
    output.push_back(merged_img1);
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img2(img7.rows, img7.cols + img8.cols, img8.type());
    
    // Copy img7 to the left side of merged_img
    Mat left_roi2 = merged_img2(Rect(0, 0, img7.cols, img7.rows));
    img7.copyTo(left_roi2);

    // Copy img8 to the right side of merged_img
    Mat right_roi2 = merged_img2(Rect(img7.cols, 0, img8.cols, img8.rows));
    img8.copyTo(right_roi2);

    // Display or save the merged image
    imwrite("epipolar_8point_FUNDAMENTAL_RANSAC_alg_surf.png", merged_img2);
    output.push_back(merged_img2);
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img3(img9.rows, img9.cols + img10.cols, img10.type());
    
    // Copy img9 to the left side of merged_img
    Mat left_roi3 = merged_img3(Rect(0, 0, img9.cols, img9.rows));
    img9.copyTo(left_roi3);

    // Copy img10 to the right side of merged_img
    Mat right_roi3 = merged_img3(Rect(img9.cols, 0, img10.cols, img10.rows));
    img10.copyTo(right_roi3);

    // Display or save the merged image
    imwrite("epipolar_5point_ESSENTIAL_RANSAC_alg_surf.png", merged_img3);
    output.push_back(merged_img3);
    
    return output;
}


vector<Mat> sift(Mat img1, Mat img2){

    Ptr<SIFT> detector = SIFT::create();
    detector->setNFeatures(450);
    
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, Mat(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, Mat(), keypoints2, descriptors2 );

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
     
     
    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < knn_matches.size(); i++){

        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
            pts2.push_back(keypoints2[knn_matches[i][0].trainIdx].pt);
            pts1.push_back(keypoints1[knn_matches[i][0].queryIdx].pt);
        }
    }
    
    // Convert points to integer
    Mat pts1Mat(pts1), pts2Mat(pts2);
    pts1Mat.convertTo(pts1Mat, CV_32S);
    pts2Mat.convertTo(pts2Mat, CV_32S);
    
    // Fundamental Matrix estimation using the 8-point algorithm
    Mat F_8pt = findFundamentalMat(pts1Mat, pts2Mat, FM_8POINT);
    
    // Fundamental Matrix estimation using the 8-point algorithm with RANSAC
    Mat F_8pt_ransac = findFundamentalMat(pts1Mat, pts2Mat, FM_RANSAC);
    
    // Essential Matrix estimation using the 5-point algorithm with RANSAC
    Mat E, mask;
    E = findEssentialMat(pts1Mat, pts2Mat, 1.0, Point2f(0,0), RANSAC, 0.999, 1.0, mask);
    
    // Enable epipolar lines to have color
    cvtColor(img1, img1, COLOR_GRAY2BGR);
    cvtColor(img2, img2, COLOR_GRAY2BGR);
    
    // Draw epipolar lines for all three methods
    vector<Vec3f> lines, lines_ransac, lines_5pt;
    computeCorrespondEpilines(pts2Mat, 2, F_8pt, lines);
    computeCorrespondEpilines(pts2Mat, 2, F_8pt_ransac, lines_ransac);
    computeCorrespondEpilines(pts2Mat, 2, E, lines_5pt);
    
    Mat img5 = img1.clone(), img6 = img2.clone(), img7 = img1.clone(), img8 = img2.clone(), img9 = img1.clone(), img10 = img2.clone();
    
    // Select random color for the lines and circles
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 255);
    uniform_int_distribution<> dis1(0, lines.size());
    
    // For the left image
    uniform_int_distribution<> dis2(0, lines_ransac.size());
    
    for (int i = 0; i < 20; ++i) {
    	int k = dis2(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines_ransac[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img1.cols, y1 = -(r[2] + r[0] * img1.cols) / r[1];
        line(img7, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img7, pts1[k], 5, color, -1);
    }
    
    uniform_int_distribution<> dis3(0, lines_5pt.size());
    
    for (int i = 0; i < 20; ++i) {
    	int k = dis3(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines_5pt[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img1.cols, y1 = -(r[2] + r[0] * img1.cols) / r[1];
        line(img9, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img9, pts1[k], 5, color, -1);
    }
    
    // Compute epipolar lines for the right image as well
    computeCorrespondEpilines(pts1Mat, 1, F_8pt, lines);
    computeCorrespondEpilines(pts1Mat, 1, F_8pt_ransac, lines_ransac);
    
    // For the right image
    for (int i = 0; i < 20; ++i) {
    	int k = dis1(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img2.cols, y1 = -(r[2] + r[0] * img2.cols) / r[1];
        line(img6, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img6, pts2[k], 5, color, -1);
    }
    
    vector<Mat> output;
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img1(img5.rows, img5.cols + img6.cols, img6.type());
    
    // Copy img5 to the left side of merged_img
    Mat left_roi1 = merged_img1(Rect(0, 0, img5.cols, img5.rows));
    img5.copyTo(left_roi1);

    // Copy img6 to the right side of merged_img
    Mat right_roi1 = merged_img1(Rect(img5.cols, 0, img6.cols, img6.rows));
    img6.copyTo(right_roi1);

    // Display or save the merged image
    imwrite("epipolar_8point_FUNDAMENTAL_alg_sift.png", merged_img1);
    output.push_back(merged_img1);
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img2(img7.rows, img7.cols + img8.cols, img8.type());
    
    // Copy img7 to the left side of merged_img
    Mat left_roi2 = merged_img2(Rect(0, 0, img7.cols, img7.rows));
    img7.copyTo(left_roi2);

    // Copy img8 to the right side of merged_img
    Mat right_roi2 = merged_img2(Rect(img7.cols, 0, img8.cols, img8.rows));
    img8.copyTo(right_roi2);

    // Display or save the merged image
    imwrite("epipolar_8point_FUNDAMENTAL_RANSAC_alg_sift.png", merged_img2);
    output.push_back(merged_img2);
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img3(img9.rows, img9.cols + img10.cols, img10.type());
    
    // Copy img9 to the left side of merged_img
    Mat left_roi3 = merged_img3(Rect(0, 0, img9.cols, img9.rows));
    img9.copyTo(left_roi3);

    // Copy img10 to the right side of merged_img
    Mat right_roi3 = merged_img3(Rect(img9.cols, 0, img10.cols, img10.rows));
    img10.copyTo(right_roi3);

    // Display or save the merged image
    imwrite("epipolar_5point_ESSENTIAL_RANSAC_alg_sift.png", merged_img3);
    output.push_back(merged_img3);
    
    return output;
}

vector<Mat> orb(Mat img1, Mat img2){

    Ptr<ORB> detector = ORB::create();
    detector->setMaxFeatures(500);
    detector->setEdgeThreshold(1);
    
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, Mat(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, Mat(), keypoints2, descriptors2 );

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
     
     
    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < knn_matches.size(); i++){

        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
            pts2.push_back(keypoints2[knn_matches[i][0].trainIdx].pt);
            pts1.push_back(keypoints1[knn_matches[i][0].queryIdx].pt);
        }
    }
    
    // Convert points to integer
    Mat pts1Mat(pts1), pts2Mat(pts2);
    pts1Mat.convertTo(pts1Mat, CV_32S);
    pts2Mat.convertTo(pts2Mat, CV_32S);
    
    // Fundamental Matrix estimation using the 8-point algorithm
    Mat F_8pt = findFundamentalMat(pts1Mat, pts2Mat, FM_8POINT);
    
    // Fundamental Matrix estimation using the 8-point algorithm with RANSAC
    Mat F_8pt_ransac = findFundamentalMat(pts1Mat, pts2Mat, FM_RANSAC);
    
    // Essential Matrix estimation using the 5-point algorithm with RANSAC
    Mat E, mask;
    E = findEssentialMat(pts1Mat, pts2Mat, 1.0, Point2f(0,0), RANSAC, 0.999, 1.0, mask);
    
    // Enable epipolar lines to have color
    cvtColor(img1, img1, COLOR_GRAY2BGR);
    cvtColor(img2, img2, COLOR_GRAY2BGR);
    
    // Draw epipolar lines for all three methods
    vector<Vec3f> lines, lines_ransac, lines_5pt;
    computeCorrespondEpilines(pts2Mat, 2, F_8pt, lines);
    computeCorrespondEpilines(pts2Mat, 2, F_8pt_ransac, lines_ransac);
    computeCorrespondEpilines(pts2Mat, 2, E, lines_5pt);
    
    Mat img5 = img1.clone(), img6 = img2.clone(), img7 = img1.clone(), img8 = img2.clone(), img9 = img1.clone(), img10 = img2.clone();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 255);
    uniform_int_distribution<> dis1(0, lines.size());
    uniform_int_distribution<> dis2(0, lines_ransac.size());
    uniform_int_distribution<> dis3(0, lines_5pt.size());
    
    // For the left image
    for (int i = 0; i < 20; ++i) {
    	int k = dis1(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img1.cols, y1 = -(r[2] + r[0] * img1.cols) / r[1];
        line(img5, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img5, pts1[k], 5, color, -1);
    }
    
    for (int i = 0; i < 20; ++i) {
    	int k = dis2(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines_ransac[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img1.cols, y1 = -(r[2] + r[0] * img1.cols) / r[1];
        line(img7, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img7, pts1[k], 5, color, -1);
    }
    
    for (int i = 0; i < 20; ++i) {
    	int k = dis3(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines_5pt[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img1.cols, y1 = -(r[2] + r[0] * img1.cols) / r[1];
        line(img9, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img9, pts1[k], 5, color, -1);
    }

    vector<Mat> output;
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img1(img5.rows, img5.cols + img6.cols, img6.type());
    
    // Copy img5 to the left side of merged_img
    Mat left_roi1 = merged_img1(Rect(0, 0, img5.cols, img5.rows));
    img5.copyTo(left_roi1);

    // Copy img6 to the right side of merged_img
    Mat right_roi1 = merged_img1(Rect(img5.cols, 0, img6.cols, img6.rows));
    img6.copyTo(right_roi1);

    // Display or save the merged image
    imwrite("epipolar_8point_FUNDAMENTAL_alg_orb.png", merged_img1);
    output.push_back(merged_img1);
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img2(img7.rows, img7.cols + img8.cols, img8.type());
    
    // Copy img5 to the left side of merged_img
    Mat left_roi2 = merged_img2(Rect(0, 0, img7.cols, img7.rows));
    img7.copyTo(left_roi2);

    // Copy img6 to the right side of merged_img
    Mat right_roi2 = merged_img2(Rect(img7.cols, 0, img8.cols, img8.rows));
    img8.copyTo(right_roi2);

    // Display or save the merged image
    imwrite("epipolar_8point_FUNDAMENTAL_RANSAC_alg_orb.png", merged_img2);
    output.push_back(merged_img2);
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img3(img9.rows, img9.cols + img10.cols, img10.type());
    
    // Copy img5 to the left side of merged_img
    Mat left_roi3 = merged_img3(Rect(0, 0, img9.cols, img9.rows));
    img9.copyTo(left_roi3);

    // Copy img6 to the right side of merged_img
    Mat right_roi3 = merged_img3(Rect(img9.cols, 0, img10.cols, img10.rows));
    img10.copyTo(right_roi3);

    // Display or save the merged image
    imwrite("epipolar_5point_ESSENTIAL_RANSAC_alg_orb.png", merged_img3);
    output.push_back(merged_img3);
    
    return output;
}

vector<Mat> fast_brief(Mat img1, Mat img2){

    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    Ptr<BriefDescriptorExtractor> descriptor = BriefDescriptorExtractor::create();

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
     
     
    const float ratio_thresh = 0.6f;
    vector<DMatch> good_matches;
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < knn_matches.size(); i++){

        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
            pts2.push_back(keypoints2[knn_matches[i][0].trainIdx].pt);
            pts1.push_back(keypoints1[knn_matches[i][0].queryIdx].pt);
        }
    }
    
    // Convert points to integer
    Mat pts1Mat(pts1), pts2Mat(pts2);
    pts1Mat.convertTo(pts1Mat, CV_32S);
    pts2Mat.convertTo(pts2Mat, CV_32S);
    
    // Fundamental Matrix estimation using the 8-point algorithm
    Mat F_8pt = findFundamentalMat(pts1Mat, pts2Mat, FM_8POINT);
    
    // Fundamental Matrix estimation using the 8-point algorithm with RANSAC
    Mat F_8pt_ransac = findFundamentalMat(pts1Mat, pts2Mat, FM_RANSAC);
    
    // Essential Matrix estimation using the 5-point algorithm with RANSAC
    Mat E, mask;
    E = findEssentialMat(pts1Mat, pts2Mat, Mat::eye(3, 3, CV_64F), RANSAC);
    
    // Enable epipolar lines to have color
    cvtColor(img1, img1, COLOR_GRAY2BGR);
    cvtColor(img2, img2, COLOR_GRAY2BGR);
    
    // Draw epipolar lines for all three methods
    vector<Vec3f> lines, lines_ransac, lines_5pt;
    computeCorrespondEpilines(pts2Mat, 2, F_8pt, lines);
    computeCorrespondEpilines(pts2Mat, 2, F_8pt_ransac, lines_ransac);
    computeCorrespondEpilines(pts2Mat, 2, E, lines_5pt);
    
    Mat img5 = img1.clone(), img6 = img2.clone(), img7 = img1.clone(), img8 = img2.clone(), img9 = img1.clone(), img10 = img2.clone();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 255);
    uniform_int_distribution<> dis1(0, lines.size());

    uniform_int_distribution<> dis3(0, lines_5pt.size());
    
    // For the left image
    for (int i = 0; i < 20; ++i) {
    	int k = dis1(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img1.cols, y1 = -(r[2] + r[0] * img1.cols) / r[1];
        line(img5, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img5, pts1[k], 5, color, -1);
    }
    
    computeCorrespondEpilines(pts1Mat, 1, F_8pt_ransac, lines_ransac);
    uniform_int_distribution<> dis2(0, lines_ransac.size());
    
    for (int i = 0; i < 20; ++i) {
    	int k = dis3(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines_5pt[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img1.cols, y1 = -(r[2] + r[0] * img1.cols) / r[1];
        line(img9, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img9, pts1[k], 5, color, -1);
    }
    
    for (int i = 0; i < 20; ++i) {
    	int k = dis2(gen);
    	Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        Vec3f r = lines_ransac[k];
        float x0 = 0, y0 = -r[2] / r[1];
        float x1 = img2.cols, y1 = -(r[2] + r[0] * img2.cols) / r[1];
        line(img8, Point(x0, y0), Point(x1, y1), color, 1);
        circle(img8, pts2[k], 5, color, -1);
    }
    
    vector<Mat> output;
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img1(img5.rows, img5.cols + img6.cols, img6.type());
    
    // Copy img5 to the left side of merged_img
    Mat left_roi1 = merged_img1(Rect(0, 0, img5.cols, img5.rows));
    img5.copyTo(left_roi1);

    // Copy img6 to the right side of merged_img
    Mat right_roi1 = merged_img1(Rect(img5.cols, 0, img6.cols, img6.rows));
    img6.copyTo(right_roi1);

    // Display or save the merged image
    imwrite("epipolar_8point_FUNDAMENTAL_alg_fast_brief.png", merged_img1);
    output.push_back(merged_img1);
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img2(img7.rows, img7.cols + img8.cols, img8.type());
    
    // Copy img5 to the left side of merged_img
    Mat left_roi2 = merged_img2(Rect(0, 0, img7.cols, img7.rows));
    img7.copyTo(left_roi2);

    // Copy img6 to the right side of merged_img
    Mat right_roi2 = merged_img2(Rect(img7.cols, 0, img8.cols, img8.rows));
    img8.copyTo(right_roi2);

    // Display or save the merged image
    imwrite("epipolar_8point_FUNDAMENTAL_RANSAC_alg_fast_brief.png", merged_img2);
    output.push_back(merged_img2);
    
    // Merge the left and right image into one pair-----------------------------------------------------------
    Mat merged_img3(img9.rows, img9.cols + img10.cols, img10.type());
    
    // Copy img5 to the left side of merged_img
    Mat left_roi3 = merged_img3(Rect(0, 0, img9.cols, img9.rows));
    img9.copyTo(left_roi3);

    // Copy img6 to the right side of merged_img
    Mat right_roi3 = merged_img3(Rect(img9.cols, 0, img10.cols, img10.rows));
    img10.copyTo(right_roi3);

    // Display or save the merged image
    imwrite("epipolar_5point_ESSENTIAL_RANSAC_alg_fast_brief.png", merged_img3);
    output.push_back(merged_img3);
    
    return output;
}


int main( int argc, char** argv ){
    
    Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
    Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );
    vector<Mat> output;

    if(strcmp(argv[3], "surf") == 0){
        output = surf(img_1, img_2);
        for(int i = 0; i<3; i++) { 
           if (i == 0) imshow("epipolar_8point_FUNDAMENTAL_alg_surf", output.at(i));
           if (i == 1) imshow("epipolar_8point_FUNDAMENTAL_RANSAC_alg_surf", output.at(i));
           if (i == 2) imshow("epipolar_5point_ESSENTIAL_RANSAC_alg_surf", output.at(i));
        }

    }else if(strcmp(argv[3], "sift") == 0){
        output = sift(img_1, img_2);
        for(int i = 0; i<3; i++) { 
           if (i == 0) imshow("epipolar_8point_FUNDAMENTAL_alg_sift", output.at(i));
           if (i == 1) imshow("epipolar_8point_FUNDAMENTAL_RANSAC_alg_sift", output.at(i));
           if (i == 2) imshow("epipolar_5point_ESSENTIAL_RANSAC_alg_sift", output.at(i));
        }

    }else if(strcmp(argv[3], "orb") == 0){
        output = orb(img_1, img_2);
        for(int i = 0; i<3; i++) { 
           if (i == 0) imshow("epipolar_8point_FUNDAMENTAL_alg_orb", output.at(i));
           if (i == 1) imshow("epipolar_8point_FUNDAMENTAL_RANSAC_alg_orb", output.at(i));
           if (i == 2) imshow("epipolar_5point_ESSENTIAL_RANSAC_alg_orb", output.at(i));
        }

    } else if (strcmp(argv[3], "fast+brief") == 0){
        output = fast_brief(img_1, img_2);
        for(int i = 0; i<3; i++) { 
           if (i == 0) imshow("epipolar_8point_FUNDAMENTAL_alg_fast_brief", output.at(i));
           if (i == 1) imshow("epipolar_8point_FUNDAMENTAL_RANSAC_alg_fast_brief", output.at(i));
           if (i == 2) imshow("epipolar_5point_ESSENTIAL_RANSAC_alg_fast_brief", output.at(i));
        }
    }

    waitKey();
 
    return 0;
}
