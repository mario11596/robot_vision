#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


Mat surf(Mat img1, Mat img2){
    
    Ptr<SURF> detector = SURF::create();

    int minHessian = 400;
    detector->setHessianThreshold(minHessian);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, Mat(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, Mat(), keypoints2, descriptors2 );


    Mat img_keypoints1, img_keypoints2;
    drawKeypoints( img1, keypoints1, img_keypoints1,Scalar::all(-1) );
    imwrite("key_points_img1_surf.png", img_keypoints1);

    drawKeypoints( img2, keypoints2, img_keypoints2, Scalar::all(-1));
    imwrite("key_points_img2_surf.png", img_keypoints2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
     
     
    const float ratio_thresh = 0.6f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){

        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imwrite("feature_matching_surf.png", img_matches);

    return img_matches;
}


Mat sift(Mat img1, Mat img2){

    Ptr<SIFT> detector = SIFT::create();
    detector->setNFeatures(450);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, Mat(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, Mat(), keypoints2, descriptors2 );

    Mat img_keypoints1, img_keypoints2;
    drawKeypoints( img1, keypoints1, img_keypoints1,Scalar::all(-1) );
    imwrite("key_points_img1_sift.png", img_keypoints1);

    drawKeypoints( img2, keypoints2, img_keypoints2, Scalar::all(-1));
    imwrite("key_points_img2_sift.png", img_keypoints2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
     
     
    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){

        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
        }
    }
     
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imwrite("feature_matching_sift.png", img_matches);

    return img_matches;
}


Mat orb(Mat img1, Mat img2){

    Ptr<ORB> detector = ORB::create();
    detector->setMaxFeatures(500);
    
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, Mat(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, Mat(), keypoints2, descriptors2 );

    Mat img_keypoints1, img_keypoints2;
    drawKeypoints( img1, keypoints1, img_keypoints1, Scalar::all(-1));
    imwrite("key_points_img1_orb.png", img_keypoints1);

    drawKeypoints( img2, keypoints2, img_keypoints2, Scalar::all(-1));
    imwrite("key_points_img2_orb.png", img_keypoints2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
     
     
    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){

        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
        }
    }
     
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imwrite("feature_matching_orb.png", img_matches);

    return img_matches;
}


Mat fast_brief(Mat img1, Mat img2){

    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    Ptr<BriefDescriptorExtractor> descriptor = BriefDescriptorExtractor::create();

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    Mat img_keypoints1, img_keypoints2;
    drawKeypoints( img1, keypoints1, img_keypoints1, Scalar::all(-1));
    imwrite("key_points_img1_fast_brief.png", img_keypoints1);

    drawKeypoints( img2, keypoints2, img_keypoints2, Scalar::all(-1));
    imwrite("key_points_img2_fast_brief.png", img_keypoints2);
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
     
    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){

        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
        }
    }
     
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imwrite("feature_matching_fast_brief.png", img_matches);
    
    return img_matches;
}


int main( int argc, char** argv ){
    
    Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
    Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );
    Mat output;

    if(strcmp(argv[3], "surf") == 0){
        output = surf(img_1, img_2);

    }else if(strcmp(argv[3], "sift") == 0){
        output = sift(img_1, img_2);

    }else if(strcmp(argv[3], "orb") == 0){
        output = orb(img_1, img_2);

    } else if (strcmp(argv[3], "fast+brief") == 0){
        output = fast_brief(img_1, img_2);
    }

    imshow("Feature detection and matching", output);
    waitKey();
 
    return 0;
}
