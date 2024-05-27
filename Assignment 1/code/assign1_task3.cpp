#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/param.h>
#include <unistd.h>
#include <fstream>
#include <regex>
#include <vector>


using namespace cv;
using namespace std;


static void saveXYZ(const char* filename, const Mat& mat, const Mat& colors)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");

    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", mat.rows*mat.cols);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");

    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;

            Vec3b color = colors.at<Vec3b>(y, x);
            fprintf(fp, "%f %f %f %d %d %d\n", point[0], point[1], point[2], color[2], color[1], color[0]);        
        }
    }
    fclose(fp);
}


Mat read_pfm(const std::string& file_path) {
    ifstream file(file_path, std::ios::binary);

    bool color;
    int width;
    int height;
    float scale;
    char endian;
    int colorChannel;
    string header;

    getline(file, header);

    if (header == "PF") {
        color = true;

    } else if (header == "Pf") {
        color = false;
    }

    string line;
    getline(file, line);
    regex dim_regex(R"((\d+)\s(\d+))");
    smatch dim_match;

    if (regex_match(line, dim_match, dim_regex)) {
        width = stoi(dim_match[1]);
        height = stoi(dim_match[2]);
    }

    getline(file, line);
    scale = stof(line);
    if (scale < 0) {
        endian = '<';
        scale = -scale;
    } else {
        endian = '>';
    }

    if(color){
        colorChannel = 3;
    } else {
        colorChannel = 1;
    }
   
    vector<float> data(width * height * colorChannel);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));

    Mat mat;
    if (color) {
        mat = Mat(height, width, CV_32FC3, data.data()).clone();
    } else {
        mat = Mat(height, width, CV_32FC1, data.data()).clone();
    }

    flip(mat, mat, 0);
 
    return mat;
}


int main(int argc, char** argv)
{
    std::string img1_filename = "";
    std::string img2_filename = "";
    std::string intrinsic_filename = "";
    std::string extrinsic_filename = "";
    std::string disparity_filename = "";
    std::string point_cloud_filename = "";

    float scale = 1.0;

    cv::CommandLineParser parser(argc, argv,
        "{@arg1||}{@arg2||}{help h||}{i||}{e||}{p||}");
  
    img1_filename = samples::findFile(parser.get<std::string>(0));
    img2_filename = samples::findFile(parser.get<std::string>(1));
   
    if( parser.has("i") )
        intrinsic_filename = parser.get<std::string>("i");
    if( parser.has("e") )
        extrinsic_filename = parser.get<std::string>("e");
    if( parser.has("p") )
        point_cloud_filename = parser.get<std::string>("p");

    
    if( img1_filename.empty() || img2_filename.empty() )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }

    if( (!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()) )
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }

    if( extrinsic_filename.empty() && !point_cloud_filename.empty() )
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }

    int color_mode = -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    if (img1.empty())
    {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }

    if (img2.empty())
    {
        printf("Command-line parameter error: could not load the second input image file\n");
        return -1;
    }


    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    if( !intrinsic_filename.empty() )
    {
        
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            return -1;
        }

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;
      
        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, img_size, &roi1, &roi2);

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;

        
        imwrite("imgLeft.png", img1);
        imwrite("imgRight.png", img2);

        imwrite("./unimatch/datasets/middlebury/imgLeft.png", img1);
        imwrite("./unimatch/datasets/middlebury/imgRight.png", img2);
    }

    
    string scriptDirectory = "./unimatch/scripts/";
    chdir(scriptDirectory.c_str());
    system("./unimatch.sh");

    sleep(5);

    scriptDirectory = "../../";
    chdir(scriptDirectory.c_str());


    Mat disp;
    disparity_filename = samples::findFile("./unimatch/output/imgLeft_disp.pfm");

    
    if(disparity_filename.empty()){
        printf("Disparity image is empty! \n");

        return -1;
    } else {
        disp = read_pfm(disparity_filename);
    }


    if(!point_cloud_filename.empty())
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;

        reprojectImageTo3D(disp, xyz, Q, false);
        saveXYZ(point_cloud_filename.c_str(), xyz, img1);
        printf("\n");
    }

    printf("Process is done!");
    fflush(stdout);
    printf("\n");
    
    return 0;
}
