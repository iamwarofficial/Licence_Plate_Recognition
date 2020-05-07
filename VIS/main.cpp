#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//Converting from RGB image to GREY image
Mat ConvertToGrey(Mat RGBImg)
{
    Mat grey = Mat::zeros(RGBImg.size(), CV_8UC1);
    for (int i = 0; i < RGBImg.rows; i++)
    {
        for (int j = 0; j < RGBImg.cols * 3; j += 3)
        {
            grey.at<uchar>(i, j / 3) = ((RGBImg.at<uchar>(i, j) + RGBImg.at<uchar>(i, j + 1) + RGBImg.at<uchar>(i, j + 2)) / 3);
        }
    }
    return grey;
}

//Equalize histogram
Mat EqualizeHist(Mat GreyImg)
{
    int count[256] = { 0 };
    float prob[256] = { 0.0 };
    float accprob[256] = { 0.0 };
    int newpixel[256] = { 0 };

    Mat equalize = Mat::zeros(GreyImg.size(), CV_8UC1); //new image

    for (int i = 0; i < GreyImg.rows; i++)//count, adding into array
    {
        for (int j = 0; j < GreyImg.cols; j++)
        {
            count[GreyImg.at<uchar>(i, j)]++;
        }
    }
    for (int k = 0; k < 256; k++)//probablity
    {
        prob[k] = (float)count[k] / (float)(GreyImg.rows * GreyImg.cols);
    }
    accprob[0] = prob[0];
    for (int a = 1; a < 256; a++)//acc probability
    {
        accprob[a] = (prob[a] + accprob[a - 1]);
    }
    for (int i = 0; i < 256; i++)//Multiply to find NEW PIXEL
    {
        newpixel[i] = accprob[i] * 255;
    }
    for (int i = 0; i < GreyImg.rows; i++)//Change Old image to new pixel image
    {
        for (int j = 0; j < GreyImg.cols; j++)
        {
            equalize.at<uchar>(i, j) = newpixel[GreyImg.at<uchar>(i, j)];
        }
    }
    return equalize;
}

//Blur an image
Mat ConvertToBlur(Mat GreyImg, int winsize)
{
    Mat blur = Mat::zeros(GreyImg.size(), CV_8UC1);
    for (int i = winsize; i < (GreyImg.rows - winsize); i++)
    {
        for (int j = winsize; j < (GreyImg.cols - winsize); j++)
        {
            int sum = 0;
            for (int ii = (-winsize); ii <= (+winsize); ii++)
            {
                for (int jj = (-winsize); jj <= (winsize); jj++)
                {
                    sum += GreyImg.at<uchar>(i + ii, j + jj);
                }
            }
            blur.at<uchar>(i, j) = sum / (((winsize * 2) + 1) * ((winsize * 2) + 1));
        }
    }
    return blur;
}

//Edge Detection
Mat EdgeDetection(Mat BlurImg)
{
    Mat edge = Mat::zeros(BlurImg.size(), CV_8UC1);
    double AVGL, AVGR;
    for (int i = 1; i < BlurImg.rows - 1; i++)
    {
        for (int j = 1; j < BlurImg.cols - 1; j++)
        {
            AVGL = (BlurImg.at<uchar>(i - 1, j - 1) + BlurImg.at<uchar>(i, j - 1) + BlurImg.at<uchar>(i + 1, j - 1)) / 3;
            AVGR = (BlurImg.at<uchar>(i - 1, j + 1) + BlurImg.at<uchar>(i, j + 1) + BlurImg.at<uchar>(i + 1, j + 1)) / 3;
            if (abs(AVGL - AVGR) > 50)
            {
                edge.at<uchar>(i, j) = 255;
            }
        }
    }
    return edge;
}

//Dilation process
Mat Dilation(Mat EdgeImg, int winsize)
{
    Mat dilation = Mat::zeros(EdgeImg.size(), CV_8UC1);
    for (int i = winsize; i < (EdgeImg.rows - winsize); i++)
    {
        for (int j = winsize; j < (EdgeImg.cols - winsize); j++)
        {
            for (int ii = (-winsize); ii <= (+winsize); ii++)
            {
                for (int jj = (-winsize); jj <= (+winsize); jj++)
                {
                    if (EdgeImg.at<uchar>(i + ii, j + jj) == 255)
                    {
                        dilation.at<uchar>(i, j) = 255;
                    }
                }
            }

        }
    }
    return dilation;
}

//Remove noises #1
Mat Blob1(Mat GreyImg, Mat DilImg)
{
    Mat Blob;
    Blob = DilImg.clone();

    vector<vector<Point> > contours1;
    vector<Vec4i> hierarchy1;
    findContours(DilImg, contours1, hierarchy1,
        RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

    Mat ccl1 = Mat::zeros(GreyImg.size(), CV_8UC3);

    if (!contours1.empty())
    {
        for (int i = 0; i < contours1.size(); i++)
        {
            Scalar colour((rand() & 255), (rand() & 255),
                (rand() & 255));
            drawContours(ccl1, contours1, i, colour, -1, 8, hierarchy1);
        }
    }

    namedWindow("CCL1", WINDOW_AUTOSIZE);
    imshow("CCL1", ccl1);

    Rect rect_first;
    Mat plate;
    Scalar black = CV_RGB(0, 0, 0);
    int count[256] = { 0 };
    for (size_t i = 0; i < contours1.size(); i++)
    {
        rect_first = boundingRect(contours1[i]);
        if (rect_first.width < 70 || rect_first.width > 200 || rect_first.height < 20 || rect_first.height > 50 || rect_first.x < (GreyImg.rows * 0.1) || rect_first.x >(GreyImg.rows * 0.9) || rect_first.y < (GreyImg.cols * 0.1) || rect_first.y >(GreyImg.cols * 0.9))
        {
            drawContours(Blob, contours1, i, black, -1, 8, hierarchy1);
        }
        else
        {
            plate = GreyImg(rect_first);
        }
    }
    return plate;
}

//Calculates the threshold for the binary value with the OTSU formula
int OTSU(Mat plate)
{
    int count[256] = { 0 };
    float prob[256] = { 0.0 };
    float accprob[256] = { 0.0 };
    float sigma[256] = { 0.0 };;
    float meu[256] = { 0.0 };
    for (int i = 0; i < plate.rows; i++)//count, adding into array
    {
        for (int j = 0; j < plate.cols; j++)
        {
            count[plate.at<uchar>(i, j)]++;
        }
    }
    for (int k = 0; k < 256; k++)//probablity
    {
        prob[k] = (float)count[k] / (float)(plate.rows * plate.cols);
    }
    accprob[0] = prob[0];
    for (int a = 1; a < 256; a++)//acc probability
    {
        accprob[a] = (prob[a] + accprob[a - 1]);
    }


    meu[0] = 0;
    for (int a = 1; a < 256; a++)//acc probability
    {
        meu[a] = (a * prob[a] + meu[a - 1]);
    }



    for (int i = 1; i < 256; i++)//acc probability
    {
        sigma[i] = (pow((meu[255] * accprob[i] - meu[i]), 2)) / (accprob[i] * (1 - accprob[i])); //Formula
    }
    float maxSigma = -1;
    int newsigma = -1;
    for (int t = 0; t < 256; t++)
    {
        if (sigma[t] > maxSigma)
        {
            maxSigma = sigma[t];
            newsigma = t;
        }
    }
    return newsigma + 30;
}

//Converts into black 0, or white 255
Mat ConvertToBinary(Mat plate, int OTSU_th)
{
    Mat binary = Mat::zeros(plate.size(), CV_8UC1);
    for (int i = 0; i < plate.rows; i++)
    {
        for (int j = 0; j < plate.cols; j++)
        {
            if (plate.at<uchar>(i, j) > OTSU_th)
            {
                binary.at<uchar>(i, j) = 255;
            }
        }
    }
    return binary;
}

//Remove noises #2
Mat Blob2(Mat plateImg, Mat plate_bin)
{
    Mat Blob;
    Blob = plate_bin.clone();

    vector<vector<Point> > contours2;
    vector<Vec4i> hierarchy2;
    findContours(plate_bin, contours2, hierarchy2,
        RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

    Mat ccl2 = Mat::zeros(plateImg.size(), CV_8UC3);

    if (!contours2.empty())
    {
        for (int i = 0; i < contours2.size(); i++)
        {
            Scalar colour((rand() & 255), (rand() & 255),
                (rand() & 255));
            drawContours(ccl2, contours2, i, colour, -1, 8, hierarchy2);
        }
    }

    imshow("CCL2", ccl2);

    Rect rect_first;
    Mat plate;
    Scalar black = CV_RGB(0, 0, 0);
    for (size_t a = 0; a < contours2.size(); a++)
    {
        rect_first = boundingRect(contours2[a]);

        if (rect_first.height < 10)
        {
            drawContours(Blob, contours2, a, black, -1, 8, hierarchy2);
        }
        else
        {
            plate = plateImg(rect_first);
            string Final = "Car_Number_Plate" + a;
            imshow(Final, plate);
        }
    }
    return plate;
}

int main()
{
    Mat img = imread("/Users/iamwarofficial/Library/Mobile Documents/com~apple~CloudDocs/Project Files/OpenCV/LPR/VIS/Dataset/Car7.jpg");
    imshow("Test1", img);

    Mat Grey_Img = ConvertToGrey(img);
    imshow("Test2", Grey_Img);

    Mat Equalize_Img = EqualizeHist(Grey_Img);
    
    Mat Blur_Img = ConvertToBlur(Equalize_Img, 1);

    Mat Edge_Img = EdgeDetection(Blur_Img);

    Mat Dilation_Img = Dilation(Edge_Img, 3);

    Mat Plate_Img = Blob1(Grey_Img, Dilation_Img);
    imshow("Test7", Plate_Img);

    int otsu_no = OTSU(Plate_Img);
    Mat Plate_Binary_Img = ConvertToBinary(Plate_Img, 205);
    imshow("Test8", Plate_Binary_Img);

    Mat Final_Img = Blob2(Plate_Img, Plate_Binary_Img);
    waitKey(0);

    return 0;
}
