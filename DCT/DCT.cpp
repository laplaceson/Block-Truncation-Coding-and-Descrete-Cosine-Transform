#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
#include <omp.h>
using namespace cv;
using namespace std;
float cosine(int x, int i, float N_1)
{
	return cos((x + 0.5)*i*CV_PI* N_1);
}
Mat DCT_2D(Mat image)
{
	Mat outpic(image.rows, image.cols, CV_32FC1, Scalar(0));
	float N_1 = pow(image.rows*image.cols, -0.5);
	float C = pow(2.0, -0.5);
#pragma omp parallel for
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			float tmpadd = 0;
			for (int x = 0; x < image.rows; ++x)
			{
				for (int y = 0; y < image.cols; ++y)
				{
					tmpadd += image.ptr<uchar>(x)[y] * cosine(x, i, N_1) * cosine(y, j, N_1);
				}
			}
			if (i == 0) tmpadd *= C;
			if (j == 0) tmpadd *= C;
			outpic.ptr<float>(i)[j] = tmpadd * 2 * N_1;
		}
	}
	//outpic.convertTo(outpic, CV_8UC1);
	return outpic;
}
Mat IDCT_2D(Mat image)
{
	Mat outpic(image.rows, image.cols, CV_8UC1, Scalar(0));
	float N_1 = pow(image.rows*image.cols, -0.5);
	float C = pow(2.0, -0.5);
#pragma omp parallel for
	for (int x = 0; x < image.rows; ++x)
	{
		for (int y = 0; y < image.cols; ++y)
		{
			float tmpadd = 0;
			for (int i = 0; i < image.rows; ++i)
			{
				for (int j = 0; j < image.cols; ++j)
				{
					float tmp = 1.0;
					if (i == 0) tmp *= C;
					if (j == 0) tmp *= C;
					tmpadd += image.ptr<float>(i)[j] * cosine(x, i, N_1) * cosine(y, j, N_1)*tmp;
				}
			}
			outpic.ptr<uchar>(x)[y] = tmpadd * 2 * N_1;
		}
	}
	return outpic;
}
int main()
{
	Mat image = imread("images.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(image, image, Size(60, 60)); 	
	double START = clock();
	Mat outpic = DCT_2D(image); 
	Mat outpic2 = IDCT_2D(outpic);
	
	cout << "Total execute times:" << (clock() - START) / CLOCKS_PER_SEC << " sec" << endl;
	system("pause");
}
