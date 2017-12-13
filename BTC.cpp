#include <omp.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace cv;
using namespace std;
int main()
{
	Mat image = imread("images.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(image, image, Size(600, 600));
	double START = clock();
	int B_sizeX = 10, B_sizeY = 10;
	int  B_SIZE = B_sizeX * B_sizeY;
	float B_XY = 1.0 / B_SIZE;
	int B_imageX = image.rows / B_sizeX, B_imageY = image.cols / B_sizeY;
	Mat BTCout(image.rows, image.cols, CV_8UC1, Scalar(0));
	int cnti = 0, cntj = 0;
	#pragma omp parallel for
	for (int i = 0; i < image.rows; i += B_sizeX)
	{
		for (int j = 0; j < image.cols; j += B_sizeY)
		{
			float h = 0; float h_2 = 0;
			for (int m = 0; m < B_sizeX; ++m)
			{
				for (int n = 0; n < B_sizeY; ++n)
				{
					int tmp = image.ptr<uchar>(i + m)[j + n];
					h += tmp;
					h_2 += tmp*tmp;
				}
			}
			h *= B_XY; h_2 *= B_XY;
			float sigma = sqrt(abs(h*h - h_2));
			float Q = 0;
			for (int m = 0; m < B_sizeX; ++m)
			{
				for (int n = 0; n < B_sizeY; ++n)
				{
					if (image.ptr<uchar>(i + m)[j + n] >= h) ++Q;
				}
			}
			float A = h - sigma * sqrt(Q / (B_SIZE - Q));
			float B = h + sigma * sqrt((B_SIZE - Q) / Q);
			for (int m = 0; m < B_sizeX; ++m)
			{
				for (int n = 0; n < B_sizeY; ++n)
				{
					BTCout.ptr<uchar>(i + m)[j + n] = image.ptr<uchar>(i + m)[j + n] >= h ? B : A;
				}
			}
			++cntj;
		}
		++cnti;
	}
	cout << "Total execute times:" << (clock() - START) / CLOCKS_PER_SEC << " sec" << endl;
	system("pause");
}
