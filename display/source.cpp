#define TEST
#define MODE_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <chrono>
#include <limits.h>

#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;
using namespace std;

#define PI 3.14159265359

typedef struct {
	int r, g, b;
}int_rgb;

class ErrorInfo
{
public:
	int BlockX;
	int BlockY;
	int Error;

	ErrorInfo();
	ErrorInfo(int block_x, int block_y, int error);
};

ErrorInfo::ErrorInfo()
{
	BlockX = 0;
	BlockY = 0;
	Error = 0;
}

ErrorInfo::ErrorInfo(int block_x, int block_y, int error)
{
	BlockX = block_x;
	BlockY = block_y;
	Error = error;
}

float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i<height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j<img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}

void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}

int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

// First Image Processing
void Ex0305() {
	/** Image Pointer */
	int** image;

	/** width, height of image */
	int height, width;

	// Initialize
	image = ReadImage("koala.jpg", &height, &width);

	// Calculate
	for (int x = 0, y = 0; x < width; x++)
	{
		// the equation of y for x
		y = (int)(0.4 * x + 100 + 0.5);

		// check index out of range
		if (y < 0 || y > height - 1) continue;
		else image[y][x] = 255;
	}

	// Show result
	ImageShow("test", image, height, width);
}

/** 2*2 Matrix
* @argument a (Row 1, Column 1)
* @argument b (Row 1, Column 2)
* @argument c (Row 2, Column 1)
* @argument d (Row 2, Column 2)
*/
typedef struct {
	double a, b;
	double c, d;
}Matrix2X2;

Matrix2X2 GetInverseMatrix(double a, double b, double c, double d)
{
	double determinant = a*d - b*c;
	Matrix2X2 MatrixOut = { d / determinant, -b / determinant, -c / determinant, a / determinant };

	return MatrixOut;
}

Matrix2X2 GetInverseMatrix(Matrix2X2 matrix)
{
	return GetInverseMatrix(matrix.a, matrix.b, matrix.c, matrix.d);
}

double inline GetLength(Point3d p, Point3d q = Point3d(0, 0, 0))
{
	return sqrtl(powl(p.x - q.x, 2) + powl(p.y - q.y, 2) + powl(p.z - q.z, 2));
}

/** Affine transform function
* @param Input (x, y) coordinate
* @param a,b,c,d,t1,t2,x1,y2 affine transform arguments
*/
Point2d Affine(Point2d Input, double a, double b, double c, double d, double t1 = 0, double t2 = 0, double x1 = 0, double y1 = 0)
{
	double NewX = a*(Input.x - x1) + b*(Input.y - y1) + t1;
	double NewY = c*(Input.x - x1) + d*(Input.y - y1) + t2;

	return Point2d(NewX, NewY);
}

/** Affine transform function
* @param Input (x, y) coordinate
* @param matrix includes a, b, c, d arguments
* @param a,b,c,d,t1,t2,x1,y2 affine transform arguments
*/
Point2d Affine(Point2d Input, Matrix2X2 matrix, double t1 = 0, double t2 = 0, double x1 = 0, double y1 = 0)
{
	double NewX = matrix.a * (Input.x - x1) + matrix.b * (Input.y - y1) + t1;
	double NewY = matrix.c * (Input.x - x1) + matrix.d * (Input.y - y1) + t2;

	return Point2d(NewX, NewY);
}

/** Affine transform function
* @param Input (x, y, w) coordinate
* @param a,b,t1 NewX = a*(Input.x - x1) + b*(Input.y - y1) + t1
* @param c,d,t2 NewY = c*(Input.x - x1) + d*(Input.y - y1) + t2
*/
Point3d Affine(Point3d Input, double a, double b, double c, double d, double t1 = 0, double t2 = 0, double x1 = 0, double y1 = 0)
{
	double NewX = a*(Input.x - x1) + b*(Input.y - y1) + t1;
	double NewY = c*(Input.x - x1) + d*(Input.y - y1) + t2;
	double NewW = 1;

	return Point3d(NewX, NewY, NewW);
}

/** Affine transform function
* @param Input (x, y, w) coordinate
* @param a,b,c,t1,x1 NewX = a*(Input.x - x1) + b*(Input.y - y1) + c*(Input.z - z1) + t1
* @param d,e,f,t2,y1 NewY = d*(Input.x - x1) + e*(Input.y - y1) + f*(Input.z - z1) + t2
* @param g,h,i,t3,z1 NewZ = g*(Input.x - x1) + h*(Input.y - y1) + i*(Input.z - z1) + t3
*/
Point3d Affine(Point3d Input, double a, double b, double c, double d, double e, double f, double g, double h, double i, double t1 = 0, double t2 = 0, double t3 = 0, double x1 = 0, double y1 = 0, double z1 = 0)
{
	double NewX = a*(Input.x - x1) + b*(Input.y - y1) + c*(Input.z - z1) + t1;
	double NewY = d*(Input.x - x1) + e*(Input.y - y1) + f*(Input.z - z1) + t2;
	double NewZ = g*(Input.x - x1) + h*(Input.y - y1) + i*(Input.z - z1) + t3;

	return Point3d(NewX, NewY, NewZ);
}

/** Draw a line
* @param Image Image file that stores color of each pixel
* @param Height height of image
* @parma Width width of image
* @param a Inclination of line
* @param b Constant added to formula
* @param Thickness Thickness of line
* @param brightness Brightness of line
*
* @formula --> y = ax + b --> ax - y + b = 0
*          --> d = |ax0 - y0 + b| / sqrt(a*a + 1)
*/
int** DrawLine(int** Image, int Height, int Width, double a, double b, double Thickness, uint8_t brightness) {
	int** ImageOut = IntAlloc2(Height, Width);

	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
		{
			double d = fabs(a*x - y + b) / sqrt(a*a + 1.0);

			if (d < Thickness) Image[y][x] = brightness;
		}

	return ImageOut;
}

/** Draw a circle filled with
* @param Image Image file that stores color of each pixel
* @param Height height of image
* @parma Width width of image
* @param a X coordinate of center of circle
* @param b Y coordinate of center of circle
* @param r radius of circle
* @param brightness Brightness of circle
* @formula --> (x - a)^2 + (y - b)^2 = r^2
*/
int** DrawCircle(int** Image, int Height, int Width, double a, double b, double r, uint8_t brightness) {
	int** ImageOut = IntAlloc2(Height, Width);

	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
		{
			if (powl(x - a, 2) + powl(y - b, 2) <= powl(r, 2))
				ImageOut[y][x] = brightness;
			else
				ImageOut[y][x] = Image[y][x];
		}

	return ImageOut;
}

/** Apply bilinear interpolation on image after affine transform
* @param Image image to apply bilinear interpolation on
* @param Height height of image
* @param Width width of image
* @param a, b, c, d arguments of invertible matrix
*/
void BilinearInterpolation(int** ImageOut, int** Image, int Height, int Width, double a, double b, double c, double d, double t1 = 0, double t2 = 0, double x1 = 0, double y1 = 0)
{
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
		{
			Point2d InversedPoint = Affine(Point2d(x, y), GetInverseMatrix(a, b, c, d), t1, t2, x1, y1);

			int LT_X = (int)InversedPoint.x;	int RT_X = LT_X + 1;
			int LT_Y = (int)InversedPoint.y;	int RT_Y = LT_Y;
			int LB_X = LT_X;					int RB_X = LT_X + 1;
			int LB_Y = LT_Y + 1;				int RB_Y = LT_Y + 1;

			// pass sequence if image processcing area isn't inside of image size
			if (LT_X < 0 || LT_Y < 0 || RB_X > Width - 1 || RB_Y > Height - 1)
				continue;

			double dx = InversedPoint.x - LT_X;
			double dy = InversedPoint.y - LT_Y;

			int A = Image[LT_Y][LT_X];			int B = Image[RT_Y][RT_X];
			int C = Image[LB_Y][LB_X];			int D = Image[RB_Y][RB_X];

			double value =
				A * (1.0 - dx) * (1.0 - dy) +
				B * dx * (1.0 - dy) +
				C * (1.0 - dx) * dy +
				D * dx * dy;

			ImageOut[y][x] = (int)(value + 0.5);
		}
}

/** Rotate image clockwise and apply bilinear interpolation on image
* @param Image image to rotate
* @param Height height of image
* @param Width width of image
* @param Angle rotation angle
* @param OriginY y coordinate of origin for rotation
* @param OriginX x coordinate of origin for rotation
*/
void RotationTransform(int** ImageOut, int** Image, double Height, double Width, double Angle, double OriginY = 0, double OriginX = 0)
{
	// transform radian to degree
	Angle /= 57.2958;

	BilinearInterpolation(ImageOut, Image, Height, Width, cos(Angle), -sin(Angle), sin(Angle), cos(Angle), OriginX, OriginY, OriginX, OriginY);
}

/** Transform 3D line to 2D line & Show transformed image
* @param Image image which line is projected to
* @param Height height of image to show
* @param Width width of image to show
* @param p starting point of line
* @param q ending point of line
* @param DotNumber number of dot expressing line (the more this parameter is bigger, the more result looks like line)
* @param PlaneDistance distance between camera and image plane that projects line (the more this parameter is bigger, the more line is enlarged)
* @param brightness brightness of dot marked on image
* @description refer to resource file: "Pinhole Camera.pptx"
*/
int** PinholeLine(int** Image, double Height, double Width, Point3d p, Point3d q, int DotNumber, double PlaneDistance = 300, uint8_t brightness = 255)
{
	int** ImageOut = Image;

	p *= -1;
	q *= -1;

	for (double t = 0; t < 1; t += (double)1 / DotNumber)
	{
		Point3d point_target = p + t*(q - p);
		if (point_target.z == 0) point_target.z = 0.0001;

		Point3d point_projected = Affine(point_target, PlaneDistance / point_target.z, 0, 0, 0, PlaneDistance / point_target.z, 0, 0, 0, PlaneDistance, Width / 2, Height / 2);
		if (point_projected.x >= Width - 1 || point_projected.x < 0 || point_projected.y >= Height - 1 || point_projected.y < 0) continue;
		ImageOut[(int)(point_projected.y + 0.5)][(int)(point_projected.x + 0.5)] = brightness;
	}

	return ImageOut;
}

/** Transform 3D Parallelogram to 2D Parallelogram & Show transformed image
* @param Image image which parallelogram is projected to
* @param Height height of image to show
* @param Width width of image to show
* @param o coordinate for cornor of parallelogram
* @param n,m side vector starting from 'o' vector
* @param DotNumber number of dot expressing line (the more this parameter is bigger, the more result looks like line)
* @param PlaneDistance distance between camera and image plane that projects line (the more this parameter is bigger, the more line is enlarged)
* @param brightness brightness of dot marked on image
* @description refer to resource file: "Pinhole Camera.pptx"
*/
int** PinholeParallelogram(int** Image, double Height, double Width, Point3d o, Point3d n, Point3d m, int DotNumber, double PlaneDistance = 300, uint8_t brightness = 255)
{
	int** ImageOut = Image;

	o.y *= -1;
	n.y *= -1;
	m.y *= -1;

	double NLength = GetLength(n);
	double MLength = GetLength(m);
	double LengthUnit = (double)DotNumber / ((GetLength(n) + GetLength(m)) * 2.0);

	ImageOut = PinholeLine(Image, Height, Width, o, n, (int)(MLength * LengthUnit + 0.5));
	ImageOut = PinholeLine(Image, Height, Width, o, m, (int)(NLength * LengthUnit + 0.5));
	ImageOut = PinholeLine(Image, Height, Width, m, n + m - o, (int)(MLength * LengthUnit + 0.5));
	ImageOut = PinholeLine(Image, Height, Width, n, n + m - o, (int)(NLength * LengthUnit + 0.5));

	return ImageOut;
}

/** Transform 3D Parallelogram to 2D Parallelogram & Show transformed image
* @param Image image which parallelogram is projected to
* @param Height height of image to show
* @param Width width of image to show
* @param o,n,m coordinate for cornor of parallelogram
* @param Density density of dots (the bigger this value is, the more dense dots composing arallelogram is)
* @param PlaneDistance distance between camera and image plane that projects line (the more this parameter is bigger, the more line is enlarged)
* @param brightness brightness of dot marked on image
* @description refer to resource file: "Pinhole Camera.pptx"
*/
int** PinholeParallelogramFilled(int** Image, double Height, double Width, Point3d o, Point3d n, Point3d m, double Density = 1, double PlaneDistance = 300, uint8_t brightness = 255)
{
	int** ImageOut = Image;

	o.y *= -1;
	n.y *= -1;
	m.y *= -1;
	Density *= 0.1;

	double NLength = GetLength(n);
	double MLength = GetLength(m);

	for (double t = 0; t < 1; t += (double)1 / (NLength * Density))
	{
		for (double q = 0; q < 1; q += (double)1 / (MLength * Density))
		{
			Point3d point_target = o + t*(n - o) + q*(m - o);
			if (point_target.z == 0) point_target.z = 0.0001;

			Point3d point_projected = Affine(point_target, PlaneDistance / point_target.z, 0, 0, 0, PlaneDistance / point_target.z, 0, 0, 0, PlaneDistance, Width / 2, Height / 2);
			if (point_projected.x >= Width - 1 || point_projected.x < 0 || point_projected.y >= Height - 1 || point_projected.y < 0) continue;
			ImageOut[(int)(point_projected.y + 0.5)][(int)(point_projected.x + 0.5)] = brightness;
		}
	}
	return ImageOut;
}

/** Transform 3D Parallelogram to 2D Parallelogram & Show transformed image
* @param ImageDest image which 3D image is projected to
* @param HeightDest height of image to show
* @param WidthDest width of image to show
* @param ImageSrc image to project
* @param HeightSrc height of image to project
* @param WidthSrc width of image to project
* @param o,n,m coordinate for cornor of parallelogram
* @param Density density of dots (the bigger this value is, the more dense dots composing arallelogram is)
* @param PlaneDistance distance between camera and image plane that projects line (the more this parameter is bigger, the more line is enlarged)
* @param brightness brightness of dot marked on image
* @description refer to resource file: "Pinhole Camera.pptx"
*/
int** RenderImage(int** ImageDest, double HeightDest, double WidthDest, int** ImageSrc, double HeightSrc, double WidthSrc, Point3d o, Point3d n, Point3d m, double PlaneDistance = 300, uint8_t brightness = 255)
{
	int** ImageOut = ImageDest;

	o.y *= -1;
	n.y *= -1;
	m.y *= -1;

	for (int y = 0; y < HeightSrc; y++)
	{
		for (int x = 0; x < WidthSrc; x++)
		{
			Point3d point_target = o + (n - o)*x / WidthSrc + (m - o)*y / HeightSrc;
			if (point_target.z == 0) point_target.z = 0.0001;

			Point3d point_projected = Affine(point_target, PlaneDistance / point_target.z, 0, 0, 0, PlaneDistance / point_target.z, 0, 0, 0, PlaneDistance, WidthDest / 2, HeightDest / 2);
			if (point_projected.x >= WidthDest - 1 || point_projected.x < 0 || point_projected.y >= HeightDest - 1 || point_projected.y < 0) continue;
			ImageOut[(int)(point_projected.y + 0.5)][(int)(point_projected.x + 0.5)] = ImageSrc[y][x];
		}
	}

	return ImageOut;
}

/** Down size the image to 1/2
* @param Image image to down size
* @param Height height of image to down size
* @param Width width of image to down size
* @description refer to resource file: "Down Sampling.png"
*/
int** DownSampling2(int** Image, int Height, int Width)
{
	int** ImageOut = IntAlloc2(Height / 2, Width / 2);

	for (int y = 0; y < Height / 2; y++)
		for (int x = 0; x < Width / 2; x++)
		{
			int target_x = x * 2;
			int target_y = y * 2;
			int value = (Image[target_y][target_x] + Image[target_y][target_x + 1] + Image[target_y + 1][target_x] + Image[target_y + 1][target_x + 1]) / 4;

			ImageOut[y][x] = value;
		}

	return ImageOut;
}

#define MIN(a, b) (a < b ? a : b)
/** Find coordinate of block that matches with template 'Block'
* @param Image image to find matching
* @param Height height of 'Image'
* @param Width width of 'Image'
* @param Block image which has pattern to find in 'Image'
* @param HeightBlock height of block image
* @param WidthBlock width of block image
* @return Point3d.x,Point3d.y coordinate of location that error value is smallest
* @return Point3d.z error value of (x, y) location
*/
Point3d TemplateMatching(int** Image, int Height, int Width, int** Block, int HeightBlock, int WidthBlock)
{
	const int ListHeight = Height - HeightBlock + 1;
	const int ListWidth = Width - WidthBlock + 1;
	int** ErrorList = IntAlloc2(ListHeight, ListWidth);

	for (int b = 0; b < ListHeight; b++)
		for (int a = 0; a < ListWidth; a++)
			for (int y = 0; y < HeightBlock; y++)
				for (int x = 0; x < WidthBlock; x++)
					ErrorList[b][a] += abs(Image[y + b][x + a] - Block[y][x]);

	Point3d MatchPoint = Point3d(0, 0, 0);
	for (int y = 0; y < ListHeight; y++)
		for (int x = 0; x < ListWidth; x++)
			if (ErrorList[(int)MatchPoint.y][(int)MatchPoint.x] > ErrorList[y][x])
			{
				MatchPoint.x = x;
				MatchPoint.y = y;
				MatchPoint.z = ErrorList[y][x];
			}

	return MatchPoint;
}

template<typename Type>
Type** Allocate2(int Height, int Width)
{
	Type** ListOut = (Type**)calloc(Height, sizeof(Type*));
	for (int i = 0; i < Height; i++)
		ListOut[i] = (Type*)calloc(Width, sizeof(Type));

	return ListOut;
}

template<typename Type>
Type**** Allocate4(int Height, int Width, int Height2 = 0, int Width2 = 0)
{
	Type**** ListOut = (Type****)calloc(Height, sizeof(Type***));
	for (int i = 0; i < Height; i++)
		ListOut[i] = (Type***)calloc(Width, sizeof(Type**));

	if (Height2 > 0)
		for (int y = 0; y < Height; y++)
			for (int x = 0; x < Width; x++)
			{
				ListOut[y][x] = (Type**)calloc(Height2, sizeof(Type*));
				for (int j = 0; j < Height2; j++)
					if (Width2 > 0)
						ListOut[y][x][j] = (Type*)calloc(Width2, sizeof(Type));
			}

	return ListOut;
}

template<typename Type>
void Free2(Type** Data, int Height, int Width = 0)
{
	for (int i = 0; i < Height; i++)
		free(Data[i]);
	free(Data);
}

template<typename Type>
void Free4(Type**** Data, int Height, int Width, int Height2, int Width2 = 0)
{
	for (int y = 0; y < Height; y++)
	{
		for (int x = 0; x < Width; x++)
		{
			for (int b = 0; b < Height2; b++)
				free(Data[y][x][b]);
			free(Data[y][x]);
		}
		free(Data[y]);
	}
	free(Data);
}

/** Get average value of brightness of block in 'Image'
* @param Image block source image
* @param Height height of 'Image'
* @param Width width of 'Image'
* @return Average value of brightness of 'Image'
*/
double GetAverageBrightness(int** Image, int Height, int Width)
{
	unsigned int Total = 0;

	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
		{
			if (x < 0 || x >= Width || y < 0 || y >= Height)
				continue;

			Total += Image[y][x];
		}

	return Total / (Height * Width);
}

/** Down size the image to 1/N
* @param Image image to down size
* @param Height height of image to down size
* @param Width width of image to down size
* @param Spot origin coordinate of starting point of block
* @param HeightBlock height of block
* @param WidthBlock width of block
*/
void ReadBlock(int** Image, int Height, int Width, int** Block, int HeightBlock, int WidthBlock, Point2i Spot = Point2i(0, 0))
{
	for (int y = 0; y < HeightBlock; y++)
		for (int x = 0; x < WidthBlock; x++)
		{
			if (Spot.y + y >= Height || Spot.x + x >= Width) continue;
			Block[y][x] = Image[Spot.y + y][Spot.x + x];
		}
}

/** Down size the image to 1/N
* @param Image image to down size
* @param Height height of image to down size
* @param Width width of image to down size
* @param ImageSrc writing block on 'ImageDest'
* @param Spot origin coordinate of starting point of block
* @param HeightSrc height of block from 'ImageSrc'
* @param WidthSrc width of block from "ImageSrc'
*/
void WriteBlock(int** ImageOut, int Height, int Width, int** ImageSrc, Point2i Spot, int HeightSrc, int WidthSrc)
{
	for (int y = 0; y < HeightSrc; y++)
		for (int x = 0; x < WidthSrc; x++)
		{
			Point2i Target = Point2i(Spot.x + x, Spot.y + y);
			if (Target.x < 0 || Target.y < 0 || Target.x >= Width || Target.y >= Height) continue;
			ImageOut[Target.y][Target.x] = ImageSrc[y][x];
		}
}

/** Down size the image to 1/N
* @param Image image to down size
* @param Height height of image to down size
* @param Width width of image to down size
* @param N downsampling ratio
* @description refer to resource file: "Down Sampling.png"
*/
void DownSampling(int** Image, int Height, int Width, int N, int** ImageOut)
{
	// Size Factor
	const int HeightOut = Height / N;
	const int WidthOut = Width / N;

	// Initialize
	int** ImageBuffer = IntAlloc2(N, N);

	for (int y = 0; y < HeightOut; y++)
		for (int x = 0; x < WidthOut; x++)
		{
			ReadBlock(Image, Height, Width, ImageBuffer, N, N, Point2i(x * N, y * N));
			ImageOut[y][x] = GetAverageBrightness(ImageBuffer, N, N);
		}

	IntFree2(ImageBuffer, N, N);
}

/** Get error between 'Image1' and 'Image2'
* @param Image1 image to get error
* @param Image2 other image to get error
* @param Height height of 'Image1' & 'Image2'
* @param Width width of 'Image1' & 'Image2'
*/
int GetError(int** Image1, int** Image2, int Height, int Width)
{
	int Total = 0;

	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
			Total += abs(Image1[y][x] - Image2[y][x]);

	return Total;
}

template<typename Type>
void ImageScaling(Type** Image, int Height, int Width, float Scale, Type** ImageOut)
{
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
			ImageOut[y][x] = Image[y][x] * Scale + 0.5;
}

template<typename Type>
void ImageCalibrating(Type** Image, int Height, int Width, int value, Type** ImageOut)
{
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
			ImageOut[y][x] = Image[y][x] + value;
}

template<typename Type>
void ImageCliping(Type** Image, int Height, int Width, Type** ImageOut, Type MaxValue, Type MinValue = 0)
{
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
		{
			if (Image[y][x] > MaxValue)
				ImageOut[y][x] = MaxValue;
			else if (Image[y][x] < MinValue)
				ImageOut[y][x] = MinValue;
			else
				ImageOut[y][x] = Image[y][x];
		}
}

void CopyImage(int** ImageDest, int** ImageSrc, int Height, int Width)
{
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
			ImageDest[y][x] = ImageSrc[y][x];
}

void GetDBlock2Mean(int** Image, int Height, int Width, int N, Point2i Spot, int** DBlock2Mean)
{
	// Initialize
	const int DBlockRow = Height - 2 * N + 1;
	const int DBlockColumn = Width - 2 * N + 1;

	int** DBlock = IntAlloc2(2 * N, 2 * N);
	int** DBlock2 = IntAlloc2(N, N);

	// Processing
	ReadBlock(Image, Height, Width, DBlock, 2 * N, 2 * N, Spot);
	DownSampling(DBlock, 2 * N, 2 * N, 2, DBlock2);
	const int DBlock2Avg = GetAverageBrightness(DBlock2, N, N);
	for (int b = 0; b < N; b++)
		for (int a = 0; a < N; a++)
			DBlock2Mean[b][a] = DBlock2[b][a] - DBlock2Avg;

	// Free Unnecessary Memory
	IntFree2(DBlock, 2 * N, 2 * N);
	IntFree2(DBlock2, N, N);
}

enum GeometricTransform {
	GT0,
	GT90,
	GT180,
	GT270,
	GTInverseX,
	GTInverseY,
	GTInverseSlash,
	GTInverseBackSlash
};

typedef struct Information{
	Point2i** MinErrorCoordinate;
	int** BlockAvg;
	GeometricTransform** MinErrorGT;
	float** MinErrorAlpha;
	Information* LastInformation = NULL;
	Point2i* LastBlockPos = NULL;
	Point2i* LastDBlockPos = NULL;
	int LastProcessNumber = 0;
}Information;

void GetGTImage(int** ImageDest, int** Image, int N, GeometricTransform GT = GT0)
{
	switch (GT)
	{
	case GT90: // 90 degree rotation
		for (int y = 0; y < N; y++)
			for (int x = 0; x < N; x++)
				ImageDest[y][x] = Image[x][N - y - 1];
		break;
	case GT180: // 180 degree rotation
		for (int y = 0; y < N; y++)
			for (int x = 0; x < N; x++)
				ImageDest[y][x] = Image[N - y - 1][N - x - 1];
		break;
	case GT270: // 270 degree rotation
		for (int y = 0; y < N; y++)
			for (int x = 0; x < N; x++)
				ImageDest[y][x] = Image[N - x - 1][y];
		break;
	case GTInverseX: // inverse to x-axis
		for (int y = 0; y < N; y++)
			for (int x = 0; x < N; x++)
				ImageDest[y][x] = Image[N - y - 1][x];
		break;
	case GTInverseY: // inverse to y-axis
		for (int y = 0; y < N; y++)
			for (int x = 0; x < N; x++)
				ImageDest[y][x] = Image[y][N - x - 1];
		break;
	case GTInverseSlash: // inverse to y=x
		for (int y = 0; y < N; y++)
			for (int x = 0; x < N; x++)
				ImageDest[y][x] = Image[N - x - 1][N - y - 1];
		break;
	case GTInverseBackSlash: // inverse to y=-x
		for (int y = 0; y < N; y++)
			for (int x = 0; x < N; x++)
				ImageDest[y][x] = Image[x][y];
		break;
	}
}

inline int GetMinErrorGTAlpha(int** BlockMean, int***** DBlock2Mean, int N, Point2i BlockSpot, GeometricTransform* MinErrorGT = NULL, float* MinErrorAlpha = NULL, float MinAlpha = 0.2f, float MaxAlpha = 1.0f, float AlphaDiff = 0.1f)
{
	// Initialize
	int** ScaledImageBuffer = IntAlloc2(N, N);
	int ErrorBuffer = 0;
	int Error = INT_MAX;

	// Processing
	for (int i = 0; i < 8; i++)
		for (float alpha = MaxAlpha; alpha >= MinAlpha; alpha -= AlphaDiff)
		{
			ImageScaling<int>(DBlock2Mean[i][BlockSpot.y][BlockSpot.x], N, N, alpha, ScaledImageBuffer);
			ErrorBuffer = GetError(BlockMean, ScaledImageBuffer, N, N);
			if (Error > ErrorBuffer)
			{
				Error = ErrorBuffer;
				*MinErrorGT = (GeometricTransform)i;
				*MinErrorAlpha = alpha;
			}
		}

	// Clear unnecessary memory
	Free2<int>(ScaledImageBuffer, N, N);

	return Error;
}

bool ErrorCmp(const ErrorInfo &Info1, const ErrorInfo &Info2) {
	if (Info1.Error > Info2.Error)
		return true;
	else
		return false;
}

Information Encode(int** Image, int Height, int Width, int N, bool IsInitialProcess = true)
{
	// Factors to save
	float** MinErrorAlpha;
	GeometricTransform** MinErrorGT;
	Point2i** MinErrorCoordinate;
	int** BlockAvg;

	// Error Factor
	int** ErrorList;

	// Block Factor
	const int BlockRow = Height / N;
	const int BlockColumn = Width / N;

	// DBlock Factor
	int**** DBlock2Mean[8];
	const int DBlockRow = Height - 2 * N + 1;
	const int DBlockColumn = Width - 2 * N + 1;

	// Initialize
	MinErrorAlpha = Allocate2<float>(BlockRow, BlockColumn);
	MinErrorGT = Allocate2<GeometricTransform>(BlockRow, BlockColumn);
	MinErrorCoordinate = Allocate2<Point2i>(BlockRow, BlockColumn);
	for (int y = 0; y < BlockRow; y++)
		for (int x = 0; x < BlockColumn; x++)
			MinErrorCoordinate[y][x] = Point2i(0, 0);

	ErrorList = Allocate2<int>(DBlockRow, DBlockColumn);
	ImageCalibrating<int>(ErrorList, DBlockRow, DBlockColumn, 10000000, ErrorList);

	BlockAvg = Allocate2<int>(BlockRow, BlockColumn);
	for(int i = 0; i < 8; i++)
		DBlock2Mean[i] = Allocate4<int>(DBlockRow, DBlockColumn, N, N);

#ifdef MODE_DEBUG
	printf("============================== Start Encoding ==============================\n");
#endif

	// Get DBlock & DBlock2 & DBlock2Mean
	for (int y = 0; y < DBlockRow; y++)
		for (int x = 0; x < DBlockColumn; x++)
		{
			GetDBlock2Mean(Image, Height, Width, N, Point2i(x, y), DBlock2Mean[0][y][x]);
			for (int i = 1; i < 8; i++)
				GetGTImage(DBlock2Mean[i][y][x], DBlock2Mean[0][y][x], N, (GeometricTransform)i);
		}

#ifdef MODE_DEBUG
	printf("============================== Finished Calculating DBlock2Mean ==============================\n\n");
#endif

	// Get Minimum Error Coordinate
	int** Block = IntAlloc2(N, N);
	int** BlockMean = IntAlloc2(N, N);
	int** MinError = IntAlloc2(BlockRow, BlockColumn);
	float** AlphaBuffer = Allocate2<float>(DBlockRow, DBlockColumn);
	GeometricTransform** GTBuffer = Allocate2<GeometricTransform>(DBlockRow, DBlockColumn);

	for (int y = 0; y < BlockRow; y++)
		for (int x = 0; x < BlockColumn; x++)
		{
			// get block & it's average
			ReadBlock(Image, Height, Width, Block, N, N, Point2i(x * N, y * N));
			BlockAvg[y][x] = GetAverageBrightness(Block, N, N);

			// get block_means
			for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					BlockMean[j][i] = Block[j][i] - BlockAvg[y][x];

			// get error
			for (int j = 0; j < DBlockRow; j++)
				for (int i = 0; i < DBlockColumn; i++)
					ErrorList[j][i] = GetMinErrorGTAlpha(BlockMean, DBlock2Mean, N, Point2i(i, j), &GTBuffer[j][i], &AlphaBuffer[j][i]);

			// get minimum error coordinate
			for (int j = 0; j < DBlockRow; j++)
				for (int i = 0; i < DBlockColumn; i++)
					if (ErrorList[MinErrorCoordinate[y][x].y][MinErrorCoordinate[y][x].x] >= ErrorList[j][i])
					{
						MinErrorCoordinate[y][x] = Point2i(i, j);
						MinErrorGT[y][x] = GTBuffer[j][i];
						MinErrorAlpha[y][x] = AlphaBuffer[j][i];
						MinError[y][x] = ErrorList[j][i];
					}

#ifdef MODE_DEBUG
			printf("Block(%3d, %3d) Minimum Error Coordinate: (%3d, %3d) Block Average Brightness: (%3d) GT: (%d) MinErrorAlpha: (%f)\n", x * N, y * N, MinErrorCoordinate[y][x].x, MinErrorCoordinate[y][x].y, BlockAvg[y][x], MinErrorGT[y][x], MinErrorAlpha[y][x]);
#endif
		}

	Information result;

	if (IsInitialProcess) {
		const int LastProcessNumber = BlockRow * BlockColumn / 5;

		Information* LastInfo = (Information*)calloc(LastProcessNumber, sizeof(Information));
		Point2i* LastBlockPos = (Point2i*)calloc(LastProcessNumber, sizeof(Point2i));
		vector<ErrorInfo> LastProcessList;

		for (int y = BlockRow - 1; y >= 0; y--)
			for (int x = BlockColumn - 1; x >= 0; x--)
				LastProcessList.push_back(ErrorInfo(x, y, MinError[y][x]));
		sort(LastProcessList.begin(), LastProcessList.end(), ErrorCmp);

		for (int i = LastProcessNumber - 1; i >= 0; i--)
		{
			ErrorInfo Target = LastProcessList.back();

			int** TempBlock = IntAlloc2(N, N);
			ReadBlock(Image, Height, Width, TempBlock, N, N, Point2i(Target.BlockX, Target.BlockY));
			
			LastInfo[i] = Encode(TempBlock, N, N, N / 2, false);
			LastBlockPos[i] = Point2i(Target.BlockX, Target.BlockY);

			std::cout << "(" << i << ")" << Target.BlockX << ", " << Target.BlockY << " -> " << ": " << Target.Error << std::endl;
			LastProcessList.pop_back();
		}

		result.LastInformation = LastInfo;
		result.LastProcessNumber = LastProcessNumber;
		result.LastBlockPos = LastBlockPos;
	}

	// Return Information which is for Decoding
	result.BlockAvg = BlockAvg;
	result.MinErrorCoordinate = MinErrorCoordinate;
	result.MinErrorGT = MinErrorGT;
	result.MinErrorAlpha = MinErrorAlpha;

	// Free Unnecessary Memory
	Free2<int>(ErrorList, DBlockRow, DBlockColumn);
	for (int i = 0; i < 8; i++)
		Free4<int>(DBlock2Mean[i], DBlockRow, DBlockColumn, N, N);
	Free2<int>(Block, N, N);
	Free2<int>(BlockMean, N, N);
	Free2<int>(MinError, BlockRow, BlockColumn);
	Free2<float>(AlphaBuffer, DBlockRow, DBlockColumn);
	Free2<GeometricTransform>(GTBuffer, DBlockRow, DBlockColumn);

	return result;
}

void Decode(int** ImageOut, int** Image, int Height, int Width, int N, Information arguments, bool IsInitialProcess = true)
{
	// Block Factor
	const int BlockRow = Height / N;
	const int BlockColumn = Width / N;

	int** ImageBuffer = IntAlloc2(N, N);
	int** DBlock2Mean = IntAlloc2(N, N);

	for (int y = 0; y < BlockRow; y++)
		for (int x = 0; x < BlockColumn; x++)
		{
			GetDBlock2Mean(Image, Height, Width, N, Point2i(arguments.MinErrorCoordinate[y][x].x, arguments.MinErrorCoordinate[y][x].y), DBlock2Mean);
			ImageScaling(DBlock2Mean, N, N, arguments.MinErrorAlpha[y][x], ImageBuffer);
			GetGTImage(DBlock2Mean, ImageBuffer, N, arguments.MinErrorGT[y][x]);
			ImageCalibrating(DBlock2Mean, N, N, arguments.BlockAvg[y][x], DBlock2Mean);
			ImageCliping(DBlock2Mean, N, N, DBlock2Mean, 255, 0);

			WriteBlock(ImageOut, Height, Width, DBlock2Mean, Point2i(x * N, y * N), N, N);
		}

	if (IsInitialProcess)
	{
		for (int i = arguments.LastProcessNumber - 1; i >= 0; i--)
		{
			int** Block = IntAlloc2(N, N);

			ReadBlock(Image, Height, Width, Block, N, N, arguments.LastBlockPos[i]);
			Decode(Block, Block, N, N, N / 2, arguments.LastInformation[i], false);
			WriteBlock(ImageOut, Height, Width, Block, arguments.LastBlockPos[i], N, N);
		}
	}

	// Free Unnecessary Memory
	IntFree2(ImageBuffer, N, N);
	IntFree2(DBlock2Mean, N, N);
}

int GetMaxBrightnessCross(int** Image, int Height, int Width, Point2i Spot)
{
	int Brightness[5] = { 0 };
	int MaxBrightness = 0;

	Brightness[0] = Image[Spot.y][Spot.x]; // Center
	if (Spot.x + 1 < Width)
		Brightness[1] = Image[Spot.y][Spot.x + 1]; // 3
	if (Spot.y - 1 >= 0)
		Brightness[2] = Image[Spot.y - 1][Spot.x]; // 6
	if (Spot.x - 1 >= 0)
		Brightness[3] = Image[Spot.y][Spot.x - 1]; // 9
	if (Spot.y + 1 < Height)
		Brightness[4] = Image[Spot.y + 1][Spot.x]; // 12

	for (int i = 0; i < 5; i++)
		if (MaxBrightness < Brightness[i])
			MaxBrightness = Brightness[i];

	return MaxBrightness;
}

int GetMinBrightnessCross(int** Image, int Height, int Width, Point2i Spot)
{
	int Brightness[5] = { 0 };
	int MaxBrightness = INT_MAX;

	Brightness[0] = Image[Spot.y][Spot.x]; // Center
	if (Spot.x + 1 < Width)
		Brightness[1] = Image[Spot.y][Spot.x + 1]; // 3
	if (Spot.y - 1 >= 0)
		Brightness[2] = Image[Spot.y - 1][Spot.x]; // 6
	if (Spot.x - 1 >= 0)
		Brightness[3] = Image[Spot.y][Spot.x - 1]; // 9
	if (Spot.y + 1 < Height)
		Brightness[4] = Image[Spot.y + 1][Spot.x]; // 12

	for (int i = 0; i < 5; i++)
		if (MaxBrightness > Brightness[i])
			MaxBrightness = Brightness[i];

	return MaxBrightness;
}

void MaxOperation(int** Image, int Height, int Width, int** ImageOut)
{
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
			ImageOut[y][x] = GetMaxBrightnessCross(Image, Height, Width, Point2i(x, y));

	ImageShow("MaxOperation", ImageOut, Height, Width);
}

void MinOperation(int** Image, int Height, int Width, int** ImageOut)
{
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
			ImageOut[y][x] = GetMinBrightnessCross(Image, Height, Width, Point2i(x, y));

	ImageShow("MinOperation", ImageOut, Height, Width);
}

void DiffOperation(int** MaxImage, int** MinImage, int Height, int Width, int** ImageOut)
{
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
			ImageOut[y][x] = MaxImage[y][x] - MinImage[y][x];

	ImageShow("DiffOperation", ImageOut, Height, Width);
}

void MaxNtimes(int** Image, int Height, int Width, int N, int** ImageOut)
{
	int** ImageBuffer = IntAlloc2(Height, Width);
	CopyImage(ImageBuffer, Image, Height, Width);

	for (int i = 0; i < N; i++)
	{
		MaxOperation(ImageBuffer, Height, Width, ImageOut);
		if (i == N - 1) break;
		CopyImage(ImageBuffer, ImageOut, Height, Width);
	}

	IntFree2(ImageBuffer, Height, Width);
}

void MinNtimes(int** Image, int Height, int Width, int N, int** ImageOut)
{
	int** ImageBuffer = IntAlloc2(Height, Width);
	CopyImage(ImageBuffer, Image, Height, Width);

	for (int i = 0; i < N; i++)
	{
		MinOperation(ImageBuffer, Height, Width, ImageOut);
		if (i == N - 1) continue;
		CopyImage(ImageBuffer, ImageOut, Height, Width);
	}

	IntFree2(ImageBuffer, Height, Width);
}

double PSNR(int** Image1, int** Image2, int Height, int Width)
{
	double err = 0.0;
	for (int i = 0; i < Height; i++) for (int j = 0; j < Width; j++)
		err += ((double)Image1[i][j] - Image2[i][j]) * (Image1[i][j] - Image2[i][j]);

	err = err / (Width*Height);

	return(10.0 * log10(255 * 255.0 / err));
}

class Timer
{
public:
	void start()
	{
		m_StartTime = std::chrono::system_clock::now();
		m_bRunning = true;
	}

	void stop()
	{
		m_EndTime = std::chrono::system_clock::now();
		m_bRunning = false;
	}

	double elapsedMilliseconds()
	{
		std::chrono::time_point<std::chrono::system_clock> endTime;

		if (m_bRunning)
		{
			endTime = std::chrono::system_clock::now();
		}
		else
		{
			endTime = m_EndTime;
		}

		return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
	}

	double elapsedSeconds()
	{
		return elapsedMilliseconds() / 1000.0;
	}

private:
	std::chrono::time_point<std::chrono::system_clock> m_StartTime;
	std::chrono::time_point<std::chrono::system_clock> m_EndTime;
	bool                                               m_bRunning = false;
};

#define LECTURE
#define TEST
int main()
{
	Timer Clock;
	Clock.start();
	double LastTime = 0;

	/** Image Pointer */
	int** DogImage;
	int** LenaImage;
	int** OriginalImage;
	int** DrawingImage;
	int** AffinedImage;
	int** BilinearInterpolationImage;
	int** RotatedImage;

	int** CalendarImage;
	int** PasteImage;
	int** WhiteImage;
	int** BlackImage;
	int** Black256Image;

	/** width, height of image */
	int Height, Width;
	int LenaHeight, LenaWidth;
	int DogHeight, DogWidth;
	int Icon48Height, Icon48Width;

	/** Initialize */
	DogImage = ReadImage("DOG256.jpg", &DogHeight, &DogWidth);
	LenaImage = ReadImage("LENA256.bmp", &LenaHeight, &LenaWidth);
	OriginalImage = ReadImage("Koala.bmp", &Height, &Width);
	DrawingImage = IntAlloc2(Height, Width);
	AffinedImage = IntAlloc2(Height, Width);
	BilinearInterpolationImage = IntAlloc2(Height, Width);
	RotatedImage = IntAlloc2(Height, Width);

	CalendarImage = ReadImage("Calendar48.png", &Icon48Height, &Icon48Width);
	PasteImage = ReadImage("Paste48.png", &Icon48Height, &Icon48Width);
	WhiteImage = ReadImage("White48.png", &Icon48Height, &Icon48Width);
	BlackImage = ReadImage("Black48.png", &Icon48Height, &Icon48Width);
	Black256Image = ReadImage("Black256.png", &LenaHeight, &LenaWidth);

	/** Image Processing */
#ifdef LECTURE
#ifndef TEST
	// Drawing Circle
	DrawingImage = DrawCircle(OriginalImage, Height, Width, 200, 200, 80, 220);
	DrawingImage = DrawCircle(OriginalImage, Height, Width, 350, 350, 40, 120);
	std::cout << "Drawing Circle: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Affine Transform
	for (int y = 0; y < Height - 1; y++)
		for (int x = 0; x < Width - 1; x++)
		{
			Point2d point(x, y);
			Point2d point_out = Affine(point, 1.0, 0.4, 0.4, 2.0);

			if (point_out.y > Height - 1 || point_out.y < 0 || point_out.x > Width - 1 || point_out.x < 0) continue;
			AffinedImage[(int)(point_out.y + 0.5)][(int)(point_out.x + 0.5)] = OriginalImage[y][x];
		}
	std::cout << "Affine Transform: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Bilinear Interpolaion
	BilinearInterpolationImage = BilinearInterpolation(OriginalImage, Height, Width, 1.5, 0, 0, 1.5);
	std::cout << "Bilinear Interpolation: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Rotation Transform
	RotatedImage = RotationTransform(OriginalImage, Height, Width, 45, Height / 2, Width / 2);
	std::cout << "Rotation Transform: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// 3D Coordinate
	Point3d a, b, c, d, e, f, g, h;
	a = Point3d(0, -100, 500);
	b = Point3d(200, -100, 700);
	c = Point3d(0, -100, 900);
	d = Point3d(-200, -100, 700);
	e = Point3d(0, -100 - 200 * sqrt(2), 500);
	f = Point3d(200, -100 - 200 * sqrt(2), 700);
	g = Point3d(0, -100 - 200 * sqrt(2), 900);
	h = Point3d(-200, -100 - 200 * sqrt(2), 700);

	// Pinhole Camera - Line 3D Projection
	const int PinholeLineImageHeight = 768;
	const int PinholeLineImageWidth = 1024;

	int** PinholeCameraLineImage = IntAlloc2(PinholeLineImageHeight, PinholeLineImageWidth);
	PinholeCameraLineImage = PinholeLine(PinholeCameraLineImage, PinholeLineImageHeight, PinholeLineImageWidth, a, b, 1000, 500);
	PinholeCameraLineImage = PinholeLine(PinholeCameraLineImage, PinholeLineImageHeight, PinholeLineImageWidth, e, f, 1000, 500);
	std::cout << "Pinhole Camera - Line 3D Projection: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Pinhole Camera - Rectangle 3D Projection
	const int PinholeRectangleImageHeight = 768;
	const int PinholeRectangleImageWidth = 1024;

	int** PinholeCameraRectangleImage = IntAlloc2(PinholeRectangleImageHeight, PinholeRectangleImageWidth);
	PinholeCameraRectangleImage = PinholeParallelogram(PinholeCameraRectangleImage, PinholeRectangleImageHeight, PinholeRectangleImageWidth, c, b, h, 500);
	std::cout << "Pinhole Camera - Rectangle 3D Projection: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Pinhole Camera - Filled Rectangle 3D Projection
	const int PinholeRectangleFilledImageHeight = 768;
	const int PinholeRectangleFilledImageWidth = 1024;

	int** PinholeCameraRectangleFilledImage = IntAlloc2(PinholeRectangleImageHeight, PinholeRectangleImageWidth);
	PinholeCameraRectangleFilledImage = PinholeParallelogramFilled(PinholeCameraRectangleFilledImage, PinholeRectangleImageHeight, PinholeRectangleImageWidth, d, c, f, 10, 500);
	std::cout << "Pinhole Camera - Filled Rectangle 3D Projection: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Rendering Image
	const int RenderingHeight = 768;
	const int RenderingWidth = 1024;

	int** RenderingImage = IntAlloc2(RenderingHeight, RenderingWidth);
	RenderingImage = RenderImage(RenderingImage, RenderingHeight, RenderingWidth, OriginalImage, Height, Width, a, b, e, 450);
	std::cout << "Rendering Image: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Cube Rendering
	const int CubeImageHeight = 768;
	const int CubeImageWidth = 1024;

	int** CubeImage = IntAlloc2(CubeImageHeight, CubeImageWidth);
	CubeImage = RenderImage(CubeImage, CubeImageHeight, CubeImageWidth, OriginalImage, Height, Width, d, a, h, 500);
	CubeImage = RenderImage(CubeImage, CubeImageHeight, CubeImageWidth, OriginalImage, Height, Width, f, e, b, 500);
	CubeImage = RenderImage(CubeImage, CubeImageHeight, CubeImageWidth, OriginalImage, Height, Width, a, b, d, 500);
	CubeImage = PinholeLine(CubeImage, CubeImageHeight, CubeImageWidth, Point3d(-100, 0, 0.1), Point3d(100, 0, 0.1), 2000, 1, 128);
	CubeImage = PinholeLine(CubeImage, CubeImageHeight, CubeImageWidth, Point3d(0, -100, 0.1), Point3d(0, 100, 0.1), 2000, 1, 128);
	CubeImage = PinholeLine(CubeImage, CubeImageHeight, CubeImageWidth, Point3d(0, 0, -100), Point3d(0, 0, 100), 2000, 1, 128);
	std::cout << "Cube Rendering: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	/** 2019-04-09 Lecture
	* Fractal Encoding
	- Template Matching
	- Geometric Transform - 8가지
	- Sampling - 0.3 ~ 1.0
	- Down Sampling
	- 평균값 + 평균제거
	* Fractal Decoding
	*/

	// Down Sampling 1/2
	int** DownSampling2Image = IntAlloc2(Height / 2, Width / 2);
	DownSampling2Image = DownSampling2(OriginalImage, Height, Width);
	std::cout << "Down Sampling 1/2: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Template Matching
	int TemplateHeight;
	int TemplateWidth;

	int** TemplateImage = ReadImage("template.bmp", &TemplateHeight, &TemplateWidth);
	int** MatchMarkingImage = IntAlloc2(Height, Width);
	for (int y = 0; y < Height; y++)
		for (int x = 0; x < Width; x++)
			MatchMarkingImage[y][x] = OriginalImage[y][x];

	Point3d point = TemplateMatching(OriginalImage, Height, Width, TemplateImage, TemplateHeight, TemplateWidth);
	for (int y = point.y; y < point.y + TemplateHeight; y++)
		for (int x = point.x; x < point.x + TemplateWidth; x++)
			MatchMarkingImage[y][x] = 255;

	std::cout << "Template Matching: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Flipped Template Matching
#define N 16
	int FlippedTemplateHeight;
	int FlippedTemplateWidth;

	int** FlippedTemplateImage = ReadImage("template(flipping).bmp", &FlippedTemplateHeight, &FlippedTemplateWidth);
	int** FlippedMatchMarkingImage[8];
	int** ReflippedTemplateImage[8];

	for (int i = 0; i < 8; i++)
	{
		FlippedMatchMarkingImage[i] = IntAlloc2(Height, Width);
		ReflippedTemplateImage[i] = IntAlloc2(FlippedTemplateHeight, FlippedTemplateWidth);
		for (int y = 0; y < Height; y++)
			for (int x = 0; x < Width; x++)
				FlippedMatchMarkingImage[i][y][x] = OriginalImage[y][x];
	}

	for (int y = 0; y < FlippedTemplateHeight; y++)
		for (int x = 0; x < FlippedTemplateWidth; x++)
		{
			ReflippedTemplateImage[0][y][x] = FlippedTemplateImage[y][x];
			ReflippedTemplateImage[1][y][x] = FlippedTemplateImage[N - 1 - y][x];
			ReflippedTemplateImage[2][y][x] = FlippedTemplateImage[y][N - 1 - x];
			ReflippedTemplateImage[3][y][x] = FlippedTemplateImage[N - 1 - y][N - 1 - x];
			ReflippedTemplateImage[4][y][x] = FlippedTemplateImage[x][y];
			ReflippedTemplateImage[5][y][x] = FlippedTemplateImage[N - 1 - x][y];
			ReflippedTemplateImage[6][y][x] = FlippedTemplateImage[x][N - 1 - y];
			ReflippedTemplateImage[7][y][x] = FlippedTemplateImage[N - 1 - x][N - 1 - y];
		}

	Point3d point_flip[8];
	std::cout << "====================Template Matching Error Value====================" << std::endl;
	for (int i = 0; i < 8; i++)
	{
		point_flip[i] = TemplateMatching(OriginalImage, Height, Width, ReflippedTemplateImage[i], FlippedTemplateHeight, FlippedTemplateWidth);
		for (int y = point_flip[i].y; y < point_flip[i].y + FlippedTemplateHeight; y++)
			for (int x = point_flip[i].x; x < point_flip[i].x + FlippedTemplateWidth; x++)
				FlippedMatchMarkingImage[i][y][x] = 255;

		std::cout << "[" << i << "] " << point_flip[i].z << std::endl;
	}
	std::cout << "=====================================================================" << std::endl;
	std::cout << "Flipped Template Matching: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	// Generic Down Sampling
#define DOWNSAMPLING_NUMBER 10
	int DownSamplingImageHeight = 500;
	int DownSamplingImageWidth = 1000;
	int** DownSamplingImage = IntAlloc2(DownSamplingImageHeight, DownSamplingImageWidth);
	int** Result[DOWNSAMPLING_NUMBER];

	for (int y = 0; y < DownSamplingImageHeight; y++)
		for (int x = 0; x < DownSamplingImageWidth; x++)
			DownSamplingImage[y][x] = x * y % 255;

	for (int i = 0; i < DOWNSAMPLING_NUMBER; i++)
	{
		Result[i] = IntAlloc2(DownSamplingImageHeight, DownSamplingImageWidth);
		Result[i] = DownSampling(DownSamplingImage, DownSamplingImageHeight, DownSamplingImageWidth, i + 1);
	}
	std::cout << "Generic Down Sampling: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;

	std::cout << "=====================================================================" << std::endl << std::endl;
	std::cout << "Total Time used for Image Processing: " << Clock.elapsedSeconds() << " second" << std::endl;

	// Reading & Writing Block
	int** BlockImage = ReadBlock(OriginalImage, Height, Width, Point2i(100, 100), 400, 400);
	int** WrittenImage = WriteBlock(OriginalImage, Height, Width, DownSamplingImage, Point2i(100, 100), DownSamplingImageHeight, DownSamplingImageWidth);

	/** Show Image */
	ImageShow("Original Image", OriginalImage, Height, Width);
	ImageShow("Drawing Image", DrawingImage, Height, Width);
	ImageShow("Affined Image", AffinedImage, Height, Width);
	ImageShow("Bilinear Interpolation Image", BilinearInterpolationImage, Height, Width);
	ImageShow("Rotated Image", RotatedImage, Height, Width);
	ImageShow("Pinhole Camera - Line Image", PinholeCameraLineImage, PinholeLineImageHeight, PinholeLineImageWidth);
	ImageShow("Pinhole Camera - Rectangle Image", PinholeCameraRectangleImage, PinholeRectangleImageHeight, PinholeRectangleImageWidth);
	ImageShow("Pinhole Camera - Rectangle Filled Image", PinholeCameraRectangleFilledImage, PinholeRectangleFilledImageHeight, PinholeRectangleFilledImageWidth);
	ImageShow("Pinhole Camera - Rendering Image", RenderingImage, RenderingHeight, RenderingWidth);
	ImageShow("Cube Image Projection", CubeImage, CubeImageHeight, CubeImageWidth);
	ImageShow("Down Sampling Image", DownSampling2Image, Height / 2, Width / 2);
	ImageShow("Template Matching Image", MatchMarkingImage, Height, Width);
	for (int i = 0; i < 8; i++) // Show Reflipped Template Matching Images
	{
		char Title[100];
		char NumberString[100];

		// make title string
		itoa(i, NumberString, 10);
		strcpy(Title, "Reflipped Template Matching Image ");
		strcat(Title, NumberString);

		// show flipped match marking image
		ImageShow(Title, FlippedMatchMarkingImage[i], Height, Width);
	}
	ImageShow("Image to \'Downsample\'", DownSamplingImage, DownSamplingImageHeight, DownSamplingImageWidth);
	for (int i = 0; i < DOWNSAMPLING_NUMBER; i++) // Show Downsampled Images
	{
		char Title[100];
		char NumberString[100];

		// make title string
		itoa(i, NumberString, 10);
		strcpy(Title, "Downsampling (Ratio: ");
		strcat(Title, NumberString);
		strcat(Title, ")");

		// process & show images

		ImageShow(Title, Result[i], DownSamplingImageHeight / (i + 1), DownSamplingImageWidth / (i + 1));
	}
	ImageShow("Reading Block Image", BlockImage, 400, 400);
	ImageShow("Writing Block Image", WrittenImage, Height, Width);
#endif
#endif

#ifdef LECTURE
#ifdef TEST
	int** TargetImage = LenaImage;
	int** OtherImage = DogImage;
	const int TargetHeight = LenaHeight;
	const int TargetWidth = LenaWidth;
	const int TargetN = 8;
	const bool LastProcessTrigger = false;

	/*int** TargetImage = PasteImage;
	int** OtherImage = CalendarImage;
	const int TargetHeight = Icon48Height;
	const int TargetWidth = Icon48Width;
	const int TargetN = 8;
	const bool LastProcessTrigger = true;*/

	int** Image = IntAlloc2(TargetHeight, TargetWidth);
	int** ImageBuffer = IntAlloc2(TargetHeight, TargetWidth);
	CopyImage(Image, TargetImage, TargetHeight, TargetWidth);

	Information a;
	a = Encode(Image, TargetHeight, TargetWidth, TargetN, LastProcessTrigger);

	std::cout << "=====================================================================" << std::endl;
	std::cout << "Encoding: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();

	CopyImage(Image, OtherImage, TargetHeight, TargetWidth);
	for (int i = 0; i < 1000; i++)
	{
		Decode(ImageBuffer, Image, TargetHeight, TargetWidth, TargetN, a, LastProcessTrigger);
		Decode(Image, ImageBuffer, TargetHeight, TargetWidth, TargetN, a, LastProcessTrigger);
	}

	std::cout << "=====================================================================" << std::endl;
	std::cout << "Decoding 200th: " << Clock.elapsedSeconds() - LastTime << " second" << std::endl;
	LastTime = Clock.elapsedSeconds();
	ImageShow("Image", Image, TargetHeight, TargetWidth);

	printf("\n PSNR = %f\n", PSNR(Image, TargetImage, TargetHeight, TargetWidth));

#endif
#endif

#ifndef LECTURE
	// Processed Image
	int** Image_Out_1 = IntAlloc2(LenaHeight, LenaWidth);
	int** Image_Out_2 = IntAlloc2(LenaHeight, LenaWidth);
	int** Image_Out_3 = IntAlloc2(LenaHeight, LenaWidth);

	// Process
	MaxOperation(LenaImage, LenaHeight, LenaWidth, Image_Out_1);
	MinOperation(LenaImage, LenaHeight, LenaWidth, Image_Out_2);
	DiffOperation(Image_Out_1, Image_Out_2, LenaHeight, LenaWidth, Image_Out_3);

	MaxNtimes(LenaImage, LenaHeight, LenaWidth, 5, Image_Out_1);
	MinNtimes(LenaImage, LenaHeight, LenaWidth, 5, Image_Out_2);
	DiffOperation(Image_Out_1, Image_Out_2, LenaHeight, LenaWidth, Image_Out_3);
#endif

	return 0;
}