#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;

#define PI 3.14159265359

typedef struct {
	int r, g, b;
}int_rgb;


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

Matrix2X2 GetInverseMatrix(Matrix2X2 matrix)
{
	double determinant = matrix.a*matrix.d - matrix.c*matrix.d;
	Matrix2X2 MatrixOut = { matrix.d / determinant, -matrix.b / determinant,
		-matrix.c / determinant, matrix.a / determinant };

	return MatrixOut;
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

/** Bilinear Interpolation
 * @
*/
int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx)*(1.0 - dy)*A + dx*(1.0 - dy)*B
		+ (1.0 - dx)*dy*C + dx*dy*D;

	return((int)(value + 0.5));
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
	for (int x = 0; x < Width; x++)
		for (int y = 0; y < Height; y++)
		{
			double d = fabs(a * x - y + b) / sqrt(a*a + 1.0);

			if (d < Thickness) Image[y][x] = brightness;
		}

	return Image;
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
	for (int x = 0; x < Width; x++)
		for (int y = 0; y < Height; y++)
			if (powl(x - a, 2) + powl(y - b, 2) <= powl(r, 2))
				Image[y][x] = brightness;

	return Image;
}

/** Affine transform function
 * @param Input (x, y) coordinate
 * @param a, b, t1 NewX = a * OldX + b * OldY + t1
 * @param c, d, t2 NewY = c * OldX + d * OldY + t2
 */
 Point2d Affine(Point2d Input, double a, double b, double c, double d, double t1 = 0, double t2 = 0)
{
	 double NewX = a * Input.x + b * Input.y + t1;
	 double NewY = c * Input.x + d * Input.y + t2;

	 return Point2d(NewX, NewY);
}

 /** Affine transform function
 * @param Input (x, y) coordinate
 * @param matrix includes a, b, c, d arguments

 * @formula a, b, t1 NewX = a * OldX + b * OldY + t1
 * @formula c, d, t2 NewY = c * OldX + d * OldY + t2
 */
 Point2d Affine(Point2d Input, Matrix2X2 matrix, double t1 = 0, double t2 = 0)
 {
	 double NewX = matrix.a * Input.x + matrix.b * Input.y + t1;
	 double NewY = matrix.c * Input.x + matrix.d * Input.y + t2;

	 return Point2d(NewX, NewY);
 }

 /** Affine transform function
 * @param Input (x, y, w) coordinate
 * @param a, b, t1 NewX = a * OldX + b * OldY + t1
 * @param c, d, t2 NewY = c * OldX + d * OldY + t2
 */
 Point3d Affine(Point3d Input, double a, double b, double c, double d, double t1 = 0, double t2 = 0)
 {
	 double NewX = a * Input.x + b * Input.y + t1;
	 double NewY = c * Input.x + d * Input.y + t2;
	 double NewW = 1;

	 return Point3d(NewX, NewY, NewW);
 }

int main()
{
	/** Image Pointer */
	int** Image;
	int** ImageOut;

	/** width, height of image */
	int Height, Width;

	// Initialize
	Image = ReadImage("koala.jpg", &Height, &Width);
	ImageOut = IntAlloc2(Height, Width);

	// Image Processing & Show Image

	/** Drawing Circle
	DrawCircle(Image, Height, Width, 200, 200, 80, 220);
	DrawCircle(Image, Height, Width, 350, 350, 40, 120);
	*/

	/** Affine Transform
	for (int y = 0; y < Height - 1; y++)
		for (int x = 0; x < Width - 1; x++)
		{
			Point2d point(x, y);
			Point2d point_out = Affine(point, 1.0, 0.4, 0.4, 2.0);

			if (point_out.y > Height - 1 || point_out.y < 0 || point_out.x > Width - 1 || point_out.x < 0) continue;
			ImageOut[(int)(point_out.y+0.5)][(int)(point_out.x+0.5)] = Image[y][x];
		}
	*/

	/** Bilinear Interpolaion */
	for (int y = 0; y < Height - 1; y++)
		for (int x = 0; x < Width - 1; x++)
		{
			Matrix2X2 matrix = { 0.5, 0, 0, 0.5 };
			Matrix2X2 inverse_matrix = GetInverseMatrix(matrix);

			Point2d point(x, y);
			Point2d point_inverse_out = Affine(point, GetInverseMatrix(matrix));

			ImageOut[y][x] = BilinearInterpolation(Image, Width, Height, point_inverse_out.x, point_inverse_out.y);
		}
	
	// Draw transformed image
	ImageShow("test", ImageOut, Height, Width);

	return 0;
}