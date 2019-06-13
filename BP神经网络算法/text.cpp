#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cmath>
#include <time.h>
#include"head.h"

#pragma warning(disable:4996)
using namespace std;
extern float success_rate_;
extern int test_size_list;//全局变量
const int first = 784;//784个输入量
const int second = 100;//100个隐含量
const int third = 10;//10个输出量
const double alpha = 0.35;//学习效率参数

int input[first];//输入
int target[third];//期望输出
double weight1[first][second];//输入层-隐含层的权值
double weight2[second][third];//隐含层-输出层的权值
double output1[second];//隐藏层的输出
double output2[third];//输出层的输出
double delta1[second];//隐含层-输出层的误差
double delta2[third];//输入层-隐含层的误差
double b1[second];//输入层-隐含层的阈值
double b2[third];//隐含层-输出层阈值

double test_num = 0.0;
double test_success_count = 0.0;

double f_(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

//隐藏层的输出函数
void op1_()
{
	for (int j = 0; j < second; j++)
	{
		double sigma = 0;
		for (int i = 0; i < first; i++)
		{
			sigma += input[i] * weight1[i][j];
		}
		double x = sigma + b1[j];//隐藏层的输入
		output1[j] = f_(x);//隐藏层的输出
	}
}

//输出层的输出函数
void op2_()
{
	for (int k = 0; k < third; k++)
	{
		double sigma = 0;
		for (int j = 0; j < second; j++)
		{
			sigma += output1[j] * weight2[j][k];
		}
		double x = sigma + b2[k];//输出层的输入
		output2[k] = f_(x);//输出层的输出
	}
}

//输出层的误差函数
void dt2_()
{
	for (int k = 0; k < third; k++)
	{
		delta2[k] = (output2[k]) * (1.0 - output2[k]) * (output2[k] - target[k]);
	}
}

//隐藏层的误差函数
void dt1_() 
{
	for (int j = 0; j < second; j++) 
	{
		double sigma = 0;
		for (int k = 0; k < third; k++) 
		{
			sigma += weight2[j][k] * delta2[k];//输出层误差的加权总和
		}
		delta1[j] = (output1[j]) * (1.0 - output1[j]) * sigma;
	}
}

//隐藏层的反向调整函数
void feedback_second() {
	for (int j = 0; j < second; j++) 
	{
		b1[j] = b1[j] - alpha * delta1[j];//调整阈值
		for (int i = 0; i < first; i++) 
		{
			weight1[i][j] = weight1[i][j] - alpha * input[i] * delta1[j];//调整权值
		}
	}
}

//输出层的反向调整函数
void feedback_third() {
	for (int k = 0; k < third; k++) 
	{
		b2[k] = b2[k] - alpha * delta2[k];//调整阈值
		for (int j = 0; j < second; j++) 
		{
			weight2[j][k] = weight2[j][k] - alpha * output1[j] * delta2[k];//调整权值
		}
	}
}

//初始化函数
void initialize() 
{
	srand((int)time(0) + rand());
	for (int i = 0; i < first; i++) 
	{
		for (int j = 0; j < second; j++) 
		{
			weight1[i][j] = rand() % 1000 * 0.001 - 0.5;//输入层与隐含层的权值进行随机值的赋值
		}
	}
	for (int j = 0; j < second; j++) 
	{
		for (int k = 0; k < third; k++) 
		{
			weight2[j][k] = rand() % 1000 * 0.001 - 0.5;//隐含层与输出层的权值进行随机值的赋值
		}
	}
	for (int j = 0; j < second; j++) 
	{
		b1[j] = rand() % 1000 * 0.001 - 0.5;//输入层与隐含层的阈值进行随机值的赋值
	}
	for (int k = 0; k < third; k++) 
	{
		b2[k] = rand() % 1000 * 0.001 - 0.5;//隐含层与输出层的阈值进行随机值的赋值
	}
}

//训练函数
void training() 
{
	FILE *image_train;
	FILE *image_label;
	image_train = fopen("E:/mnist/mnist_train/train-images.idx3-ubyte", "rb");//打开训练图片数据集
	image_label = fopen("E:/mnist/mnist_train/train-labels.idx1-ubyte", "rb");//打开训练标签数据集
	if (image_train == NULL || image_label == NULL) 
	{
		cout << "can't open the file!" << endl;
		exit(0);
	}

	unsigned char image_buf[784];
	unsigned char label_buf[10];

	int useless[1000];
	fread(useless, 1, 16, image_train);//去掉数据集前16个字节
	fread(useless, 1, 8, image_label);//去掉数据集前8个字节

	int cnt = 0;//定义已训练图片数目
	cout << "Start training..." << endl;
	while (!feof(image_train) && !feof(image_label))
	{
		memset(image_buf, 0, 784);//每次进入循环将image_buf数组置零
		memset(label_buf, 0, 10);//每次进入循环将label_buf数组置零
		fread(image_buf, 1, 784, image_train);//将image_train中前784个字节读入到image_buf中
		fread(label_buf, 1, 1, image_label);//将image_label中前1个字节读入到label_buf中

		//初始化输入层的输入
		for (int i = 0; i < 784; i++) 
		{
			if ((unsigned int)image_buf[i] < 128) 
				input[i] = 0;
			else 
				input[i] = 1;
		}

		//初始化输出层的期望输出
		int target_value = (unsigned int)label_buf[0];
		for (int k = 0; k < third; k++) {
			target[k] = 0;
		}
		target[target_value] = 1;

		//不断调整网络的权值和阈值
		op1_();
		op2_();
		dt2_();
		dt1_();
		feedback_second();
		feedback_third();

		cnt++;
		//if (cnt % 1000 == 0) {
		//	cout << "training image: " << cnt << endl;
		//}
		//if (cnt ==2000) break;//当测试cnt张图片是跳出循环
	}
	cout << endl;
}

//测试函数
void testing()
{
	FILE *image_test;
	FILE *image_test_label;
	image_test = fopen("E:\\mnist\\t10k-images-idx3-ubyte", "rb+");
	image_test_label = fopen("E:\\mnist\\t10k-labels-idx1-ubyte", "rb+");
	if (image_test == NULL || image_test_label == NULL) {
		cout << "can't open the file!" << endl;
		exit(0);
	}

	unsigned char image_buf[784];
	unsigned char label_buf[10];

	int useless[1000];
	fread(useless, 1, 16, image_test);
	fread(useless, 1, 8, image_test_label);

	while (test_size_list)
	{
		test_size_list--;
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);
		fread(image_buf, 1, 784, image_test);
		fread(label_buf, 1, 1, image_test_label);

		//初始化输入层的输入
		for (int i = 0; i < 784; i++) {
			if ((unsigned int)image_buf[i] < 128) {
				input[i] = 0;
			}
			else {
				input[i] = 1;
			}
		}
		
		//以0/1进行图片样式输出
		int n = 0;
		for (int i = 0; i < 784; i++)
		{
			
			if (n == 28)
			{
				/*cout << endl;*/
				n = 0;
			}				
			/*cout << input[i];*/
			n++;
		}

		//初始化输出层的期望输出
		for (int k = 0; k < third; k++) {
			target[k] = 0;
		}
		int target_value = (unsigned int)label_buf[0];
		target[target_value] = 1;

		//初始化输出层的实际输出
		op1_();
		op2_();

		//选出输出层的最大值
		double max_value = -99999;
		int max_index = 0;
		int k;
		for (k = 0; k < third; k++) 
		{
			if (output2[k] > max_value) 
			{
				max_value = output2[k];
				max_index = k;
			}
		}

		//识别结果显示
		if (target[max_index] == 1)
		{
			test_success_count++;
			//cout << "识别结果：" << max_index <<endl;
		}
		//else
		//	cout << "识别结果：" << max_index << endl;
		test_num++;

	}
	//cout << endl;
	//cout << "The success rate: " << test_success_count / test_num << endl;
	success_rate_ = test_success_count / test_num;

}
