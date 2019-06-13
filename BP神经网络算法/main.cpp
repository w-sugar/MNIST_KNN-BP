#include "head.h"
#include<ctime>
void initialize();
void training();
void testing();
int test_size_list;//全局变量
float success_rate_;
int main()
{
	Image2BinaryData IBD(28, 28);											//设置图片大小(Height,Width)
	clock_t time1, time2,time3,time4;
	cout << "----------生成测试集文件-------------\n" << endl;
	string testfilefolder = "E:\\mnist\\100mnist";		//测试图片文件路径
	vector<string> testfileLists = IBD.getFileLists(testfilefolder);			//获得文件名列表

	test_size_list = testfileLists.size();
	cout << "Images Number: " << test_size_list << endl;									//输出文件个数
	string testimagebinfilepath = "E:\\mnist\\t10k-images-idx3-ubyte";		//测试图片转换保存路径
	string testlabelbinfilepath = "E:\\mnist\\t10k-labels-idx1-ubyte";		//测试标签转换保存路径
	vector<cv::Mat> TestImagesMat;															//用来存储测试图片像素值
	vector<int> test_image_labels(test_size_list);											//用来存储测试类标签列表
	IBD.ReadImage(testfilefolder, testfileLists, test_image_labels, TestImagesMat);			//读取测试图片
	IBD.Image2BinaryFile(testimagebinfilepath, TestImagesMat, test_image_labels);			//测试图片转二进制文件
	IBD.Label2BinaryFile(testlabelbinfilepath, test_image_labels);							//测试标签转二进制文件
	
	cout << "开始测试：" << endl;
	time1 = clock();
	initialize();
	time2 = clock();
	cout << "初始化时间：" << (double)(time2 - time1)/ CLOCKS_PER_SEC << 's' << endl;
	training();
	time3 = clock();
	cout << "训练时间：" << (double)(time3 - time2) / CLOCKS_PER_SEC << 's' << endl;
	testing();
	time4 = clock();
	cout << "测试时间：" << (double)(time4 - time3) / CLOCKS_PER_SEC << 's' << endl;
	cout << "准确率：" << success_rate_ << endl;
	cout << "运行总时间" << (double)(time4 - time1) / CLOCKS_PER_SEC <<'s'<< endl;
	system("pause");
	return 0;
}