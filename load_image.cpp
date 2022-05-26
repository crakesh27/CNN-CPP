#include "load_image.h"
#include "layers.h"

int reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float *load_mnist(char *fname, int &idim, int *&isize)
{
	ifstream file(fname, ios::binary);

	if (!file.is_open())
	{
		cout << "Could not open file\n";
		return NULL;
	}

	int magic_num, num_images, rows, cols;

	file.read((char *)&magic_num, sizeof(int));	 // magic number
	file.read((char *)&num_images, sizeof(int)); // number of images
	file.read((char *)&rows, sizeof(int));		 // rows
	file.read((char *)&cols, sizeof(int));		 // cols

	magic_num = reverseInt(magic_num);
	num_images = reverseInt(num_images);
	rows = reverseInt(rows);
	cols = reverseInt(cols);

	cout << num_images << " images" << endl;

	idim = 4;
	isize = new int[idim];
	isize[0] = num_images;
	isize[1] = 1;
	isize[2] = rows;
	isize[3] = cols;

	// cout << idim << " " <<isize[0] << " " << isize[1] <<" "<< isize[2] << " " << isize[3] << endl;

	float *data = new float[num_images * rows * cols];

	unsigned char ch;

	for (int i = 0; i < num_images; ++i)
	{
		for (int r = 0; r < rows; ++r)
		{
			for (int c = 0; c < cols; ++c)
			{
				file.read((char *)&ch, sizeof(ch));
				data[idx(i, r, c, rows, cols)] = ((float)(255 - ch)) / 255.0;
			}
		}
	}

	return data;
}

float *load_targets(char *fname, int &idim, int *&isize)
{
	ifstream file(fname, ios::binary);

	if (!file.is_open())
	{
		cout << "Could not open file\n";
		return NULL;
	}

	int magic_num, num_images;

	file.read((char *)&magic_num, sizeof(int));	 // magic number
	file.read((char *)&num_images, sizeof(int)); // number of images

	magic_num = reverseInt(magic_num);
	num_images = reverseInt(num_images);

	cout << num_images << " labels" << endl;

	idim = 2;
	isize = new int[idim];
	isize[0] = num_images;
	isize[1] = 10;

	// cout << idim << " " <<isize[0] << " " << isize[1] << endl;

	float *target = new float[num_images * 10];
	memset(target, 0, sizeof(float) * num_images * 10);

	unsigned char ch;

	for (int i = 0; i < num_images; ++i)
	{
		file.read((char *)&ch, sizeof(ch));
		target[idx(i, (int)ch, 10)] = 1.0f;
	}

	return target;
}