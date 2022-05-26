#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <iterator>
#include <random>
#include <time.h>

using namespace std;

enum BORDER
{
	valid,
	full
};

int idx(int a, int b, int c, int d, int B, int C, int D);
int idx(int a, int b, int c, int B, int C);
int idx(int a, int b, int B);

class Layer
{
public:
	float *weights;
	int wdim, *wsize;
	float *weight_grad;

	float *bias;
	int bdim, *bsize;
	float *bias_grad;

	float *inp;
	int idim, *isize;
	float *inp_grad;

	float *out;
	int odim, *osize;

	virtual float *forward_prop(float *inp, int idim, int *isize,
								int &odim, int *&osize)
	{
		cout << "Base class is not callable\n";
		return NULL;
	}

	virtual void grad_weights(float *out_grad)
	{
		cout << "Base class is not callable\n";
	}

	virtual float *grad_inputs(float *out_grad)
	{
		cout << "Base class is not callable\n";
		return NULL;
	}

	virtual void init(int idim, int *isize, int &odim, int *&osize)
	{
		cout << "Base class is not callable\n";
	}

	void random_init();
};

class Network
{
	std::vector<Layer *> layers;

public:
	float *out;
	int odim, *osize;

	void add_layer(Layer *l)
	{
		this->layers.push_back(l);
	}

	void forward_prop(float *inp, int idim, int *isize);

	void back_prop(float *target);

	float loss(float *target);

	float *loss_grad(float *target);

	void train(float *inp, int idim, int *isize, float *target, int num_epochs, int batch_size = 10, float alpha = 0.1, float eps = 0.01);

	void predict(float *inp, int idim, int *isize, float *target, int batch_size = 10);

	void update_weights(float alpha);

	void init(int idim, int *isize);
};

class Conv : public Layer
{
	int fW, fH;
	BORDER mode;
	int omaps;
	int strW, strH;

public:
	Conv(int fH, int fW, int omaps, BORDER mode = valid, int strW = 1, int strH = 1)
	{
		this->fW = fW;
		this->fH = fH;
		this->omaps = omaps;
		this->mode = mode;
		this->strW = strW;
		this->strH = strH;
	}

	float *forward_prop(float *inp, int idim, int *isize,
						int &odim, int *&osize);

	void grad_weights(float *out_grad);

	float *grad_inputs(float *out_grad);

	void init(int idim, int *isize, int &odim, int *&osize);
};

class Pool : public Layer
{
	int fW, fH;

public:
	Pool(int fH, int fW)
	{
		this->fW = fW;
		this->fH = fH;
	}

	float *forward_prop(float *inp, int idim, int *isize,
						int &odim, int *&osize);

	void grad_weights(float *out_grad);

	float *grad_inputs(float *out_grad);

	void init(int idim, int *isize, int &odim, int *&osize);
};

class ReLU : public Layer
{
public:
	float *forward_prop(float *inp, int idim, int *isize,
						int &odim, int *&osize);

	void grad_weights(float *out_grad);

	float *grad_inputs(float *out_grad);

	void init(int idim, int *isize, int &odim, int *&osize);
};

class FullyConnected : public Layer
{
	int num_out;

public:
	FullyConnected(int num_out)
	{
		this->num_out = num_out;
	}

	float *forward_prop(float *inp, int idim, int *isize,
						int &odim, int *&osize);

	void grad_weights(float *out_grad);

	float *grad_inputs(float *out_grad);

	void init(int idim, int *isize, int &odim, int *&osize);
};

class Sigmoid : public Layer
{
public:
	float *forward_prop(float *inp, int idim, int *isize,
						int &odim, int *&osize);

	void grad_weights(float *out_grad);

	float *grad_inputs(float *out_grad);

	void init(int idim, int *isize, int &odim, int *&osize);
};
