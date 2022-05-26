#include "layers.h"

int idx(int a, int b, int c, int d, int B, int C, int D)
{
	return d + c * D + b * C * D + a * B * C * D;
}
int idx(int a, int b, int c, int B, int C)
{
	return c + b * C + a * B * C;
}
int idx(int a, int b, int B)
{
	return b + a * B;
}

void Layer::random_init()
{
	int i, size = 1;

	for (i = 0; i < wdim; ++i)
		size *= wsize[i];

	this->weights = new float[size];

	std::random_device rd;
	std::mt19937 e2(rd());
	std::normal_distribution<> dist(0, 1);

	for (i = 0; i < size; ++i)
	{
		weights[i] = dist(e2);
	}

	size = 1;
	for (i = 0; i < bdim; ++i)
		size *= bsize[i];

	this->bias = new float[size];

	for (i = 0; i < size; ++i)
	{
		bias[i] = dist(e2);
	}
}

void Network::init(int idim, int *isize)
{

	int *in_size = new int[idim];

	for (int i = 0; i < idim; ++i)
		in_size[i] = isize[i];

	for (int i = 0; i < this->layers.size(); ++i)
	{
		layers[i]->init(idim, in_size, odim, osize);
		in_size = osize;
		idim = odim;
	}
}

void Network::forward_prop(float *inp, int idim, int *isize)
{
	float *temp = inp;

	int *in_size = new int[idim];

	for (int i = 0; i < idim; ++i)
		in_size[i] = isize[i];

	for (int i = 0; i < this->layers.size(); ++i)
	{
		temp = layers[i]->forward_prop(temp, idim, in_size,
									   odim, osize);
		in_size = osize;
		idim = odim;
	}

	this->out = temp;
}

void Network::back_prop(float *target)
{
	float *out_grad = loss_grad(target);

	for (int i = this->layers.size() - 1; i >= 0; --i)
	{
		layers[i]->grad_weights(out_grad);
		out_grad = layers[i]->grad_inputs(out_grad);
	}
}

void Network::train(float *inp, int idim, int *isize, float *target, int num_epochs, int batch_size, float alpha, float eps)
{
	int basize[4];
	basize[0] = batch_size;
	basize[1] = isize[1];
	basize[2] = isize[2];
	basize[3] = isize[3];

	init(idim, basize);

	int num_batches = isize[0] / basize[0];

	float l = 0;

	for (int e = 0; e < num_epochs; ++e)
	{
		l = 0;
		clock_t start = clock();
		cout << "Epoch : " << e + 1 << "/" << num_epochs << '\n';

		for (int b = 0; b < num_batches; ++b)
		{
			int inp_offset = isize[1] * isize[2] * isize[3] * b * batch_size;
			int target_offset = 10 * b * batch_size;

			forward_prop(inp + inp_offset, idim, basize);

			l += loss(target + target_offset);

			cout << "Batch : " << b + 1 << "/" << num_batches << " Training Loss: " << l / (b + 1) << '\r';

			back_prop(target + target_offset);
			update_weights(alpha);
		}
		l /= num_batches;

		cout << "\nTime taken : ";
		cout.precision(10);
		cout << fixed << float(clock() - start) / CLOCKS_PER_SEC << "sec" << endl;

		predict(inp, idim, isize, target, batch_size);

		if (l < eps)
			break;
	}
}

void Network::update_weights(float alpha)
{
	for (int i = 0; i < this->layers.size(); ++i)
	{
		int size = 1;
		for (int d = 0; d < layers[i]->wdim; ++d)
			size *= layers[i]->wsize[d];

		for (int j = 0; j < size; ++j)
		{
			layers[i]->weights[j] -= alpha * layers[i]->weight_grad[j];
		}

		size = 1;
		for (int d = 0; d < layers[i]->bdim; ++d)
			size *= layers[i]->bsize[d];

		for (int j = 0; j < size; ++j)
		{
			layers[i]->bias[j] -= alpha * layers[i]->bias_grad[j];
		}
	}
}

void Network::predict(float *inp, int idim, int *isize, float *target, int batch_size)
{
	int basize[4];
	basize[0] = batch_size;
	basize[1] = isize[1];
	basize[2] = isize[2];
	basize[3] = isize[3];

	int num_batches = isize[0] / basize[0];

	int acc_count = 0;
	float l = 0;

	for (int b = 0; b < num_batches; ++b)
	{
		int inp_offset = isize[1] * isize[2] * isize[3] * b * batch_size;
		int target_offset = 10 * b * batch_size;

		forward_prop(inp + inp_offset, idim, basize);
		l += loss(target + target_offset);

		for (int i = 0; i < basize[0]; ++i)
		{
			int max_out = 0;
			int max_target = 0;
			for (int j = 0; j < 10; ++j)
			{
				// cout << this->out[idx(i,j,10)] << '\n';

				if (this->out[idx(i, j, 10)] > out[idx(i, max_out, 10)])
					max_out = j;
				if (target[idx(b, i, j, basize[0], 10)] > target[idx(b, i, max_target, basize[0], 10)])
					max_target = j;
			}
			if (max_out == max_target)
				++acc_count;
		}
	}
	l /= num_batches;

	cout << "Test Accuracy : " << acc_count / (float)isize[0] << "\tTest Loss : " << l << endl;
}

float Network::loss(float *target)
{
	float l = 0.0f;

	int size = 1;

	for (int i = 1; i < this->odim; ++i)
	{
		size *= this->osize[i];
	}

	for (int b = 0; b < this->osize[0]; ++b)
	{
		for (int i = 0; i < size; ++i)
		{
			l += (1.0 / (2 * this->osize[0])) * pow(this->out[idx(b, i, size)] - target[idx(b, i, size)], 2);
		}
	}

	return l;
}

float *Network::loss_grad(float *target)
{
	int size = 1;

	for (int i = 1; i < this->odim; ++i)
	{
		size *= this->osize[i];
	}

	float *grad = new float[osize[0] * size];

	for (int b = 0; b < this->osize[0]; ++b)
	{
		for (int i = 0; i < size; ++i)
		{
			grad[idx(b, i, size)] = (1.0 / this->osize[0]) * (this->out[idx(b, i, size)] - target[idx(b, i, size)]);
		}
	}

	return grad;
}

float *Conv::forward_prop(float *inp, int idim, int *isize,
						  int &odim, int *&osize)
{
	if (idim != 4)
		cout << "Error : Input to Conv must be 4D\n";

	this->inp = inp;

	odim = this->odim;
	osize = this->osize;

	for (int b = 0; b < osize[0]; b++)
	{
		for (int f = 0; f < osize[1]; f++)
		{
			for (int d = 0; d < isize[1]; d++)
			{
				for (int i = 0; i < osize[2]; i++)
				{
					for (int j = 0; j < osize[3]; j++)
					{
						for (int i1 = 0; i1 < wsize[2]; i1++)
						{
							for (int j1 = 0; j1 < wsize[3]; j1++)
							{
								out[idx(b, f, i, j, osize[1], osize[2], osize[3])] +=
									inp[idx(b, d, i * strH + i1, j * strW + j1, isize[1], isize[2], isize[3])] *
										weights[idx(f, d, i1, j1, wsize[1], wsize[2], wsize[3])] +
									bias[idx(f, d, bsize[1])];
							}
						}
					}
				}
			}
		}
	}

	return out;
}

void Conv::grad_weights(float *out_grad)
{
	// conv(inp, out_grad);
	memset(weight_grad, 0, wsize[0] * wsize[1] * wsize[2] * wsize[3] * sizeof(float));

	memset(bias_grad, 0, bsize[0] * bsize[1] * sizeof(float));

	for (int b = 0; b < osize[0]; b++)
	{
		for (int f = 0; f < osize[1]; f++)
		{
			for (int d = 0; d < isize[1]; d++)
			{

				for (int i = 0; i < wsize[2]; i++)
				{
					for (int j = 0; j < wsize[3]; j++)
					{
						for (int i1 = 0; i1 < osize[2]; i1++)
						{
							for (int j1 = 0; j1 < osize[3]; j1++)
							{
								weight_grad[idx(f, d, i, j, wsize[1], wsize[2], wsize[3])] +=
									inp[idx(b, d, i + i1, j + j1, isize[1], isize[2], isize[3])] *
									out_grad[idx(b, f, i1, j1, osize[1], osize[2], osize[3])];

								bias_grad[idx(f, d, bsize[1])] += out_grad[idx(b, f, i1, j1, osize[1], osize[2], osize[3])];
							}
						}
					}
				}
			}
		}
	}
}

float *Conv::grad_inputs(float *out_grad)
{
	// conv(out_grad, weights);
	memset(inp_grad, 0, isize[0] * isize[1] * isize[2] * isize[3] * sizeof(float));

	for (int b = 0; b < osize[0]; b++)
	{
		for (int f = 0; f < osize[1]; f++)
		{
			for (int d = 0; d < isize[1]; d++)
			{

				for (int i = 0; i < isize[2]; i++)
				{
					for (int j = 0; j < isize[3]; j++)
					{
						for (int i1 = 0; i1 < wsize[2]; i1++)
						{
							for (int j1 = 0; j1 < wsize[3]; j1++)
							{
								if (i - wsize[2] + 1 + i1 < 0 || i - wsize[2] + 1 + i1 > osize[2])
									continue;
								if (j - wsize[3] + 1 + j1 < 0 || j - wsize[3] + 1 + j1 > osize[3])
									continue;
								inp_grad[idx(b, d, i, j, isize[1], isize[2], isize[3])] +=
									out_grad[idx(b, f, i - wsize[2] + 1 + i1, j - wsize[3] + 1 + j1, osize[1], osize[2], osize[3])] *
									weights[idx(f, d, i1, j1, wsize[1], wsize[2], wsize[3])];
							}
						}
					}
				}
			}
		}
	}

	return inp_grad;
}

void Conv::init(int idim, int *isize, int &odim, int *&osize)
{
	this->odim = 4;
	this->osize = new int[this->odim];
	this->osize[0] = isize[0];
	this->osize[1] = omaps;
	this->osize[2] = (isize[2] - fH) / strH + 1;
	this->osize[3] = (isize[3] - fW) / strW + 1;

	odim = this->odim;
	osize = this->osize;

	this->out = new float[osize[0] * osize[1] * osize[2] * osize[3]];
	memset(this->out, 0, sizeof(float) * osize[0] * osize[1] * osize[2] * osize[3]);

	this->idim = idim;
	this->isize = isize;

	this->inp_grad = new float[isize[0] * isize[1] * isize[2] * isize[3]];

	this->wdim = 4;
	this->wsize = new int[wdim];
	this->wsize[0] = omaps;
	this->wsize[1] = isize[1];
	this->wsize[2] = fH;
	this->wsize[3] = fW;

	this->weight_grad = new float[wsize[0] * wsize[1] * wsize[2] * wsize[3]];

	this->bdim = 4;
	this->bsize = new int[wdim];
	this->bsize[0] = omaps;
	this->bsize[1] = isize[1];
	this->bias_grad = new float[bsize[0] * bsize[1]];

	this->random_init();
}

float *Pool::forward_prop(float *inp, int idim, int *isize,
						  int &odim, int *&osize)
{
	if (idim != 4)
		cout << "Error : Input to Pool must be 4D\n";

	this->inp = inp;

	odim = this->odim;
	osize = this->osize;

	for (int b = 0; b < osize[0]; b++)
	{
		for (int d = 0; d < osize[1]; d++)
		{
			for (int i = 0; i < osize[2]; i++)
			{
				for (int j = 0; j < osize[3]; j++)
				{
					// Assign it to the first value and check if there is anything greater
					float temp = inp[idx(b, d, i * fH, j * fW, isize[1], isize[2], isize[3])];

					for (int i1 = 0; i1 < fH; i1++)
					{
						for (int j1 = 0; j1 < fW; j1++)
						{
							temp = max(temp, inp[idx(b, d, i * fH + i1, j * fW + j1, isize[1], isize[2], isize[3])]);
						}
					}
					out[idx(b, d, i, j, osize[1], osize[2], osize[3])] = temp;
				}
			}
		}
	}

	return out;
}

void Pool::grad_weights(float *out_grad)
{
	weight_grad[0] = 0;
	bias_grad[0] = 0;
}

float *Pool::grad_inputs(float *out_grad)
{
	for (int b = 0; b < osize[0]; b++)
	{
		for (int d = 0; d < osize[1]; d++)
		{
			for (int i = 0; i < osize[2]; i++)
			{
				for (int j = 0; j < osize[3]; j++)
				{
					// Assign it to the first value and check if there is anything greater
					float temp = inp[idx(b, d, i * fH, j * fW, isize[1], isize[2], isize[3])];
					float max_i1 = 0, max_j1 = 0;

					for (int i1 = 0; i1 < fH; i1++)
					{
						for (int j1 = 0; j1 < fW; j1++)
						{
							inp_grad[idx(b, d, i * fH + i1, j * fW + j1, isize[1], isize[2], isize[3])] = 0.0f;

							if (temp < inp[idx(b, d, i * fH + i1, j * fW + j1, isize[1], isize[2], isize[3])])
							{
								temp = inp[idx(b, d, i * fH + i1, j * fW + j1, isize[1], isize[2], isize[3])];
								max_i1 = i1;
								max_j1 = j1;
							}
						}
					}
					inp_grad[idx(b, d, i * fH + max_i1, j * fW + max_j1, isize[1], isize[2], isize[3])] =
						out_grad[idx(b, d, i, j, osize[1], osize[2], osize[3])];
				}
			}
		}
	}
	return inp_grad;
}

void Pool::init(int idim, int *isize, int &odim, int *&osize)
{
	this->odim = 4;
	this->osize = new int[this->odim];
	this->osize[0] = isize[0];
	this->osize[1] = isize[1];
	this->osize[2] = isize[2] / fH;
	this->osize[3] = isize[3] / fW;

	odim = this->odim;
	osize = this->osize;

	this->out = new float[osize[0] * osize[1] * osize[2] * osize[3]];

	this->idim = idim;
	this->isize = isize;

	this->inp_grad = new float[isize[0] * isize[1] * isize[2] * isize[3]];

	this->wdim = 0;

	this->weight_grad = new float[1];

	this->bdim = 0;

	this->bias_grad = new float[1];

	random_init();
}

float *ReLU::forward_prop(float *inp, int idim, int *isize,
						  int &odim, int *&osize)
{
	this->inp = inp;

	odim = this->odim;
	osize = this->osize;

	int size = 1;

	for (int i = 0; i < odim; ++i)
	{
		size *= this->osize[i];
	}

	for (int i = 0; i < size; ++i)
	{
		out[i] = max(0.0f, inp[i]);
	}

	return out;
}

void ReLU::grad_weights(float *out_grad)
{
	weight_grad[0] = 0;
	bias_grad[0] = 0;
}

float *ReLU::grad_inputs(float *out_grad)
{
	int size = 1;
	for (int i = 0; i < this->idim; ++i)
		size *= this->isize[i];

	for (int i = 0; i < size; ++i)
	{
		if (inp[i] >= 0)
			inp_grad[i] = out_grad[i];
		else
			inp_grad[i] = 0;
	}

	return inp_grad;
}

void ReLU::init(int idim, int *isize, int &odim, int *&osize)
{

	this->odim = idim;
	this->osize = new int[this->odim];

	for (int i = 0; i < odim; ++i)
		this->osize[i] = isize[i];

	odim = this->odim;
	osize = this->osize;

	int size = 1;

	for (int i = 0; i < odim; ++i)
	{
		size *= osize[i];
	}

	this->out = new float[size];

	this->idim = idim;
	this->isize = isize;

	size = 1;
	for (int i = 0; i < this->idim; ++i)
		size *= this->isize[i];

	this->inp_grad = new float[size];

	this->wdim = 0;
	this->weight_grad = new float[1];

	this->bdim = 0;
	this->bias_grad = new float[1];

	random_init();
}

float *FullyConnected::forward_prop(float *inp, int idim, int *isize,
									int &odim, int *&osize)
{
	this->inp = inp;

	odim = this->odim;
	osize = this->osize;
	for (int b = 0; b < osize[0]; ++b)
	{
		for (int i = 0; i < wsize[0]; ++i)
		{
			out[idx(b, i, wsize[0])] = bias[i];
			for (int j = 0; j < wsize[1]; ++j)
			{
				out[idx(b, i, wsize[0])] += inp[idx(b, j, wsize[1])] * weights[idx(i, j, wsize[1])];
			}
		}
	}
	return out;
}

void FullyConnected::grad_weights(float *out_grad)
{

	for (int i = 0; i < wsize[0]; ++i)
	{
		for (int j = 0; j < wsize[1]; ++j)
		{
			weight_grad[idx(i, j, wsize[1])] = 0.0f;
		}
	}

	for (int b = 0; b < osize[0]; ++b)
	{
		for (int i = 0; i < wsize[0]; ++i)
		{
			for (int j = 0; j < wsize[1]; ++j)
			{
				weight_grad[idx(i, j, wsize[1])] += out_grad[idx(b, i, wsize[0])] * inp[idx(b, j, wsize[1])];
			}
		}
	}

	for (int i = 0; i < bsize[0]; ++i)
		bias_grad[i] = 0.0f;

	for (int b = 0; b < osize[0]; ++b)
	{
		for (int i = 0; i < wsize[0]; ++i)
		{
			bias_grad[i] += out_grad[idx(b, i, wsize[0])];
		}
	}
}

float *FullyConnected::grad_inputs(float *out_grad)
{

	int size = 1;
	for (int i = 0; i < this->idim; ++i)
		size *= this->isize[i];

	for (int i = 0; i < size; ++i)
		inp_grad[i] = 0.0f;

	for (int b = 0; b < osize[0]; ++b)
	{
		for (int i = 0; i < wsize[1]; ++i)
		{
			for (int j = 0; j < wsize[0]; ++j)
			{
				inp_grad[idx(b, i, wsize[1])] += weights[idx(j, i, wsize[1])] * out_grad[idx(b, j, wsize[0])];
			}
		}
	}
	return inp_grad;
}

void FullyConnected::init(int idim, int *isize, int &odim, int *&osize)
{
	this->odim = 2;
	this->osize = new int[this->odim];
	this->osize[0] = isize[0];
	this->osize[1] = num_out;

	odim = this->odim;
	osize = this->osize;

	this->out = new float[osize[0] * osize[1]];

	this->idim = idim;
	this->isize = isize;

	int num_inp = 1;
	for (int i = 1; i < idim; ++i)
		num_inp *= isize[i];

	this->inp_grad = new float[isize[0] * num_inp];

	this->wdim = 2;
	this->wsize = new int[wdim];
	this->wsize[0] = num_out;
	this->wsize[1] = num_inp;

	this->weight_grad = new float[num_out * num_inp];

	this->bdim = 1;
	this->bsize = new int[bdim];
	this->bsize[0] = num_out;

	this->bias_grad = new float[num_out];

	this->random_init();
}

float *Sigmoid::forward_prop(float *inp, int idim, int *isize,
							 int &odim, int *&osize)
{
	this->inp = inp;

	odim = this->odim;
	osize = this->osize;

	int size = 1;

	for (int i = 0; i < odim; ++i)
	{
		size *= osize[i];
	}

	for (int i = 0; i < size; ++i)
	{
		out[i] = 1.0 / (1.0 + exp(-inp[i]));
	}

	return out;
}

void Sigmoid::grad_weights(float *out_grad)
{
	weight_grad[0] = 0;
	bias_grad[0] = 0;
}

float *Sigmoid::grad_inputs(float *out_grad)
{
	int size = 1;
	for (int i = 0; i < this->idim; ++i)
		size *= this->isize[i];

	for (int i = 0; i < size; ++i)
	{
		inp_grad[i] = out[i] * (1 - out[i]) * out_grad[i];
	}

	return inp_grad;
}

void Sigmoid::init(int idim, int *isize, int &odim, int *&osize)
{
	this->odim = idim;
	this->osize = new int[this->odim];

	for (int i = 0; i < odim; ++i)
		this->osize[i] = isize[i];

	odim = this->odim;
	osize = this->osize;

	int size = 1;

	for (int i = 0; i < odim; ++i)
	{
		size *= osize[i];
	}

	this->out = new float[size];

	this->idim = idim;
	this->isize = isize;

	size = 1;
	for (int i = 0; i < this->idim; ++i)
		size *= this->isize[i];

	this->inp_grad = new float[size];

	this->wdim = 0;

	this->weight_grad = new float[1];

	this->bdim = 0;

	this->bias_grad = new float[1];

	random_init();
}