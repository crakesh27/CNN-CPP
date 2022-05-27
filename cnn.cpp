#include <iostream>

#include "layers.h"
#include "load_image.h"

using namespace std;

int main(int argc, char **argv)
{
	int idim, *isize;
	int odim, *osize;

	char data_fname[] = "MNIST/train-images-idx3-ubyte";
	float *inp = load_mnist(data_fname, idim, isize);

	char target_fname[] = "MNIST/train-labels-idx1-ubyte";
	float *target = load_targets(target_fname, odim, osize);

	Network n;

	// n.add_layer(new Conv(5, 5, 16));
	// n.add_layer(new ReLU());

	// n.add_layer(new Pool(2, 2));

	n.add_layer(new FullyConnected(100));
	n.add_layer(new Sigmoid());

	n.add_layer(new FullyConnected(10));
	n.add_layer(new Sigmoid());

	n.train(inp, idim, isize, target, 20, 10, 0.5);

	free(isize);
	free(osize);

	char test_fname[] = "MNIST/t10k-images-idx3-ubyte";
	float *test = load_mnist(test_fname, idim, isize);

	char test_target_fname[] = "MNIST/t10k-labels-idx1-ubyte";
	float *test_tar = load_targets(test_target_fname, odim, osize);

	n.predict(test, idim, isize, test_tar, 10);

	return 0;
}