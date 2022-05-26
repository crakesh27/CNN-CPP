#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

float *load_mnist(char *fname, int &idim, int *&isize);

float *load_targets(char *fname, int &idim, int *&isize);