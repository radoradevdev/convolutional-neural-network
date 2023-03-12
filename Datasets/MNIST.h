#ifndef MNIST_H
#define MNIST_H

#define MNIST_TRAIN_LEN 50000
#define MNIST_VALID_LEN 10000
#define MNIST_TEST_LEN 10000
#define IMAGE_SIDE 28

#include <fstream>
#include <iostream>
#include <vector>

#include "Elements.h"
#include "Util.h"

using namespace std;

class MNIST {
public:
    void getDataset(Elements &Train_DS, vector<int> &Train_EV, Elements &Test_DS,
                   vector<int> &Test_EV, Elements &Valid_DS, vector<int> &Valid_EV);

private:
    void loadDataset(string path, Elements &set,
                     Elements &valid_set, bool hasValid = false);
    void loadExpectedValues(string path, vector<int> &values,
                            vector<int> &valid_values, bool hasValid = false);

    void initDataset(Elements &Train_DS, vector<int> &Train_EV, Elements &Test_DS,
                     vector<int> &Test_EV, Elements &Valid_DS, vector<int> &Valid_EV);
};

// Other datasets to be implemented

#endif
