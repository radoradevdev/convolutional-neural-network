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
    //! Get the dataset
    /*!
      \param Train_DS   training dataset
      \param Train_EV   training expected values
      \param Valid_DS   validation dataset
      \param Valid_EV   validation expected values
      \param Test_DS    testing dataset
      \param Test_EV    testing expected values
    */
    void getDataset(
            Elements &Train_DS,
            vector<int> &Train_EV,
            Elements &Valid_DS,
            vector<int> &Valid_EV,
            Elements &Test_DS,
            vector<int> &Test_EV
            );

private:
    //! Load dataset from binary files
    /*!
      \param path       path to binary file
      \param set        dataset
      \param valid_set  validation dataset, depends on hasValid
      \param hasValid   divide in training and validating?
    */
    void loadDataset(
            string path,
            Elements &set,
            Elements &valid_set,
            bool hasValid = false
            );

    //! Load expected values for dataset from binary files
    /*!
      \param path           path to binary file
      \param values         expected values
      \param valid_values   validation expected values, depends on hasValid
      \param hasValid       divide in training and validating?
    */
    void loadExpectedValues(
            string path,
            vector<int> &values,
            vector<int> &valid_values,
            bool hasValid = false
            );

    //! Initialze the dataset with its properties
    /*!
      \param Train_DS   training dataset
      \param Train_EV   training expected values
      \param Valid_DS   validation dataset
      \param Valid_EV   validation expected values
      \param Test_DS    testing dataset
      \param Test_EV    testing expected values
    */
    void initDataset(
            Elements &Train_DS,
            vector<int> &Train_EV,
            Elements &Valid_DS,
            vector<int> &Valid_EV,
            Elements &Test_DS,
            vector<int> &Test_EV);

    //! Normaliza a set
    /*!
      \param set    dataset
      \param length length of dataset
      \param width  width of images
      \param height height of images
    */
    void normalizeSet(
            Elements &set,
            int length,
            int width,
            int height
            );
};

#endif
