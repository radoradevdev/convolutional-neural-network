#ifndef CARTEDUCIEL_H
#define CARTEDUCIEL_H

// Here the parameters of the dataset must be set
#define CARTEDUCIEL_TRAIN_LEN 9600
#define CARTEDUCIEL_TEST_LEN 4800
#define IMAGE_SIDE 28

#include <fstream>
#include <iostream>
#include <vector>

#include "Elements.h"
#include "Util.h"

class CarteDuCiel
{
public:
    //! Get the dataset
    /*!
      \param Train_DS   training dataset
      \param Train_EV   training expected values
      \param Test_DS    testing dataset
      \param Test_EV    testing expected values
    */
    void getDataset(
            Elements &Train_DS,
            vector<int> &Train_EV,
            Elements &Test_DS,
            vector<int> &Test_EV
            );

private:
    //! Load dataset from folder
    /*!
      \param path       path to folder
      \param set        dataset
      \param values     expected values
      \param length     length of the dataset
    */
    void loadDataset(string path,
            Elements &set,
            vector<int> &values,
            int length
            );

    //! Initialze the dataset with its properties
    /*!
      \param Train_DS   training dataset
      \param Train_EV   training expected values
      \param Test_DS    testing dataset
      \param Test_EV    testing expected values
    */
    void initDataset(
            Elements &Train_DS,
            vector<int> &Train_EV,
            Elements &Test_DS,
            vector<int> &Test_EV
            );

    //! Normalise a set
    /*!
      \param set        dataset
      \param length     length of dataset
      \param width      width of images
      \param height     height of images
    */
    void normalizeSet(
            Elements &set,
            int length,
            int width,
            int height
            );
};

#endif // CARTEDUCIEL_H
