#ifndef UTIL_H
#define UTIL_H

// Hyperparameter for the relu function
#define ALPHA 0.001
#define DIMS 4

#include "Elements.h"

#include <QImage>

//! Utility class
/*!
  The class contains static helper functions
*/
class Util {
public:
    // Activation functions

    //! Leaky ReLU stands for Leaky Rectified Linear Unit,
    //! which is a type of activation function used in neural networks.
    //! It is defined as the function f(x) = max(alpha*x, x),
    //! where x is the input to the function.
    /*!
      \param data   elements data
    */
    static void ReLu(Elements &data);

    //! Neutralizes all values with the ALPHA value
    /*!
      \param data   elements data
    */
    static void deLeReLu(Elements &data);

    // Dataset helpers

    //! Converts an integer i into its equivalent 4-byte
    //! representation as a sequence of unsigned characters
    static int reverseInt(int i);

    //! Generates a random float number
    static double frand();

    //! Converts Elements to QImage
    static QImage elementsToQImage(const Elements &image);

    //! Prints QImage
    static void printQImage(const QImage &image);

    //! Converts Grayscale image to heatmap
    static QImage grayscaleToHeatmap(const QImage &grayscaleImage);
};

#endif // UTIL_H
