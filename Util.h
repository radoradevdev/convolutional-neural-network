#ifndef UTIL_H
#define UTIL_H

#define ALPHA 0.001
#define DIMS 4

#include "Elements.h"

//! Utility class
/*!
  The class contains static helper functions
*/
class Util {
public:
    // Activation functions

    //! ReLU stands for Rectified Linear Unit,
    //! which is a type of activation function used in neural networks.
    //! It is defined as the function f(x) = max(0, x),
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

    //! TODO
    static int reverseInt(int i);

    //! Generates a random float number
    static double frand();
};

#endif // UTIL_H
