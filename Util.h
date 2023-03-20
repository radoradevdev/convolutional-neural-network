#ifndef UTIL_H
#define UTIL_H

#define ALPHA 0.001
#define DIMS 4

#include "Elements.h"

class Util {
public:
    // Activation functions
    static void ReLu(Elements &input_volume);
    static void deLeReLu(Elements &input_volume);

    // Dataset helpers
    static int reverseInt(int i);
    static double frand();
};

#endif // UTIL_H
