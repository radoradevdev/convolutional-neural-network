#ifndef UTIL_H
#define UTIL_H

#define ALPHA 0.001
#define DIMS 4

#include "Elements.h"

class Util {
public:
    Util();

    // Activation functions
    static void ReLu(Elements &input_volume);
    static void deLeReLu(Elements &input_volume);

    // Dataset helpers
    static int reverseInt(int i);
    static void normalizeSet(Elements &set, int len, int n_rows, int n_cols);
    static double frand();
};

#endif // UTIL_H
