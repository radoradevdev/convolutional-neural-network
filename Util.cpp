#include "Util.h"

void Util::ReLu(Elements &input_volume) {
    for (int a = 0; a < input_volume.get_length(); a++)
        if (input_volume[a] < 0)
            input_volume[a] *= ALPHA;
}

void Util::deLeReLu(Elements &input_volume) {
    for (int a = 0; a < input_volume.get_length(); a++)
        if (input_volume[a] < 0)
            input_volume[a] = ALPHA;
}

int Util::reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

double Util::frand() {
    return (double)(arc4random() % 100) / 1000;
    // return (2.0*(double)rand() / RAND_MAX) - 1.0;
}
