#include "Util.h"

void Util::ReLu(Elements &data) {
    for (int a = 0; a < data.getLength(); a++)
        if (data[a] < 0)
            data[a] *= ALPHA;
}

void Util::deLeReLu(Elements &data) {
    for (int a = 0; a < data.getLength(); a++)
        if (data[a] < 0)
            data[a] = ALPHA;
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
}
