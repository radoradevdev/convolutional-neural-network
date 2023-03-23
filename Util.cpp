#include "Util.h"

void Util::ReLu(Elements &data) {
    for (int indx = 0; indx < data.getLength(); indx++) {
        if (data[indx] < 0) {
            data[indx] *= ALPHA;
        }
    }
}

void Util::deLeReLu(Elements &data) {
    for (int indx = 0; indx < data.getLength(); indx++) {
        if (data[indx] < 0) {
            data[indx] = ALPHA;
        }
    }
}

int Util::reverseInt(int i) {
    // Converts an integer i into its equivalent 4-byte representation
    // as a sequence of unsigned characters (ch1, ch2, ch3, and ch4)
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;          // the first-least significant byte of i
    ch2 = (i >> 8) & 255;   // the second-least significant byte of i
    ch3 = (i >> 16) & 255;  // the third-least significant byte of i
    ch4 = (i >> 24) & 255;  // the fourth-least significant byte of i

    // 4-byte representation as a single integer value, all shifted to their
    // correct position
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

double Util::frand() {
    // arc4random() generates random numbers that are uniformly
    // distributed across the entire range of possible values,
    // while rand may have biases that can cause some numbers
    // to be generated more frequently than others.
    return (double)(arc4random() % 100) / 1000;
}
