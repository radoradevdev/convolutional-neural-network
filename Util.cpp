#include "Util.h"

#include <QTextStream>

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

QImage Util::elementsToQImage(const Elements &image) {
    int height = image.getParam(1);
    int width = image.getParam(2);

    QImage out(height, width, QImage::Format_Grayscale16);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int in_cache[3] = { 0, y, x };
            double val = image.getValue(in_cache, 3);
            // Convert the pixel value to grayscale
            int grayValue = static_cast<int>(val * 255);

            // Create a grayscale color
            QRgb grayColor = qRgb(grayValue, grayValue, grayValue);

            QPoint point(y, x);
            out.setPixel(point, grayColor);
        }
    }

    return out;
}

void Util::printQImage(const QImage &image) {
    // Iterate over the pixels of the image
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            // Get the pixel value at coordinates (x, y)
            QRgb pixelValue = image.pixel(x, y);

            // Extract the grayscale value
            int grayValue = qRed(pixelValue);


            // Print the grayscale value
            QTextStream(stdout) << "Grayscale value at (" << x << ", " << y << "):" << grayValue;
        }
    }
}

QImage Util::grayscaleToHeatmap(const QImage &grayscaleImage) {
    QImage heatmapImage(grayscaleImage.size(), QImage::Format_RGB32); // Creating a new QImage for the heatmap

    // Define the color map
    QVector<QRgb> colorMap;
    colorMap << qRgb(0, 128, 0);    // Green
    colorMap << qRgb(150, 0, 7);
    colorMap << qRgb(143, 0, 16);
    colorMap << qRgb(136, 0, 25);
    colorMap << qRgb(129, 0, 34);
    colorMap << qRgb(123, 0, 43);
    colorMap << qRgb(116, 0, 52);
    colorMap << qRgb(109, 0, 61);
    colorMap << qRgb(102, 0, 70);
    colorMap << qRgb(96, 0, 79);
    colorMap << qRgb(89, 0, 88);
    colorMap << qRgb(82, 0, 97);
    colorMap << qRgb(76, 0, 106);
    colorMap << qRgb(70, 0, 115);
    colorMap << qRgb(63, 0, 125);   // Dark Purple

    // Iterate over the pixels of the grayscale image
    for (int y = 0; y < grayscaleImage.height(); ++y) {
        for (int x = 0; x < grayscaleImage.width(); ++x) {
            // Get the grayscale value at coordinates (x, y)
            int grayValue = qRed(grayscaleImage.pixel(x, y));

            // Map the grayscale value to a color index
            int colorIndex = grayValue * (colorMap.size() - 1) / 255;

            // Set the corresponding color to the pixel in the heatmap image
            heatmapImage.setPixel(x, y, colorMap[colorIndex]);
        }
    }

    return heatmapImage;
}
