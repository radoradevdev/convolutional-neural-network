#include "carteduciel.h"

#include <QDir>
#include <QImage>
#include <QTextStream>

void CarteDuCiel::loadDataset(
        string path,
        Elements &set,
        vector<int> &values,
        int length
        ) {
    try {
        // Create a QDir object to access the folder
        QDir dir(QString::fromStdString(path));

        // Filter for image files
        QStringList filters;
        filters << "*.tiff" << "*.tif";
        dir.setNameFilters(filters);
        dir.setSorting(QDir::Name | QDir::IgnoreCase);

        // Retrieve all image files in the folder
        QFileInfoList fileList = dir.entryInfoList();

        std::sort(fileList.begin(), fileList.end(), [](const QFileInfo &a, const QFileInfo &b) {
            return a.baseName().toInt() < b.baseName().toInt();
        });

        int indx_img = 0;
        // Iterate through the image files and map them to QImage objects
        foreach (const QFileInfo &fileInfo, fileList) {
            if(indx_img == length) {
                break;
            }
            QString filePath = fileInfo.absoluteFilePath();
            QImage image;
            if (!image.load(filePath)) {
                // Failed to load image, handle error
                throw std::runtime_error("Failed to load image:" + filePath.toStdString());
            } else {
                for (int indx_row = 0; indx_row < IMAGE_SIDE; ++indx_row) {
                    quint8* ptr_row = (quint8*)(image.bits() + indx_row * image.bytesPerLine());
                    for (int indx_col = 0; indx_col < IMAGE_SIDE; ++indx_col) {
                        quint8 temp = ptr_row[indx_col];
                        int index[4] = {indx_img, 0, indx_row, indx_col};
                        set.assign((double)temp, index, DIMS);
                    }
                }

                QString baseName = fileInfo.baseName();
                int firstUnderscoreIndex = baseName.indexOf("_");
                int expectedValue = baseName[firstUnderscoreIndex+1].digitValue();
                if(expectedValue < 1 || expectedValue > 5) {
                    throw std::runtime_error("no expected value");
                }
                if(expectedValue > 2) {
                    values[indx_img] = expectedValue - 2;
                } else {
                    values[indx_img] = expectedValue - 1;
                }
                indx_img++;
            }
        }

        QTextStream(stdout) << "\n\tDatasets loaded" << Qt::endl;
    } catch(const std::exception& e) {
        QTextStream(stderr) << "The dataset could not be found." << Qt::endl;
    }
}

void CarteDuCiel::initDataset(Elements &Train_DS, vector<int> &Train_EV,
                        Elements &Test_DS, vector<int> &Test_EV) {

    int train_shapes[4] = { CARTEDUCIEL_TRAIN_LEN, 1, IMAGE_SIDE, IMAGE_SIDE };
    int test_shapes[4]  = { CARTEDUCIEL_TEST_LEN, 1, IMAGE_SIDE, IMAGE_SIDE };

    Train_DS.init(train_shapes, 4);
    Test_DS.init(test_shapes, 4);

    Train_EV.assign(CARTEDUCIEL_TRAIN_LEN, 0);
    Test_EV.assign(CARTEDUCIEL_TEST_LEN, 0);
}

void CarteDuCiel::getDataset(Elements &Train_DS, vector<int> &Train_EV,
                       Elements &Test_DS, vector<int> &Test_EV ) {

    initDataset(Train_DS, Train_EV, Test_DS, Test_EV);

    QTextStream(stdout) << "\n> Loading CarteDuCiel datasets" << Qt::endl;

    string path_to_folder = "/Users/radoslavradev/Sites/bachelor-thesis/"
                            "bachelor-thesis-source-code/CDCA1_data/";

    loadDataset(path_to_folder + "/ROBO33_000008_train", Train_DS, Train_EV, CARTEDUCIEL_TRAIN_LEN);

    loadDataset(path_to_folder + "/ROBO33_000008_test", Test_DS, Test_EV, CARTEDUCIEL_TEST_LEN);

    normalizeSet(Train_DS, CARTEDUCIEL_TRAIN_LEN, IMAGE_SIDE, IMAGE_SIDE);
    normalizeSet(Test_DS, CARTEDUCIEL_TEST_LEN, IMAGE_SIDE, IMAGE_SIDE);
}

// Should require also the depth value
void CarteDuCiel::normalizeSet(Elements &set, int length, int width, int height) {

    for (int indx_img = 0; indx_img < length; indx_img++) {

        // Get max and min values
        double max = 0, min = 255, val;

        for (int indx_row = 0; indx_row < width; ++indx_row) {
            for (int indx_col = 0; indx_col < height; ++indx_col) {
                int index[4] = { indx_img, 0, indx_row, indx_col };
                val = set.getValue(index, DIMS);

                if (val > max)
                    max = val;
                if (val < min)
                    min = val;
            }
        }

        // Normalize every value
        for (int indx_row = 0; indx_row < width; ++indx_row) {
            for (int indx_col = 0; indx_col < height; ++indx_col) {
                int index[4] = { indx_img, 0, indx_row, indx_col };
                val = set.getValue(index, DIMS);

                val = (val - min) / (max - min);

                set.assign(val, index, DIMS);
            }
        }
    }
}
