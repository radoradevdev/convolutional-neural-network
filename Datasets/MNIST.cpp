#include "MNIST.h"

#include <QTextStream>

void MNIST::loadExpectedValues(
        string path,
        vector<int> &values,
        vector<int> &valid_values,
        bool hasValid
        ) {

    ifstream file(path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0; /**< magic number is a sequence of
                                bytes at the beginning of a file that indicates
                                its file type or format. */
        int number_of_images = 0;

        // Reads and performs byte order reversal
        // using a utility function to ensure correct interpretation of the data
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = Util::reverseInt(magic_number);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = Util::reverseInt(number_of_images);

        // divide the training the dataset into training and validation sets
        int len = hasValid ? number_of_images - MNIST_VALID_LEN : number_of_images;
        for (int indx_val = 0; indx_val < len; ++indx_val) {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            values[indx_val] = (int)temp;
        }

        // Also load the validation set
        if (hasValid) {
            for (int indx_val = len+1; indx_val < number_of_images; ++indx_val) {
                unsigned char temp = 0;
                file.read((char *)&temp, sizeof(temp));
                valid_values[indx_val] = (int)temp;
            }
        }
    }
}

void MNIST::loadDataset(
        string path,
        Elements &set,
        Elements &valid_set,
        bool hasValid
        ) {

    // Load the dataset and fill the Elements (_DS and _EV)
    ifstream file(path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0; /**< magic number is a sequence of
                                bytes at the beginning of a file that indicates
                                its file type or format. */
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        // Reads and performs byte order reversal
        // using a utility function to ensure correct interpretation of the data
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = Util::reverseInt(magic_number);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = Util::reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = Util::reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = Util::reverseInt(n_cols);

        int len = hasValid ? number_of_images - MNIST_VALID_LEN : number_of_images;
        for (int indx_img = 0; indx_img < len; ++indx_img) {
            for (int indx_row = 0; indx_row < n_rows; ++indx_row) {
                for (int indx_col = 0; indx_col < n_cols; ++indx_col) {
                    unsigned char temp = 0;
                    file.read((char *)&temp, sizeof(temp));
                    int index[4] = {indx_img, 0, indx_row, indx_col};
                    set.allocate((double)temp, index, DIMS);
                }
            }
        }

        // Also load the validation set
        if (hasValid) {
            for (int indx_img = len+1; indx_img < number_of_images; ++indx_img) {
                for (int indx_row = 0; indx_row < n_rows; ++indx_row) {
                    for (int indx_col = 0; indx_col < n_cols; ++indx_col) {
                        unsigned char temp = 0;
                        file.read((char *)&temp, sizeof(temp));
                        int index[4] = {indx_img, 0, indx_row, indx_col};
                        valid_set.allocate((double)temp, index, DIMS);
                    }
                }
            }
        }

        QTextStream(stdout) << "\n\tDatasets loaded" << Qt::endl;
    } else {
        throw std::runtime_error("The dataset could not be found.");
    }
}

void MNIST::initDataset(Elements &Train_DS, vector<int> &Train_EV,
                        Elements &Valid_DS, vector<int> &Valid_EV,
                        Elements &Test_DS,  vector<int> &Test_EV
                        ) {

    int train_images[4] = { MNIST_TRAIN_LEN, 1, IMAGE_SIDE, IMAGE_SIDE };
    int valid_images[4] = { MNIST_VALID_LEN, 1, IMAGE_SIDE, IMAGE_SIDE };
    int test_images[4]  = { MNIST_TEST_LEN,  1, IMAGE_SIDE, IMAGE_SIDE };

    Train_DS.init(train_images, 4);
    Valid_DS.init(valid_images, 4);
    Test_DS.init(test_images, 4);

    Train_EV.assign(MNIST_TRAIN_LEN, 0);
    Valid_EV.assign(MNIST_VALID_LEN, 0);
    Test_EV.assign(MNIST_TEST_LEN, 0);
}

void MNIST::getDataset(Elements &Train_DS, vector<int> &Train_EV,
                       Elements &Valid_DS, vector<int> &Valid_EV,
                       Elements &Test_DS,  vector<int> &Test_EV
                       ) {

    initDataset(Train_DS, Train_EV, Valid_DS, Valid_EV, Test_DS, Test_EV);

    QTextStream(stdout) << "\n> Loading MNIST dataset" << Qt::endl;

    string path_to_folder = "/Users/radoslavradev/Sites/bachelor-thesis/"
                            "bachelor-thesis-source-code/MNIST_data/";

    // load the datasets from the binary files
    loadDataset(path_to_folder + "train-images.idx3-ubyte", Train_DS, Valid_DS, true);
    loadExpectedValues(path_to_folder + "train-labels.idx1-ubyte", Train_EV, Valid_EV, true);

    loadDataset(path_to_folder + "t10k-images.idx3-ubyte", Test_DS, Test_DS);
    loadExpectedValues(path_to_folder + "t10k-labels.idx1-ubyte", Test_EV, Test_EV);

    // normalize the datasets
    normalizeSet(Train_DS, MNIST_TRAIN_LEN, IMAGE_SIDE, IMAGE_SIDE);
    normalizeSet(Valid_DS, MNIST_VALID_LEN, IMAGE_SIDE, IMAGE_SIDE);
    normalizeSet(Test_DS, MNIST_TEST_LEN, IMAGE_SIDE, IMAGE_SIDE);
}

void MNIST::normalizeSet(
        Elements &set,
        int length,
        int width,
        int height
        ) {

    for (int indx_img = 0; indx_img < length; indx_img++) {

        // Get max and min values
        double max = 0, min = 255, val;

        for (int indx_row = 0; indx_row < width; ++indx_row) {
            for (int indx_col = 0; indx_col < height; ++indx_col) {
                int index[4] = { indx_img, 0, indx_row, indx_col };
                val = set.getValue(index, DIMS);

                // get the min and max values in the whole image
                if (val > max)
                    max = val;
                if (val < min)
                    min = val;
            }
        }

        // Normalize every value with the min and max
        // (min-max normalization) or (feature scaling)
        for (int indx_row = 0; indx_row < width; ++indx_row) {
            for (int indx_col = 0; indx_col < height; ++indx_col) {
                int index[4] = { indx_img, 0, indx_row, indx_col };
                val = set.getValue(index, DIMS);

                val = (val - min) / (max - min);

                set.allocate(val, index, DIMS);
            }
        }
    }
}
