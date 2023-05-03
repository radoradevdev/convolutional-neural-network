#include <QApplication>
#include <vector>

#include <Network.h>

using namespace std;

// Hyperparameters
bool adam;
double bias, roc;
vector<int> conv_1{1, 28, 28}, conv_kernels_1{8, 3, 3, 1};
vector<int> conv_2{8, 13, 13}, conv_kernels_2{2, 3, 3, 8}, hidden{72};
int input_layer, num_classes, epochs, padding, stride;

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    // network initialization
    Network network;

    // configure the network
    network.addConvolutionalLayer(conv_1, conv_kernels_1, padding = 0, stride = 2,
                                  bias = 0.1, roc = 0.01);

    network.addConvolutionalLayer(conv_2, conv_kernels_2, padding = 0, stride = 2,
                                  bias = 0.1, roc = 0.01);

    network.addFullyConnectedLayer(input_layer = 2 * 6 * 6, hidden,
                                   num_classes = 4, bias = 1.0, adam = false,
                                   roc = 0.5);

    // load the dataset
    network.loadDataset(DatasetType::CARTEDUCIEL);

    // train the network
    network.train(epochs = 1, 100, false);

    network.plotResults();

    // test the network, with another set
    // network.test(100);

    // network.plotResults();

    QTextStream(stdout) << "End";

    return a.exec();
}
