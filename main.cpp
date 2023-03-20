#include <QCoreApplication>
#include <vector>

#include <Network.h>

using namespace std;

bool adam;
double bias, eta;
vector<int> image_1{1, 28, 28}, kernels_1{8, 3, 3, 1};
vector<int> image_2{8, 13, 13}, kernels_2{2, 3, 3, 8}, hidden{72};
int input_layer, num_classes, epochs, padding, stride;

int main(int argc, char *argv[]) {
    QCoreApplication a(argc, argv);

    // network initialization
    Network network;

    // configure the network
    network.addConvolutionalLayer(image_1, kernels_1, padding = 0, stride = 2,
                                  bias = 0.1, eta = 0.01);

    network.addConvolutionalLayer(image_2, kernels_2, padding = 0, stride = 2,
                                  bias = 0.1, eta = 0.01);

    network.addFullyConnectedLayer(input_layer = 2 * 6 * 6, hidden,
                                   num_classes = 10, bias = 1.0, adam = false,
                                   eta = 0.5);

    // load the dataset
    network.loadDataset(DatasetType::MNIST);

    // sanity check
    network.checkConfiguration();

    // train the network
    network.train(epochs = 1, 10);

    // test the network, with another set
    network.test(10);

    QTextStream(stdout) << "End";

    return a.exec();
}
