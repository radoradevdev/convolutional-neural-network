#include "Elements.h"

#include <QTextStream>

Elements::Elements(int height, int width) {

    _data.resize(height * width);
    _length = height * width;
    _params_length = 2;
    _params.resize(_params_length);
    _params[0] = height;
    _params[1] = width;
}

Elements::Elements(int height, int width, int depth) {

    _data.resize(height * width * depth, 0);
    _length = height * width * depth;
    _params_length = 3;
    _params.resize(_params_length);
    _params[0] = height;
    _params[1] = width;
    _params[2] = depth;
}

Elements::Elements(int layers, int height, int width, int depth) {

    _data.resize(layers * height * width * depth);
    _length = layers * height * width * depth;
    _params_length = 4;
    _params.resize(_params_length);
    _params[0] = layers;
    _params[1] = height;
    _params[2] = width;
    _params[3] = depth;
}

Elements::Elements(int *params, int length) {

    _length = 1;
    _params_length = length;
    _params.assign(params, params + length);

    for (int indx = 0; indx < length; indx++)
        _length *= params[indx];

    _data.resize(_length);
}

void Elements::init(const int *params, int length) {
    if (_data.size() != 0) {
        throw std::runtime_error("Data already set");
    } else {
        _init(params, length);

        _data.resize(_length);
    }
}

void Elements::reinit(int *params, int length) {
    _init(params, length);

    _data.assign(_length, 0);
}

void Elements::_init(const int *params, int length) {
    _length = 1;
    _params_length = length;
    _params.assign(params, params + length);

    for (int indx = 0; indx < length; indx++) {
        _length *= params[indx];
    }
}

int Elements::_find(int *indices, int length) const {
    int element = 0, offset = 1;

    for (int indx = 0; indx < length; indx++) {
        offset = 1;
        for (int params_indx = 0; params_indx < indx; params_indx++) {
            offset *= _params[params_indx];
        }

        element += indices[indx] * offset;
    }

    return element;
}

void Elements::allocate(double val, int *indices, int length) {

    if (length != _params_length) {
        throw std::runtime_error("Incorrect params length");
    } else {
        // When _params_length = 4:
        // _data[l + h*l + w*l*h + d*l*h*w]

        // When _params_length = 3:
        // _data[h + w*h + d*h*w]

        // When _params_length = 2:
        // _data[h + w*h]
        int element = _find(indices, length);

        if(element >= _data.size()) {
            throw std::runtime_error("Too large index for the data vector!");
        }
        _data[element] = val;
    }
}

double Elements::getValue(int *indices, int params_length) const {
    double res = -1;

    if (params_length != _params_length) {
        throw std::runtime_error("Incorrect params length");
    } else {
        // When _params_length = 4:
        // _data[l + h*l + w*l*h + d*l*h*w]

        // When _params_length = 3:
        // _data[h + w*h + d*h*w]

        // When _params_length = 2:
        // _data[h + w*h]

        int element = _find(indices, params_length);

        res = _data[element];
    }

    return res;
}

void Elements::aggregate(double val, int *indices, int length) {
    if (length != _params_length) {
        throw std::runtime_error("Incorrect params length");
    } else {
        // When _params_length = 4:
        // _data[l + h*l + w*l*h + d*l*h*w]

        // When _params_length = 3:
        // _data[h + w*h + d*h*w]

        // When _params_length = 2:
        // _data[h + w*h]
        int element = _find(indices, length);

        if(element >= _data.size()) {
            throw std::runtime_error("Too large index for the data vector!");
        }
        _data[element] += val;
    }
}

int Elements::getParam(int index) const {
    return _params[index];
}

int Elements::getLength() {
    return _length;
}

vector<double> &Elements::getData() {
    return _data;
}

Elements &Elements::operator=(const Elements &elements) {
    if (this == &elements) {
        return *this;
    } else {
        _data.resize(0), _params.resize(0);
        _params_length = 0, _length = 0;

        this->init(&(elements._params[0]), elements._params_length);

        this->_data = elements._data;
    }

    return *this;
}


double &Elements::operator[](int index) {
    if (index >= _length) {
        throw std::runtime_error("Index out of bound");
        return _data.back();
    }

    return _data[index];
}
