#ifndef ELEMENTS_H
#define ELEMENTS_H

#include <iostream>
#include <vector>

using namespace std;

class Elements {
public:
    Elements(); // Void volume object

    Elements(int H, int W);
    Elements(int H, int W, int Depth);
    Elements(int Layers, int H, int W, int Depth);
    Elements(int *shapes, int dimensions);

    void init(const int *shapes,
              int dimensions); // Usefull to postpone istantiation
    void rebuild(int *shapes, int dimensions);

    int get_shape(int dim_n);
    int get_length();
    double get_value(int *index, int dimensions);
    vector<double> &get_vector();

    void assign(double val, int *index, int dimensions);
    void sum(double val, int *index, int dimensions);
    // void adjust(double val);

    Elements &operator=(const Elements &start_vol);

    double &operator[](int index);

private:
    vector<double> _mtx;
    vector<int> _shape;
    int _dim = 0, _length = 0;
};

#endif
