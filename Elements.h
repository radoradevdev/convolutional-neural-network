#ifndef ELEMENTS_H
#define ELEMENTS_H

#include <iostream>
#include <vector>

using namespace std;

class Elements {
public:
    Elements() {}

    //! Constructor height and width
    /*!
      \param height height of every image
      \param width  width of every image
    */
    Elements(int height, int width);

    //! Constructor height, width and depth
    /*!
      \param height height of every image
      \param width  width of every image
      \param depth  TODO
    */
    Elements(int height, int width, int depth);

    //! Constructor layers, height, width and depth
    /*!
      \param layers TODO
      \param height height of every image
      \param width  width of every image
      \param depth  TODO
    */
    Elements(int layers, int height, int width, int depth);

    //! Constructor parameters and dimensions
    /*!
      \param params parameters of the Elements
      \param length length of the parameters
    */
    Elements(int *params, int length);

    //! Initialize after constructor
    /*!
      \param params parameters of the Elements
      \param length length of the parameters
    */
    void init(const int *params, int length);

    //! Reinitialize after constructor
    /*!
      \param params parameters of the Elements
      \param length length of the parameters
    */
    void reinit(int *params, int length);

    //! Get parameter by index
    /*!
      \param index index of the parameter(0=layers, 1=height, 2=width, 3=depth)
    */
    int getParam(int index);

    //! Get length, the product of all parameters
    int getLength();

    //! Get value
    /*!
      \param index          TODO
      \param params_length  length of the parameters
    */
    double getValue(int *index, int params_length);

    //! Get data, getter
    vector<double>& getData();

    //! Assign value to a specific position in the data vector
    /*!
      \param val    value to be inserted
      \param index  index on which to assign
      \param length length of the parameters
    */
    void assign(double val, int *index, int length);

    //! Add a value to a specific position in the data vector, Sum
    /*!
      \param val    value to be added
      \param index  index on which to assign
      \param length length of the parameters
    */
    void add(double val, int *index, int length);

    //! Overloaded assignment operator
    /*!
      \param elements   objedt to be copied from
    */
    Elements &operator=(const Elements &elements);

    //! Overloaded array index operator [], identical to getValue
    /*!
      \param index  index in the data
    */
    double &operator[](int index);

private:
    vector<double> _data;   /*!< One-dimensional data */
    vector<int> _params;    /*!< Parameters, (height, width) or
                                (height, width, depth) or
                                (layers, height, width, depth) */
    int _params_length = 0, /*!< Parameters length */
        _length = 0;        /*!< Length of the data */
};

#endif
