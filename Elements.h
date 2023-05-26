#ifndef ELEMENTS_H
#define ELEMENTS_H

#include <iostream>
#include <vector>

using namespace std;

//! Elements class
/*!
  The class describes the data as a tensor after it has been read from the dataset.
  Like Numpy
*/
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
    int getParam(int index) const;

    //! Get length, the product of all parameters
    int getLength();

    //! Get value
    /*!
      \param indices          TODO
      \param params_length  length of the parameters
    */
    double getValue(int *indices, int params_length) const;

    //! Get data, getter
    vector<double>& getData();

    //! Assign value to a specific position in the data vector
    /*!
      \param val        value to be inserted
      \param indices    index on which to assign
      \param length     length of the parameters
    */
    void allocate(double val, int *indices, int length);

    //! Add a value to a specific position in the data vector, Sum
    /*!
      \param val        value to be added
      \param indices    index on which to assign
      \param length     length of the parameters
    */
    void aggregate(double val, int *indices, int length);

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

    //! (Helper function) Initialize after constructor
    /*!
      \param params parameters of the Elements
      \param length length of the parameters
    */
    void _init(const int *params, int length);

    //! (Helper function) Find element to be assigned or get
    /*!
      \param index  index on which to assign
      \param length length of the parameters
    */
    int _find(int *indices, int length) const;
};

#endif
