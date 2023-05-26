#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include "Elements.h"

enum PoolingOperation {
    MAX,    /*!< Max-pooling */
    AVG     /*!< Avg-pooling */
};

//! PoolingLayer class
/*!
  The class describes the operations of the pooling layers.
*/
class PoolingLayer {
public:
    //! Constructor for initializing the Pooling Layer
    /*!
     * \param image_dim     image dimensions
     * \param mode          method of pooling
     * \param size          size of pooling
     * \param stride        stride of pooling
     */
    PoolingLayer(
            int image_dim[3],
            PoolingOperation mode = PoolingOperation::AVG,
            int size = 2,
            int stride = 2
            );

    //! Forward propagation in the pooling layer
    /*!
      \param image  original image
      \param out    result
    */
    void fwd(const Elements &image, Elements &out);

    //! Backwards propagation in the pooling layer
    /*!
      \param out    result
      \param image  original image
    */
    void bp(Elements out, Elements &image);
private:
    int _image_dim[3] = {1, 16, 16};    /*!< Default image specification */

    int _size = 2;      /*!< width and height of the pooling window */
    int _stride = 2;    /*!< How many pixels every convolution to shift */
    PoolingOperation _mode = PoolingOperation::AVG;   /*!< Mode of pooling */

    Elements _cache;    /*!< the cached forward elements */

    int _out_dim[3] = {2, 13, 13};      /*!< depth, height, width */
};

#endif // POOLINGLAYER_H
