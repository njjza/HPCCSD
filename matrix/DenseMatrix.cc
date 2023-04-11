#include "Matrix.h"
#include <map>
#include <tuple>

template <typename T>
class DenseMatrix : public Matrix<T> {
private:
    T* data;
public:
    DenseMatrix(int *dimensions, T *data);
    DenseMatrix(int *dimensions, T *data);
    ~DenseMatrix();
    
    template <... int>
    double at(&) {

        if(data.find(std::make_tuple(x, y, z, w)) != data.end()) {
            return data[std::make_tuple(x, y, z, w)];
        }

        return 0.0
    };
};