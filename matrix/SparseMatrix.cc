#include "Matrix.h"

template <typename T>
class SparseMatrix : public Matrix<T> {
private:
    int *dimensions;
    std::map<int, T> *data;

public:
    SparseMatrix() = delete;
    SparseMatrix(int *dimensions, T *data);
    SparseMatrix(int *dimensions);
    ~SparseMatrix();
    void at();
    void transpose();


    void mpi_transmit();
    void mpi_recv();
};
