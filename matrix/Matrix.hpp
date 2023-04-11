#include <tuple>

template <class T>
class Tensor{
protected:
    int dimensions;
    int col_max;
    int row_max;
    T *data;

public:
    T at(int *indices) {
        int index = 0;
        
        for (int i = 0; i < sizeof(dimensions); i++) {
            index += indices[i] * dimensions[i];
        }

        return data[index];
    };

    void transpose();
};