#include <iostream>
#include <map>

class SparseMatrix {
public:
    SparseMatrix() = default;

    double& operator[](int index) {
        if (data.find(index) == data.end()) {
            data[index] = 0;
        }
        return data[index];
    }

    const double& operator[](int index) const {
        auto it = data.find(index);
        if (it != data.end()) {
            return it->second;
        }
        static const double defaultValue = 0;
        return defaultValue;
    }

    SparseMatrix(const SparseMatrix& other) {
        data = other.data;
    }

    SparseMatrix& operator=(SparseMatrix other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(SparseMatrix& first, SparseMatrix& second) {
        using std::swap;
        swap(first.data, second.data);
    }

private:
    std::map<int, double> data;
};