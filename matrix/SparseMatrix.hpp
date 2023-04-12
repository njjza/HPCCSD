#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H
#include <map>
class SparseMatrix {
public:
    SparseMatrix(){
        this->m_map = std::map<int, double>();
    };
    class Proxy {
        public:
            Proxy(SparseMatrix& parent, int key) : m_parent(parent), m_key(key) {}

            operator double() const {
                return m_parent.get_value(m_key);
            }

            Proxy& operator=(double value) {
                m_parent.set_value(m_key, value);
                return *this;
            }

        private:
            SparseMatrix& m_parent;
            int m_key;
    };

    Proxy operator[](int key) {
        return Proxy(*this, key);
    }

    double operator[](int key) const {
        return get_value(key);
    }

    bool has_key(int key) {
        return m_map.find(key) != m_map.end();
    }

    void swap(SparseMatrix& other) {
        std::swap(m_map, other.m_map);
    }

private:
    double get_value(int key) const {
        auto it = m_map.find(key);
        if (it != m_map.end()) {
            return it->second;
        }
        return 0.0;
    }

    void set_value(int key, double value) {
        if (value == 0.0) {
            m_map.erase(key);
        } else {
            m_map[key] = value;
        }
    }

    std::map<int, double> m_map;
};
#endif