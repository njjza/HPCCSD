#ifndef ECCSD_H
#define ECCSD_H

#include <cmath>
#include <cstdlib>
#include <map>
#include <string.h>
#include <stdio.h>

class CCSD {
private:
    // private members
    int num_electron;                               // number of electrons
    int dimension;                                  // dimension of the matrix
    int nuclear_repulsion_energy;                   // nuclear repulsion energy
    int scf_energy;                                 // scf energy
    double *orbital_energy;                         // molecular orbital energy
    std::map<double, double> two_electron_integral; // two electron repulsion integral

    double *single_excitation;                      // T1
    double *double_excitation;                      // T2

    double *denominator_ai;                         // denominator of T1
    double *denominator_abij;                       // denominator of T2

    double *spin_ints;                              // spin basis double bar integral
    double *fs;                                     // fock matrix

    // private methods
    void update_intermediate(
        double *fae, double *fmi, double *fme, double *wmnij, 
        double *wabef, double *wmbej
    );
    double update_energy();
    void makeT1(const double *fme, const double *fmi, const double *fae);
    void makeT2(const double *fae, const double *fmi, const double *fme, 
                const double *wabef, const double *wmnij, const double *wmbej);

    inline int index(int i, int j, int k, int l) {
        return i * dimension * dimension * dimension + \
               j * dimension * dimension + k * dimension + l;
    }

    inline int index(int i, int j) {
        return i * dimension + j;
    }

    inline void init_fs();
    inline void init_spin_ints();
    inline void init_denominators();
    
    double teimo(int a, int b, int c, int d) {
        // compound index given two indices
        auto eint = [](double x, double y) -> double {
            return x > y ? (x * (x + 1.0)) / 2.0 + y : (y * (y + 1.0)) / 2.0 + x; 
        };

        double result = eint(
            eint((double)a, (double)b), eint((double)c,(double)d)
        );

        if (two_electron_integral.find(result) != two_electron_integral.end()) {
            return two_electron_integral[result];
        } else {
            return 0.0;
        }
    }

    void test_arr_output(double *arr, int size) {
        printf("{");
        for (int i = 0; i < size; i++) {
            printf("%f, ", arr[i]);
        }
        printf("}\n");
    }

public:
    // public methods
    CCSD() = default;

    CCSD( int num_electron, int dimension, int nuclear_repulsion_energy, 
          int scf_energy, double *orbital_energy, 
          std::map<double, double> two_electron_integral);

    ~CCSD();
    double run();
};

#endif