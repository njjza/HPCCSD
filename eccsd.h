#ifndef ECCSD_H
#define ECCSD_H

#include <cmath>
#include <cstdlib>
#include <map>
#include <string.h>

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

    // def teimo(a,b,c,d):
    // eint = lambda x, y: x*(x+1)/2 + y if x>y else y*(y+1)/2 + x # compound index given two indices
    // return ttmo.get(eint(eint(a,b),eint(c,d)),0.0e0)

    double teimo(int a, int b, int c, int d) {
        auto eint = [](int x, int y) -> int {
            return x > y ? x*(x+1)/2 + y : y*(y+1)/2 + x; // compound index given two indices
        };

        return (
                two_electron_integral.find(eint(eint(a,b),eint(c,d))) != \
                two_electron_integral.end()
            ) ? two_electron_integral[eint(eint(a,b),eint(c,d))] : 0.0;
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