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
    void makeT1(double *fme, double *fmi, double *fae);
    void makeT2(double *fae, double *fmi, double *fme, 
                double *wabef, double *wmnij, double *wmbej);

    inline int index(int i, int j, int k, int l) {
        return i * dimension * dimension * dimension + \
               j * dimension * dimension + k * dimension + l;
    }

    inline int index(int i, int j) {
        return i * dimension + j;
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