#ifndef ECCSD_H
#define ECCSD_H

#include <cmath>
#include <unordered_map>
#include <string.h>
#include <stdio.h>
#include <vector>
// #include "./matrix/Matrix.hpp"

class CCSD {
private:
    // private members
    int num_electron;                           // number of electrons
    int dimension;                              // dimension of the matrix
    int nuclear_repulsion_energy;               // nuclear repulsion energy
    int scf_energy;                             // scf energy
    double *orbital_energy;                     // molecular orbital energy
    double *single_excitation;                  // T1   2d matrixes
    double *double_excitation;                  // T2   4d matrixes
    double *denominator_ai;                     // denominator of T1
    double *denominator_abij;                   // denominator of T2
    double *spin_ints;                          // spin basis double bar integral
    double *fs;                                 // fock matrix

    //intermediate 2d matrixes 
    double *single_intermediate;                                  // single intermediate matrix
    double *tsnew;

    //intermediate 4d matrixes
    double *double_intermediate;
    double *tdnew;

    void update_intermediate();
    void makeT1(const double *single_intermediate);
    void makeT2(const double *single_intermediate, const double *double_intermediate);
    double update_energy();
    
    // constructor helper methods
    void init_fs();
    void init_spin_ints(double *two_electron_integral, int two_electron_integral_size);
    void init_spin_ints(std::unordered_map<double, double> two_electron_integral);
    void init_denominators();
    
    double teimo(int a, int b, int c, int d, double* two_electron_integral, 
                 int two_electron_integral_size);
    double teimo(int a, int b, int c, int d, std::unordered_map<double, double> two_electron_integral);

    double taus(int a, int b, int c, int d);
    double tau(int a, int b, int i, int j);
    
    inline int index(int i, int j, int k, int l) {
        return i * dimension * dimension * dimension + \
               j * dimension * dimension + k * dimension + l;
    }

    inline int index(int i, int j) {
        return i * dimension + j;
    }

public:
    // public methods
    CCSD() = delete;

    CCSD( int num_electron, int dimension, int nuclear_repulsion_energy, 
          int scf_energy, double *orbital_energy, 
          double* two_electron_integral, int two_electron_integral_size);
    CCSD( int num_electron, int dimension, int nuclear_repulsion_energy, 
          int scf_energy, double *orbital_energy, 
          std::unordered_map<double, double> two_electron_integral);
    ~CCSD();

    double run();
};

#endif