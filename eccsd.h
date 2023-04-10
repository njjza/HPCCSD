#ifndef ECCSD_H
#define ECCSD_H

#include <cmath>
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
    void update_intermediate(
        double *fae, double *fmi, double *fme, double *wmnij, 
        double *wabef, double *wmbej, int x
    );
    void makeT1(double *tsnew, const double *fme, const double *fmi, const double *fae);
    void makeT2(double *tdnew, const double *fae, const double *fmi, const double *fme, 
                const double *wabef, const double *wmnij, const double *wmbej);
    double update_energy();
    void update_fae(double *fae);
    void update_fmi(double *fmi);
    void update_fme(double *fme);
    
    // constructor helper methods
    void init_fs();
    double teimo(int a, int b, int c, int d, std::map<double, double> two_electron_integral);
    double taus(int a, int b, int c, int d);
    double tau(int a, int b, int i, int j);
    // initialize spin basis double bar integral and denominators
    void init_spin_ints(std::map<double, double> two_electron_integral);
    void init_denominators();
    
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