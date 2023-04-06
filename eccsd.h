#ifndef ECCSD_H
#define ECCSD_H

#include <map>

class CCSD {
private:
    // private members
    int num_electron;                               // number of electrons
    int dimension;                                  // dimension of the matrix
    const int nuclear_repulsion_energy;             // nuclear repulsion energy
    int scf_energy;                                 // scf energy
    double *orbital_energy;                         // molecular orbital energy
    std::map<double, double> two_electron_integral; // two electron repulsion integral

    // private methods
    void update_intermediate();
    void makeT1();
    void makeT2();
    
public:
    // public methods
    CCSD();
    ~CCSD();
    void run();
};

#endif