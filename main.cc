#include "eccsd.h"
#include <stdio.h>
#include <iostream>
#include <limits>
#include <iomanip>

int main(int argc, char *argv[]) {
    int nelec = 2;
    int dim = 2;
    double *orbital_energy = new double[2]{-1.52378656, -0.26763148};
    double e_nuc = 1.1386276671;
    double e_n = -3.99300007772;

    double ttmo[] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.9454269558303762, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17535895381500544, 
        0.0, 0.12682234020148653, 0.0, 0.0, 0.598553277016419, 
        0.0, -0.05682114362143326, 0.7471546478436311
    };
    // std::map<double, double> ttmo = {
    //     {5.0, 0.94542695583037617}, {12.0, 0.17535895381500544}, 
    //     {14.0, 0.12682234020148653}, {17.0, 0.59855327701641903}, 
    //     {19.0, -0.056821143621433257}, {20.0, 0.74715464784363106}
    // };
    
    CCSD ccsd = CCSD(nelec, dim, e_nuc, e_n, orbital_energy, ttmo, sizeof(ttmo) / sizeof(double));
    double result = ccsd.run();
    double result2 = result + e_n + e_nuc;

    printf(" E(corr, CCSD): %f \t E(CCSD): %f\n", result, result2);
    return 0;
}