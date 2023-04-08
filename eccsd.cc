#include "eccsd.h"
#include <stdio.h>

CCSD::CCSD( int num_electron, int dimension, 
            const int nuclear_repulsion_energy, int scf_energy, 
            double *orbital_energy, 
            const std::map<double, double> two_electron_integral) {
    
    this->num_electron = num_electron;
    this->dimension = dimension * 2;

    this->nuclear_repulsion_energy = nuclear_repulsion_energy;
    this->scf_energy = scf_energy;
    this->orbital_energy = orbital_energy;
    this->two_electron_integral = two_electron_integral;

    // printf("constructing CCSD object\n");


    // 2d array
    size_t arr_size = dimension * dimension * 4;
    this->fs = new double[arr_size];
    this->single_excitation = new double[arr_size];            // T1
    this->denominator_ai = new double[arr_size];               // Dai
    memset(fs, 0, arr_size * sizeof(double));
    memset(single_excitation, 0, arr_size * sizeof(double));
    memset(denominator_ai, 0, arr_size * sizeof(double));

    init_fs();

    // 4d array
    arr_size *= arr_size;
    this->double_excitation = new double[arr_size];            // T2
    this->denominator_abij = new double[arr_size];             // Dabij
    memset(this->double_excitation, 0, arr_size * sizeof(double));
    memset(this->single_excitation, 0, arr_size * sizeof(double));

    this->spin_ints = new double[arr_size]; //spin basis double bar integral
    init_spin_ints();

    init_denominators();
}

CCSD::~CCSD() {
    delete[] this->single_excitation;
    delete[] this->double_excitation;
    delete[] this->denominator_ai;
    delete[] this->denominator_abij;
    delete[] this->spin_ints;
    delete[] this->fs;
}

double CCSD::run() {
    // run the CCSD
    double ECCSD = 0.0;    // CCSD energy
    double DECC = 1.0;     // CCSD energy difference

    // int dimensions = this->dimension;
    
    int dimensions = dimension * dimension;
    double fae[dimensions] = {};         // by def this would init a zero array
    double fmi[dimensions] = {};
    double fme[dimensions] = {};
    
    dimensions *= dimensions;
    
    double wmnij[dimensions] = {};
    double wabef[dimensions] = {};
    double wmbej[dimensions] = {};

    while(DECC > 1.0e-6) {
        double OLDCC = ECCSD;   // CC energy of previous iteration

        // update the intermediate
        update_intermediate(fae, fmi, fme, wmnij, wabef, wmbej);

        // make T1
        makeT1(fme, fmi, fae);

        // make T2
        makeT2(fae, fmi, fme, wabef, wmnij, wmbej);

        // update the energy
        ECCSD = update_energy();

        // update the energy difference
        DECC = abs(ECCSD - OLDCC);
    }

    return ECCSD;
}

void CCSD::update_intermediate
(double *fae, double *fmi, double *fme, double *wmnij, double *wabef, double *wmbej) {
    // taus = lambda a, b, i, j: td[a,b,i,j] + 0.5*(ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j])
    auto taus = [this](int a, int b, int i, int j) {
        return double_excitation[index(a, b, i, j)] + \
        0.5*(
            single_excitation[index(a, i)] * single_excitation[index(b, j)] - \
            single_excitation[index(b, i)] * single_excitation[index(a, j)]
        );
    };

    // tau = lambda a, b, i, j: td[a,b,i,j] + ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j]
    auto tau = [this](int a, int b, int i, int j) {
        return double_excitation[index(a, b, i, j)] + \
        single_excitation[index(a, i)] * single_excitation[index(b, j)] - \
        single_excitation[index(b, i)] * single_excitation[index(a, j)];
    };


    // initialize the intermediate
    for (int a = num_electron; a < dimension; a++) {
        for (int e = num_electron; e < dimension; e++) {
            int m = a - num_electron; 
            int i = e - num_electron;
            fae[index(a, e)] = (1 - (a == e)) * fs[index(a, e)];
            fmi[index(m, i)] = (1 - (m == i)) * fs[index(m, i)];
            fme[index(m, e)] = fs[index(m, e)];
        }
    }

    // update the intermediate
    for (int a = num_electron; a < dimension; a++) {
        for (int b = num_electron; b < dimension; b++) {
            for (int e = num_electron; e < dimension; e++) {
                int tmp1 = a - num_electron;
                int tmp2 = b - num_electron;
                int tmp3 = e - num_electron;

                fae[index(a, e)] += -0.5 * fs[index(tmp2, e)] * single_excitation[index(a, tmp2)];
                fmi[index(tmp1,tmp2)] += 0.5 * fs[index(e, tmp2)] * single_excitation[index(tmp1, e)];

                for (int f = num_electron; f < dimension; f++) {
                    int tmp4 = f - num_electron;
                    
                    fae[index(a, e)] += single_excitation[index(f, tmp2)] * spin_ints[index(tmp2, a, f, e)];
                    fmi[index(tmp1, tmp2)] += single_excitation[index(e, tmp4)] * spin_ints[index(a, tmp4, tmp2, e)];
                    fme[index(tmp1, e)] += single_excitation[index(f, tmp2)] * spin_ints[index(tmp1, tmp2, e, f)];
                    wmnij[index(tmp1, tmp2, tmp3, tmp4)] = spin_ints[index(tmp1, tmp2, tmp3, tmp4)];
                    
                    for (int n = 0; n < num_electron; n++) {
                        int tmp5 = n + num_electron;
                        fae[index(a, e)] += -0.5 * taus(a, f, tmp2, n) * spin_ints[index(tmp2, n, e, f)];
                        fmi[index(tmp1, tmp2)] += 0.5 * taus(e, tmp5, tmp2, tmp4) * spin_ints[index(tmp1, tmp4, e, tmp5)];
                        wmnij[index(tmp1, tmp2, tmp3, tmp4)] += single_excitation[index(tmp5, tmp4)] * spin_ints[index(tmp1, tmp2, tmp3, tmp5)] - \
                        single_excitation[index(tmp5, tmp2)] * spin_ints[index(a, tmp4, tmp3, tmp5)];

                        for(int i = 0; i < num_electron; i++) {
                            int tmp6 = i + num_electron;
                            wmnij[index(tmp1, tmp2, tmp3, tmp4)] += 0.25 * tau(tmp5, tmp6, tmp3, tmp4) * spin_ints[index(tmp1, tmp2, tmp5, tmp6)];
                        }
                    }
                }
            }
        }
    }
 
    for(int a = num_electron; a < dimension; a++) {
        for(int b = num_electron; b < dimension; b++) {
            for(int e = num_electron; e < dimension; e++) {
                for(int f = num_electron; f < dimension; f++) {
                    int m = a - num_electron;
                    int j = f - num_electron;

                    wabef[index(a, b, e, f)] = spin_ints[index(a, b, e, f)];
                    wmbej[index(m, b, e, j)] = spin_ints[index(m, b, e, j)];

                    for(int i = 0; i < num_electron; i++) {
                        wabef[index(a, b, e, f)] += -single_excitation[index(b, i)] * spin_ints[index(a, i, e, f)] + \
                            single_excitation[index(a, i)] * spin_ints[index(b, i, e, f)];
                        wmbej[index(m, b, e, j)] += single_excitation[index(i + num_electron, j)] * spin_ints[index(m, b, e, i + num_electron)];
                        wmbej[index(m, b, e, j)] += -single_excitation[index(b, i)] * spin_ints[index(m, i, e, j)];

                        for(int n = 0; n < num_electron; n++) {
                            wabef[index(a, b, e, f)] += 0.25 * tau(a, b, i, n) * spin_ints[index(i, n, e, f)];
                            wmbej[index(m, b, e, j)] -= (0.5 * double_excitation[index(n + num_electron, b, j, i)] + \
                            single_excitation[index(n + num_electron, j)] * single_excitation[index(b, i)]) * spin_ints[index(m, i, e, n + num_electron)];
                        }
                    }
                }
            }
        }
    }
}

void CCSD::makeT1
(const double *fme, const double *fmi, const double *fae) {
    double tsnew[dimension][dimension] = {};
    
    for (int a = num_electron; a < dimension; a++) {
        for (int i=0; i<this->num_electron; i++) {
            tsnew[a][i] = fs[index(i, a)];
            
            for (int e = num_electron; e < dimension; e++) {
                tsnew[a][i] += single_excitation[index(e, i)] * fae[index(a, e)];
            }
            
            for (int m=0; m < num_electron; m++) {
                tsnew[a][i] -= single_excitation[index(a, m)] * fmi[index(m, i)];
                
                for (int e=num_electron; e < dimension; e++) {
                    tsnew[a][i] += double_excitation[index(a, e, i, m)] * fme[index(m, e)];
                    
                    for (int f=num_electron; f<dimension; f++) {
                        tsnew[a][i] -= 0.5 * double_excitation[index(e, f, i, m)] * \
                            this->spin_ints[index(m, a, e, f)];
                    }
                    
                    for (int n=0; n <  num_electron; n++) {
                        tsnew[a][i] -= 0.5 * double_excitation[index(a, e, m, n)] * \
                            this->spin_ints[index(n, m, e, i)];
                    }
                }
                
                for (int n=0; n<num_electron; n++) {
                    for (int f=num_electron; f<dimension; f++) {
                        tsnew[a][i] -= single_excitation[index(f, n)] * \
                            this->spin_ints[index(n, a, i, f)];
                    }
                }
            }

            tsnew[a][i] /= this->denominator_ai[index(a, i)];
        }
    }

    memcpy(single_excitation, tsnew, dimension * dimension * sizeof(double));
}

void CCSD::makeT2
(   const double *fae, const double *fmi, const double *fme, 
    const double *wabef, const double *wmnij, const double *wmbej
) {
    // make T2
    double tdnew[dimension][dimension][dimension][dimension];

    auto tau = [this](int a, int b, int i, int j) {
        return double_excitation[index(a, b, i, j)] + \
        single_excitation[index(a, i)] * single_excitation[index(b, j)] - \
        single_excitation[index(b, i)] * single_excitation[index(a, j)];
    };

    for (int a = num_electron; a < dimension; a++) {
        for (int b = num_electron; b < dimension; b++) {
            for (int i = 0; i < num_electron; i++) {
                for (int j = 0; j < num_electron; j++) {
                    tdnew[a][b][i][j] = spin_ints[index(i, j, a, b)];

                    for (int e = num_electron; e < dimension; e++) {
                        tdnew[a][b][i][j] += \
                            double_excitation[index(a, e, i, j)] * fae[index(b, e)] - \
                            double_excitation[index(b, e, i, j)] * fae[index(a, e)];

                        for (int m = 0; m < num_electron; m++) {
                            tdnew[a][b][i][j] += 0.5 * (
                                -double_excitation[index(a, e, i, j)] * single_excitation[index(b, m)] * fme[index(m, e)] + \
                                double_excitation[index(b, e, i, j)] * single_excitation[index(a, m)] * fme[index(m, e)]
                            );
                        }
                    }

                    for (int m = 0; m < num_electron; m++) {
                        tdnew[a][b][i][j] += \
                            -double_excitation[index(a, b, i, m)] * fmi[index(m, j)] + \
                            double_excitation[index(a, b, j, m)] * fmi[index(m, i)];

                        for (int e = num_electron; e < dimension; e++) {
                            tdnew[a][b][i][j] += 0.5 * (
                                -double_excitation[index(a, b, i, m)] * single_excitation[index(e, j)] * fme[index(m, e)] + \
                                double_excitation[index(a, b, j, m)] * single_excitation[index(e, i)] * fme[index(m, e)]
                            );
                        }
                    }
                    
                    for (int e = num_electron; e < dimension; e++) {
                        tdnew[a][b][i][j] += \
                            single_excitation[index(e, i)] * spin_ints[index(a, b, e, j)] - \
                            single_excitation[index(e, j)] * spin_ints[index(a, b, e, i)];
                        
                        for (int f = num_electron; f < dimension; f++) {
                            tdnew[a][b][i][j] += 0.5 * tau(e, f, i, j) *
                                wabef[index(a, b, e, f)];
                        }
                    }

                    for (int m = 0; m < num_electron; m++) {
                        tdnew[a][b][i][j] += \
                            -single_excitation[index(a, m)] * spin_ints[index(m, b, i, j)] + \
                            single_excitation[index(b, m)] * spin_ints[index(m, a, i, j)];

                        for (int e = num_electron; e < dimension; e++) {
                            tdnew[a][b][i][j] += (
                                double_excitation[index(a, e, i, m)] * wmbej[index(m, b, e, j)] - single_excitation[index(e, i)] * single_excitation[index(a, m)] * spin_ints[index(m, b, e, j)] + \
                                -double_excitation[index(a, e, j, m)] * wmbej[index(m, b, e, i)] + single_excitation[index(e, j)] * single_excitation[index(a, m)] * spin_ints[index(m, b, e, i)] + \
                                -double_excitation[index(b, e, i, m)] * wmbej[index(m, a, e, j)] + single_excitation[index(e, i)] * single_excitation[index(b, m)] * spin_ints[index(m, a, e, j)] + \
                                double_excitation[index(b, e, j, m)] * wmbej[index(m, a, e, i)] - single_excitation[index(e, j)] * single_excitation[index(b, m)] * spin_ints[index(m, a, e, i)]
                            );
                        }

                        for (int n = 0; n < num_electron; n++) {
                            tdnew[a][b][i][j] += 0.5 * tau(a, b, m, n) *
                                wmnij[index(a, b, m, n)];
                        }
                    }

                    tdnew[a][b][i][j] /= denominator_abij[index(a, b, i, j)];
                }
            }
        }
    }

    memcpy(double_excitation, tdnew, dimension * dimension * dimension * dimension * sizeof(double));

}

double CCSD::update_energy() {
    double energy = 0.0;

    for (int i = 0; i < num_electron; i++) {
        for (int a = num_electron; a < dimension; a++) {
            for (int j = 0; j < num_electron; j++) {
                for (int b = num_electron; b < dimension; b++) {
                    energy += 0.25 * spin_ints[index(i, j, a, b)] * double_excitation[index(a, b, i, j)];
                    energy += 0.5 * spin_ints[index(i, j, a, b)] * single_excitation[index(a, i)] * single_excitation[index(b, j)];
                }
            }
        }
    }

    return energy;
}

inline void CCSD::init_fs() {
    int dimensions = dimension * 2; 
    double tmp[dimensions];

    for (int i = 0; i < dimensions; i++) {
        tmp[i] = orbital_energy[i >> 1];
    }

    // diagonalizing the orbital tmp array
    for (int i = 0; i < dimensions; i++) {
        fs[index(i, i)] = tmp[i];
    }
}

inline void CCSD::init_spin_ints() {
    for (int i = 2; i < dimension + 2; i++) {
        for (int j = 2; j < dimension + 2; j++) {
            for (int k = 2; k < dimension + 2; k++) {
                for (int l = 2; l < dimension + 2; l++) {
                    int p = i >> 1;
                    int q = j >> 1;
                    int r = k >> 1;
                    int s = l >> 1;
                    
                    double value1 = teimo(p, r, q, s);
                    value1 *= (i % 2 == k % 2) * (j % 2 == l % 2);
                    
                    double value2 = teimo(p, s, q, r);
                    value2 *= (i % 2 == l % 2) * (j % 2 == k % 2);
                    
                    spin_ints[index(i - 2, j - 2, k - 2, l - 2)] = value1 - value2;
                }
            }
        }
    }
}

inline void CCSD::init_denominators() {
    for(int a = num_electron; a < dimension; a++) {
        for(int i = 0; i < num_electron; i++) {
            denominator_ai[index(a, i)] = fs[index(i, i)] - fs[index(a, a)];

            for (int b = num_electron; b < dimension; b++) {
                for (int j = 0; j < num_electron; j++) {
                    double tmp = fs[index(i, i)] + fs[index(j, j)] - \
                              fs[index(a, a)] - fs[index(b, b)];

                    double_excitation[index(a, b, i, j)] += \
                        spin_ints[index(i, j, a, b)] / tmp;

                    denominator_abij[index(a, b, i, j)] = tmp;
                }
            }
        }
    }
}