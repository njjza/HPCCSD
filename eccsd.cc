#include "eccsd.hpp"
#include "omp.h"
// #include "mpi.h"
#include <immintrin.h> // Header for AVX2

CCSD::CCSD( int num_electron, int dimension, 
            const int nuclear_repulsion_energy, int scf_energy, 
            double *orbital_energy, 
            double* two_electron_integral, int two_electron_integral_size) {
    
    this->num_electron = num_electron;
    this->dimension = dimension * 2;

    this->nuclear_repulsion_energy = nuclear_repulsion_energy;
    this->scf_energy = scf_energy;
    this->orbital_energy = orbital_energy;

    // 2d array
    size_t arr_size = dimension * dimension * 4;
    this->fs = new double[arr_size];
    this->single_excitation = new double[arr_size];            // T1
    this->denominator_ai = new double[arr_size];               // Dai
    
    // intermediate matrixes
    // this->f = new double[arr_size];
    this->fae = new double[arr_size];
    this->fmi = new double[arr_size];
    this->fme = new double[arr_size];
    this->tsnew = new double[arr_size];

    memset(fs, 0, arr_size * sizeof(double));
    memset(single_excitation, 0, arr_size * sizeof(double));
    memset(denominator_ai, 0, arr_size * sizeof(double));

    init_fs();

    // 4d array
    arr_size *= arr_size;
    this->double_excitation = new double[arr_size];            // T2
    this->denominator_abij = new double[arr_size];             // Dabij
    this->spin_ints = new double[arr_size]; //spin basis double bar integral
    memset(spin_ints, 0, arr_size * sizeof(double));
    memset(double_excitation, 0, arr_size * sizeof(double));
    memset(denominator_abij, 0, arr_size * sizeof(double));

    // intermediate matrixes
    this->wmnij = new double[arr_size];
    this->wabef = new double[arr_size];
    this->wmbej = new double[arr_size];
    this->tdnew = new double[arr_size];
    
    init_spin_ints(two_electron_integral, two_electron_integral_size);
    init_denominators();

}

CCSD::~CCSD() {
    delete[] this->single_excitation;
    delete[] this->double_excitation;
    delete[] this->denominator_ai;
    delete[] this->denominator_abij;
    delete[] this->spin_ints;
    delete[] this->fs;
    delete[] this->fae;
    delete[] this->fmi;
    delete[] this->fme;
    delete[] this->wmnij;
    delete[] this->wabef;
    delete[] this->wmbej;
    delete[] this->tsnew;
    delete[] this->tdnew;
}

double CCSD::run() {
    // allocate all the intermediate matrixes on the heap
    // it is to prevent segmentation fault once the dimension is too large

    int dimensions = dimension * dimension;

    dimensions *= dimensions;
    
    
    double ECCSD = 0.0;    // CCSD energy
    double OLDCC = 0.0;    // CCSD energy of previous iteration
    double DECC = 1.0;     // CCSD energy difference
    
    // through testing, the system has the better performance when the number
    // of the cores matches the number of dimensions.
    #ifdef DEBUG_OMP
    omp_set_num_threads(dimension);
    #pragma omp parallel shared(DECC, ECCSD, OLDCC)
    #endif
    {
        do {
            // update the intermediate
            update_intermediate(fae, fmi, fme, wmnij, wabef, wmbej);

            #pragma omp barrier

            makeT1(tsnew, fme, fmi, fae);
            makeT2(tdnew, fae, fmi, fme, wabef, wmnij, wmbej);

            #pragma omp barrier

            // won't effect the single thread scenario, not encapsulated
            // in the DEBUG_OMP section

            // it is better to have single thread perform each memcpy
            // produces huge i/o delay once the dimension get large.
            // might be better to have opnempi in this case?
            #pragma omp sections nowait
            {
                #pragma omp section
                {
                    // update the single excitation
                    memcpy(single_excitation, tsnew, dimension * dimension * 
                           sizeof(double));
                }
                #pragma omp section
                {
                    // update the double excitation
                    memcpy(double_excitation, tdnew, dimensions * 
                           sizeof(double));
                }
            }

            #pragma omp single nowait
            {
                OLDCC = ECCSD;
                #pragma omp flush(OLDCC)
            }
            
            // update the energy
            ECCSD = update_energy();

            // update the energy difference
            #pragma omp single 
            {
                DECC = std::abs(ECCSD - OLDCC);
                #pragma omp flush(ECCSD)
                #pragma omp flush(DECC)
            }

            #pragma omp barrier
        }
        while(DECC > 1.0e-9);
    }

    return ECCSD;
}

inline double CCSD::taus(int a, int b, int i, int j) {
    return double_excitation[index(a, b, i, j)] + \
        0.5*(
            single_excitation[index(a, i)] * single_excitation[index(b, j)] - \
            single_excitation[index(b, i)] * single_excitation[index(a, j)]
        );
}

inline double CCSD::tau(int a, int b, int i, int j) {
    return double_excitation[index(a, b, i, j)] + \
        single_excitation[index(a, i)] * single_excitation[index(b, j)] - \
        single_excitation[index(b, i)] * single_excitation[index(a, j)];
}

#ifndef OMP
void CCSD::update_intermediate(double *fae, double *fmi, double *fme, 
                               double *wmnij, double *wabef, double *wmbej) {
    // taus = lambda a, b, i, j: td[a,b,i,j] + 0.5*(ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j])

    // initialize the intermediate
    for (int a = num_electron; a < dimension; a++) {
        for (int b = num_electron; b < dimension; b++) {
            int m = a - num_electron; 
            int i = b - num_electron;
            fae[index(a, b)] = (1 - (a == b)) * fs[index(a, b)];
            fmi[index(m, i)] = (1 - (m == i)) * fs[index(m, i)];
            fme[index(m, b)] = fs[index(m, b)];
        }
    }

    // update the intermediate
    for (int a = num_electron; a < dimension; a++) {
        for (int b = num_electron; b < dimension; b++) {
            for (int e = num_electron; e < dimension; e++) {
                int m = a - num_electron;
                int i = b - num_electron;
                int tmp3 = e - num_electron;

                fae[index(a, e)] += -0.5 * fs[index(i, e)] * single_excitation[index(a, i)];
                fmi[index(m,i)] += 0.5 * single_excitation[index(e, i)] * fs[index(m, e)];

                for (int f = num_electron; f < dimension; f++) {
                    int tmp4 = f - num_electron;
                    
                    fae[index(a, e)] += single_excitation[index(f, i)] * spin_ints[index(i, a, f, e)];
                    fmi[index(m, i)] += single_excitation[index(e, tmp4)] * spin_ints[index(m, tmp4, i, e)];
                    fme[index(m, e)] += single_excitation[index(f, i)] * spin_ints[index(m, i, e, f)];
                    wmnij[index(m, i, tmp3, tmp4)] = spin_ints[index(m, i, tmp3, tmp4)];
                    
                    for (int n = 0; n < num_electron; n++) {
                        int tmp5 = n + num_electron;
                        fae[index(a, e)] += -0.5 * taus(a, f, i, n) * spin_ints[index(i, n, e, f)];
                        fmi[index(m, i)] += 0.5 * taus(e, tmp5, i, tmp4) * spin_ints[index(m, tmp4, e, tmp5)];
                        wmnij[index(m, i, tmp3, tmp4)] += single_excitation[index(tmp5, tmp4)] * spin_ints[index(m, i, tmp3, tmp5)] - \
                        single_excitation[index(tmp5, tmp3)] * spin_ints[index(m, i, tmp4, tmp5)];
                        
                        for(int i = 0; i < num_electron; i++) {
                            int tmp6 = i + num_electron;
                            wmnij[index(m, i, tmp3, tmp4)] += 0.25 * tau(tmp5, tmp6, tmp3, tmp4) * spin_ints[index(m, i, tmp5, tmp6)];
                        }
                    }
                }
            }
        }
    }
 
    // #pragma omp parallel for collapse(4)
    for(int a = num_electron; a < dimension; a++) {
        for(int b = num_electron; b < dimension; b++) {
            for(int e = num_electron; e < dimension; e++) {
                for(int f = num_electron; f < dimension; f++) {
                    int m = a - num_electron;
                    int j = f - num_electron;

                    wabef[index(a, b, e, f)] = spin_ints[index(a, b, e, f)];
                    wmbej[index(m, b, e, j)] = spin_ints[index(m, b, e, j)];

                    // #pragma omp simd
                    for(int i = 0; i < num_electron; i++) {
                        wabef[index(a, b, e, f)] += -single_excitation[index(b, i)] * spin_ints[index(a, i, e, f)] + \
                            single_excitation[index(a, i)] * spin_ints[index(b, i, e, f)];
                        wmbej[index(m, b, e, j)] += single_excitation[index(i + num_electron, j)] * spin_ints[index(m, b, e, i + num_electron)];
                        wmbej[index(m, b, e, j)] += -single_excitation[index(b, i)] * spin_ints[index(m, i, e, j)];
                        
                        // #pragma omp simd
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
#endif

#ifdef OMP
void CCSD::update_intermediate(double *fae, double *fmi, double *fme, 
                               double *wmnij, double *wabef, double *wmbej) {
    
    // update fae
    #pragma omp for nowait
    for(int a = num_electron; a < dimension; a++) {
        for (int e = num_electron; e < dimension; e++) {
            double fae_result = (1 - a == e) * fs[index(a, e)];

            for(int m = 0; m < num_electron; m++) {
                fae_result -= 0.5 * fs[index(m, e)] * \
                              single_excitation[index(a, m)];

                for (int f = num_electron; f < dimension; f++) {
                    fae_result += single_excitation[index(f, m)] * \
                                  spin_ints[index(m, a, f, e)];

                    for (int n = 0; n < num_electron; n++) {
                        fae_result -= 0.5 * taus(a, f, m, n) * \
                                      spin_ints[index(m, n, e, f)];
                    }
                }
            }

            #pragma omp atomic write
            fae[index(a, e)] = fae_result;
        }
    }

    // update fmi
    #pragma omp for nowait
    for (int m = 0; m < num_electron; m++) {
        for (int i = 0; i < num_electron; i++) {
            double fmi_result = (1 - m == i) * fs[index(m, i)];

            for (int e = num_electron; e < dimension; e++) {
                fmi_result += 0.5 * fs[index(e, i)] * \
                              single_excitation[index(m, e)];

                for (int n = 0; n < num_electron; n++) {
                    fmi_result += single_excitation[index(e, n)] * \
                                  spin_ints[index(m, n, i, e)];

                    for (int f = num_electron; f < dimension; f++) {
                        fmi_result += 0.5 * taus(e, f, i, n) * \
                                      spin_ints[index(m, n, e, f)];
                    }
                }
            }

            #pragma omp atomic write
            fmi[index(m, i)] = fmi_result;
        }
    }

    // update fme
    #pragma omp for nowait
    for(int m = 0; m < num_electron; m++) {
        for(int e = num_electron; e < dimension; e++) {
            double fme_result = fs[index(m, e)];

            #pragma omp simd collapse(2)
            for(int n = 0; n < num_electron; n++) {                        
                for(int f = num_electron; f < dimension; f++) {
                    fme_result += single_excitation[index(f, n)] * \
                                  spin_ints[index(m, n, e, f)];   
                }
            }

            #pragma omp atomic write
            fme[index(m, e)] = fme_result;
        }
    }

    // update wmnij 
    #pragma omp for nowait
    for (int m = 0; m < num_electron; m++) {
        for (int n = 0; n < num_electron; n++) {
            for(int i = 0; i < num_electron; i++) {
                for(int j = 0; j < num_electron; j++) {
                    double wmnij_result = spin_ints[index(m, n, i, j)];

                    for (int e = num_electron; e < dimension; e++) {
                        wmnij_result += single_excitation[index(e, j)] * \
                                        spin_ints[index(m, n, i, e)];
                        wmnij_result -= single_excitation[index(e, i)] * \
                                        spin_ints[index(m, n, i, e)];

                        for (int f = num_electron; f < dimension; f++) {
                            wmnij_result += 0.25 * tau(e, f, i, j) * \
                                            spin_ints[index(m, n, f, e)];
                        }
                    }

                    #pragma omp atomic write
                    wmnij[index(m, n, i, j)] = wmnij_result;
                }
            }
        }
    }

    // update wabef
    #pragma omp for nowait
    for (int a = num_electron; a < dimension; a++) {
        for (int b = num_electron; b < dimension; b++) {
            for (int e = num_electron; e < dimension; e++) {
                for (int f = num_electron; f < dimension; f++) {
                    double wabef_result = spin_ints[index(a, b, e, f)];

                    for (int m = 0; m < num_electron; m++) {
                        wabef_result -= single_excitation[index(b, m)] * spin_ints[index(a, m, e, f)];
                        wabef_result += single_excitation[index(a, m)] * spin_ints[index(b, m, e, f)];

                        for (int n = 0; n < num_electron; n++) {
                            wabef_result -= 0.25 * tau(a, b, m, n) * spin_ints[index(m, n, e, f)];
                        }
                    }

                    #pragma omp atomic write
                    wabef[index(a, b, e, f)] = wabef_result;
                }
            }
        }
    }

    // update wmbej
    #pragma omp for    //remove wait so the for loop can act as a barrier
    for (int m = 0; m < num_electron; m++) {
        for (int b = num_electron; b < dimension; b++) {
            for (int e = num_electron; e < dimension; e++) {
                for (int j = 0; j < num_electron; j++) {
                    double wmbej_result = spin_ints[index(m, b, e, j)];

                    for (int f = num_electron; f < dimension; f++) {
                        wmbej_result += single_excitation[index(f, j)] * spin_ints[index(m, b, e, f)];

                        for (int n = 0; n < num_electron; n++) {
                            wmbej_result -= (
                                0.5 * double_excitation[index(f, b, j, n)] + \
                                single_excitation[index(f, j)] * single_excitation[index(b, n)]
                            ) * spin_ints[index(m, n, e, f)];
                        }
                    }

                    for (int n = 0; n < num_electron; n++) {
                        wmbej_result -= single_excitation[index(b, n)] * spin_ints[index(m, n, e, j)];
                    }

                    wmbej[index(m, b, e, j)] = wmbej_result;
                }
            }
        }
    }
}
#endif

void CCSD::makeT1
(   double *tsnew, const double *fme, const double *fmi, 
    const double *fae) {
    
    #pragma omp for nowait
    for (int a = num_electron; a < dimension; a++) {
        for (int i = 0; i< num_electron; i++) {
            double result = fs[index(i, a)];

            for (int e = num_electron; e < dimension; e++) {
                result += single_excitation[index(e, i)] * fae[index(a, e)];
            }

            #pragma omp simd collapse(2) reduction(-:result)
            for (int n = 0; n < num_electron; n++) {
                for (int f = num_electron; f < dimension; f++) {
                    result -= single_excitation[index(f, n)] * \
                              spin_ints[index(n, a, i, f)];
                }
            }
            
            for (int m = 0; m < num_electron; m++) {
                result -= single_excitation[index(a, m)] * fmi[index(m, i)];
                
                for (int e = num_electron; e < dimension; e++) {
                    result += double_excitation[index(a, e, i, m)] * \
                              fme[index(m, e)];
                    
                    #pragma omp simd reduction(-:result)
                    for (int f=num_electron; f<dimension; f++) {
                        // fusing loop by introducing another variable n
                        // n iterate over the range(0, num_electron)
                        int n = f - num_electron;

                        result -= 0.5 * double_excitation[index(e, f, i, m)] * \
                            spin_ints[index(m, a, e, f)];
                        result -= 0.5 * double_excitation[index(a, e, m, n)] * \
                            spin_ints[index(n, m, e, i)];
                    }
                }
            }

            result /= denominator_ai[index(a, i)];
            tsnew[index(a, i)] = result;
        }
    }
}

void CCSD::makeT2
(   double *tdnew, const double *fae, const double *fmi, const double *fme, 
    const double *wabef, const double *wmnij, const double *wmbej
) {
    
    // make T2 O(n^6)
    #pragma omp for collapse(4)
    for (int a = num_electron; a < dimension; a++) {
        for (int b = num_electron; b < dimension; b++) {
            for (int i = 0; i < num_electron; i++) {
                for (int j = 0; j < num_electron; j++) {
                    
                    double result = spin_ints[index(i, j, a, b)];

                    for (int e = num_electron; e < dimension; e++) {
                        double td_aeij = double_excitation[index(a, e, i, j)];
                        double td_beij = double_excitation[index(b, e, i, j)];

                        result += td_aeij * fae[index(b, e)] - \
                                  td_beij * fae[index(a, e)];

                        #pragma omp simd reduction(-:result)
                        for (int m = 0; m < num_electron; m++) {
                            double val1 = td_aeij * fme[index(m, e)] * \
                                            single_excitation[index(b, m)];
                            double val2 = td_beij * fme[index(m, e)] * \
                                            single_excitation[index(a, m)];
                            result += 0.5 * (val2 - val1);
                        }
                    }

                    for (int m = 0; m < num_electron; m++) {
                        double n_td_abim = -double_excitation[index(a, b, i, m)];
                        double td_abjm = double_excitation[index(a, b, j, m)];

                        result += n_td_abim * fmi[index(m, j)] + td_abjm * fmi[index(m, i)];

                        #pragma omp simd reduction(-:result)
                        for (int e = num_electron; e < dimension; e++) {
                            double val1 = n_td_abim * single_excitation[index(e, j)] * fme[index(m, e)];
                            double val2 = td_abjm * single_excitation[index(e, i)] * fme[index(m, e)];
                            result += 0.5 * (val1 - val2);
                        }
                    }
                    
                    for (int e = num_electron; e < dimension; e++) {
                        result += \
                            single_excitation[index(e, i)] * spin_ints[index(a, b, e, j)] - \
                            single_excitation[index(e, j)] * spin_ints[index(a, b, e, i)];
                        
                        #pragma omp simd reduction(-:result)
                        for (int f = num_electron; f < dimension; f++) {
                            result += 0.5 * tau(e, f, i, j) *
                                wabef[index(a, b, e, f)];
                        }
                    }

                    for (int m = 0; m < num_electron; m++) {
                        result += \
                            -single_excitation[index(a, m)] * spin_ints[index(m, b, i, j)] + \
                            single_excitation[index(b, m)] * spin_ints[index(m, a, i, j)];
                        
                        #pragma omp simd reduction(-:result)
                        for (int e = num_electron; e < dimension; e++) {
                            result += double_excitation[index(a, e, i, m)] * wmbej[index(m, b, e, j)] \
                                    - single_excitation[index(e, i)] * single_excitation[index(a, m)] \
                                            * spin_ints[index(m, b, e, j)];
                            result += -double_excitation[index(a, e, j, m)] * wmbej[index(m, b, e, i)] \
                                    + single_excitation[index(e, j)] * single_excitation[index(a, m)] \
                                        * spin_ints[index(m, b, e, i)];
                            result += -double_excitation[index(b, e, i, m)] * wmbej[index(m, a, e, j)] \
                                    + single_excitation[index(e, i)] * single_excitation[index(b, m)] \
                                        * spin_ints[index(m, a, e, j)];
                            result += double_excitation[index(b, e, j, m)] * wmbej[index(m, a, e, i)] \
                                    - single_excitation[index(e, j)] * single_excitation[index(b, m)] \
                                        * spin_ints[index(m, a, e, i)];
                        }

                        #pragma omp simd reduction(-:result)
                        for (int n = 0; n < num_electron; n++) {
                            result += 0.5 * tau(a, b, m, n) * \
                                      wmnij[index(m, n, i, j)];
                        }
                    }

                    int abij_index = index(a, b, i, j);
                    result /= denominator_abij[abij_index];
                    #pragma omp atomic write
                    tdnew[abij_index] = result;
                }
            }
        }
    }
}

inline double CCSD::update_energy() {
    double energy = 0.0;

#ifdef OMP_DEBUG
    #pragma omp for reduction(+:energy)
#endif
    for (int i = 0; i < num_electron; i++) {
        for (int a = num_electron; a < dimension; a++) {
            for (int j = 0; j < num_electron; j++) {
                for (int b = num_electron; b < dimension; b++) {
                    double spin_ints_ijab = spin_ints[index(i, j, a, b)];

                    energy += 0.25 * spin_ints_ijab * double_excitation[index(a, b, i, j)];
                    energy += 0.5 * spin_ints_ijab * single_excitation[index(a, i)] * single_excitation[index(b, j)];
                }
            }
        }
    }

    return energy;
}

inline void CCSD::init_fs() {
    int dimensions = dimension; 
    double tmp[dimensions];

    for (int i = 0; i < dimensions; i++) {
        tmp[i] = orbital_energy[i >> 1];
    }

    // diagonalizing the orbital tmp array
    for (int i = 0; i < dimensions; i++) {
        fs[index(i, i)] = tmp[i];
    }
}

void CCSD::init_spin_ints(double* two_electron_integral, 
                          int two_electron_integral_size) {
    #pragma omp simd collapse(4)
    for (int i = 2; i < dimension + 2; i++) {
        for (int j = 2; j < dimension + 2; j++) {
            for (int k = 2; k < dimension + 2; k++) {
                for (int l = 2; l < dimension + 2; l++) {
                    int p = i >> 1;
                    int q = j >> 1;
                    int r = k >> 1;
                    int s = l >> 1;
                    
                    double value1 = teimo(p, r, q, s, 
                                          two_electron_integral, 
                                          two_electron_integral_size);
                    double value2 = teimo(p, s, q, r, two_electron_integral, two_electron_integral_size);
                    value1 *= (i % 2 == k % 2) * (j % 2 == l % 2);
                    value2 *= (i % 2 == l % 2) * (j % 2 == k % 2);
                    
                    spin_ints[index(i - 2, j - 2, k - 2, l - 2)] = value1 - value2;
                }
            }
        }
    }
}

void CCSD::init_spin_ints(
    std::unordered_map<double, double> two_electron_integral) {
    #pragma omp simd collapse(4)
    for (int i = 2; i < dimension + 2; i++) {
        for (int j = 2; j < dimension + 2; j++) {
            for (int k = 2; k < dimension + 2; k++) {
                for (int l = 2; l < dimension + 2; l++) {
                    int p = i >> 1;
                    int q = j >> 1;
                    int r = k >> 1;
                    int s = l >> 1;
                    
                    double value1 = teimo(p, r, q, s, two_electron_integral);
                    double value2 = teimo(p, s, q, r, two_electron_integral);
                    value1 *= (i % 2 == k % 2) * (j % 2 == l % 2);
                    value2 *= (i % 2 == l % 2) * (j % 2 == k % 2);
                    
                    spin_ints[index(i - 2, j - 2, k - 2, l - 2)] = value1 - value2;
                }
            }
        }
    }
}

void CCSD::init_denominators() {
    for(int a = num_electron; a < dimension; a++) {
        for(int i = 0; i < num_electron; i++) {
            double fs_ii = fs[index(i, i)];
            double fs_aa = fs[index(a, a)];

            denominator_ai[index(a, i)] = fs_ii - fs_aa;

            #pragma omp simd
            for (int b = num_electron; b < dimension; b++) {
                for (int j = 0; j < num_electron; j++) {
                    double tmp = fs_ii + fs[index(j, j)] - fs_aa - fs[index(b, b)];

                    int index_abij = index(a, b, i, j);

                    double_excitation[index_abij] = spin_ints[index(i, j, a, b)] / tmp;
                    denominator_abij[index_abij] = tmp;
                }
            }
        }
    }
}

inline double CCSD::teimo( int a, int b, int c, int d, double 
                           *two_electron_integral, 
                           int two_electron_integral_size) {
    auto eint = [] (int x, int y) {
        return (x > y) ? (x * (x + 1) / 2 + y) : (y * (y + 1) / 2 + x);
    };

    int index = eint(eint(a, b), eint(c, d));
    if(index > 0 && index < two_electron_integral_size)
        return two_electron_integral[index];
    
    return 0.0;
}

inline double CCSD::teimo(
    int a, int b, int c, int d, 
    std::unordered_map<double, double> two_electron_integral) {
    auto eint = [] (int x, int y) {
        return (x > y) ? (x * (x + 1) / 2 + y) : (y * (y + 1) / 2 + x);
    };

    int index = eint(eint(a, b), eint(c, d));
    if(two_electron_integral.find(index) != two_electron_integral.end())
        return two_electron_integral[index];
    
    return 0.0;
}