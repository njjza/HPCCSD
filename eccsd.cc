#include "eccsd.h"

CCSD::CCSD( int num_electron, int dimension, 
            const int nuclear_repulsion_energy, int scf_energy, 
            double *orbital_energy, 
            std::map<double, double> two_electron_integral) {
    
    this->num_electron = num_electron;
    this->dimension = dimension * 2;
    this->nuclear_repulsion_energy = nuclear_repulsion_energy;
    this->scf_energy = scf_energy;
    this->orbital_energy = orbital_energy;
    this->two_electron_integral = two_electron_integral;

    double tmp[dimension];
    for (int i = 0; i < dimension; i++) {
        tmp[i] = orbital_energy[i >> 1];
    }

    // 2d array
    int dimensions = dimension * dimension;

    this->fs = new double[dimension];
    for (int i = 0; i < dimension; i++) {
        this->fs[index(i, i)] = tmp[i];
    }

    this->single_excitation = new double[dimensions];            // T1
    this->denominator_ai = new double[dimensions];               // Dai
    
    // diagonalize the orbital energy
    

    // 4d array
    dimensions *= dimensions;
    this->double_excitation = new double[dimensions];            // T2
    this->denominator_abij = new double[dimensions];             // Dabij

    //spin basis double bar integral
    this->spin_ints = new double[dimensions];

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            for (int k = 0; k < dimension; k++) {
                for (int l = 0; l < dimension; l++) {
                    int p = i >> 1;
                    int q = j >> 1;
                    int r = k >> 1;
                    int s = l >> 1;
                    double value1 = two_electron_integral[index(p, r, q, s)] * (i % 2 == k % 2) * (j % 2 == l % 2);
                    double value2 = two_electron_integral[index(p, s, q, r)] * (i % 2 == s % 2) * (j % 2 == r % 2);
                    spin_ints[index(i, j, k, l)] = value1 - value2;
                }
            }
        }
    }
}

double CCSD::run() {
    // run the CCSD
    double ECCSD = 0.0;    // CCSD energy
    double DECC = 0.0;     // CCSD energy difference
    double OLDCC = 0.0;    // old CCSD energy

    int dimensions = this->dimension;
    
    dimensions *= dimensions;
    double *fae = new double[dimensions];
    double *fmi = new double[dimensions];
    double *fme = new double[dimensions];
    
    dimensions *= dimensions;
    
    double *wmnij = new double[dimensions];
    double *wabef = new double[dimensions];
    double *wmbej = new double[dimensions];

    while(DECC > 1.0e-6) {
        double OLDCC = ECCSD;

        // update the intermediate
        this->update_intermediate(fae, fmi, fme, wmnij, wabef, wmbej);

        // make T1
        this->makeT1(fme, fmi, fae);

        // make T2
        this->makeT2(fae, fmi, fme, wabef, wmnij, wmbej);

        // update the energy
        ECCSD = this->update_energy();

        // update the energy difference
        DECC = abs(ECCSD - OLDCC);
    }

    return ECCSD;
}

/**
 * @brief update the intermediate
 * @param fae
 * @param fmi
 * @param fme
 * @param wmnij
 * @return void
 * @note
*/
void CCSD::update_intermediate
(double *fae, double *fmi, double *fme, double *wmnij, double *wabef, double *wmbej) {
    // update the intermediate
    double *ts = this->single_excitation;
    double *td = this->double_excitation;

    // taus = lambda a, b, i, j: td[a,b,i,j] + 0.5*(ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j])
    auto taus = [ts, td, this](int a, int b, int i, int j) {
        return td[index(a, b, i, j)] + \
        0.5*(
            ts[index(a, i)] * ts[index(b, j)] - \
            ts[index(b, i)] * ts[index(a, j)]
        );
    };

    // tau = lambda a, b, i, j: td[a,b,i,j] + ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j]
    auto tau = [ts, td, this](int a, int b, int i, int j) {
        return td[index(a, b, i, j)] + \
        ts[index(a, i)] * ts[index(b, j)] - \
        ts[index(b, i)] * ts[index(a, j)];
    };


    // initialize the intermediate
    for (int a = this->num_electron; a < this -> dimension; a++) {
        for (int e = this->num_electron; e < this -> dimension; e++) {
            int m = a - this->num_electron; 
            int i = e - this->num_electron;
            fae[index(a, e)] = (1 - (a == e)) * fs[index(a, e)];
            fmi[index(m, i)] = (1 - (m == i)) * fs[index(m, i)];
            fme[index(m, e)] = fs[index(m, e)];
        }
    }

    // update the intermediate
    int n_elec = this->num_electron;
    int dimension = this->dimension;
    double *spin_ints = this->spin_ints;
    double *fs = this->fs;

    for (int a = n_elec; a < dimension; a++) {
        for (int b = n_elec; b < dimension; b++) {
            for (int e = n_elec; e < dimension; e++) {
                int tmp = a - this->num_electron;
                int tmp2 = b - this->num_electron;
                int tmp3 = e - this->num_electron;

                fae[index(a, e)] += -0.5 * \
                    fs[index(tmp2, e)] * ts[index(a, tmp2)];
                fmi[index(tmp,tmp2)] += 0.5 * \
                    fs[index(tmp, e)] * ts[index(tmp2, e)];

                for (int f = n_elec; f < dimension; f++) {
                    int tmp4 = f - this->num_electron;
                    
                    fae[index(a, e)] += ts[index(f, tmp2)] * \
                        spin_ints[index(tmp2, a, f, e)];

                    fmi[index(tmp, tmp2)] += ts[index(e, tmp4)] * spin_ints[index(a, f, tmp2, e)];
                    fme[index(tmp, e)] += ts[index(f, tmp2)] * spin_ints[index(a, tmp2, e, tmp4)];
                    wmnij[index(tmp, tmp2, tmp3, tmp4)] = spin_ints[index(a, tmp2, tmp3, tmp4)];
                    
                    for (int n = 0; n < n_elec; n++) {
                        int tmp5 = n + n_elec;
                        fae[index(a, e)] += -0.5 * taus(a, f, tmp2, n) * spin_ints[index(tmp2, tmp5, e, f)];
                        fmi[index(tmp, tmp2)] += 0.5 * taus(e, tmp5, tmp2, tmp4) * spin_ints[index(a, tmp4, e, tmp5)];
                        wmnij[index(tmp, tmp2, tmp3, tmp4)] += ts[index(tmp5, tmp4)] * spin_ints[index(a, tmp2, tmp3, tmp5)] - \
                        ts[index(tmp5, tmp2)] * spin_ints[index(a, tmp4, tmp3, tmp5)];

                        for(int i = 0; i < n_elec; i++) {
                            // Wmnij[a-Nelec,b-Nelec,e-Nelec,f-Nelec] += 0.25*tau(n+Nelec,i+Nelec,e-Nelec,f-Nelec)*spinints[a-Nelec,b-Nelec,n+Nelec,i+Nelec]
                            wmnij[index(tmp, tmp2, tmp3, tmp4)] += 0.25 * tau(tmp5, tmp4, tmp3, tmp2) * spin_ints[index(tmp, tmp2, tmp5, i + n_elec)];
                        }
                    }
                }
            }
        }
    }

    // Wabef = np.zeros((dim,dim,dim,dim))
    // Wmbej = np.zeros((dim,dim,dim,dim))
    // gen_iter.append(range(Nelec,dim))
    // for a, b, e, f in product(*gen_iter):
    //     m, j = a-Nelec, f-Nelec
    //     Wabef[a,b,e,f] = spinints[a,b,e,f]
    //     Wmbej[m,b,e,j] = spinints[m,b,e,j]
    //     for i in range(Nelec):
    //         Wabef[a,b,e,f] += -ts[b,i]*spinints[a,i,e,f] + ts[a,i]*spinints[b,i,e,f]
    //         Wmbej[m,b,e,j] += ts[i+Nelec,j]*spinints[m,b,e,i+Nelec]
    //         Wmbej[m,b,e,j] += -ts[b,i]*spinints[m,i,e,j]
    //         for n in range(Nelec):
    //             Wabef[a,b,e,f] += 0.25*tau(a,b,i,n)*spinints[i,n,e,f]
    //             Wmbej[m,b,e,j] -= (0.5*td[n+Nelec,b,j,i] + ts[n+Nelec,j]*ts[b,i])*spinints[m,i,e,n+Nelec]
    double *wabef = new double[this->dimension * this->dimension * this->dimension * this->dimension];

    return;
}

void CCSD::makeT1
(double *fme, double *fmi, double *fae) {
    int dim = this->dimension;
    int Nelec = this->num_electron;
    double *ts = this->single_excitation;
    double *td = this->double_excitation;
    double *fs = this->fs;

    double tsnew[dim][dim];
    memset(tsnew, 0, dim * dim);
    
    for (int a = Nelec; a<dim; a++) {
        for (int i=0; i<this->num_electron; i++) {
            tsnew[a][i] = fs[index(i, a)];
            
            for (int e = this->num_electron; e<dim; e++) {
                tsnew[a][i] += ts[index(e, i)] * fae[index(a, e)];
            }
            
            for (int m=0; m<Nelec; m++) {
                tsnew[a][i] -= ts[index(a, m)] * fmi[index(m, i)];
                
                for (int e=Nelec; e<dim; e++) {
                    tsnew[a][i] += td[index(a, e, i, m)] * fme[index(m, e)];
                    
                    for (int f=Nelec; f<dim; f++) {
                        tsnew[a][i] -= 0.5 * td[index(e, f, i, m)] * \
                            this->spin_ints[index(m, a, e, f)];
                    }
                    
                    for (int n=0; n<Nelec; n++) {
                        tsnew[a][i] -= 0.5 * td[index(a, e, m, n)] * \
                            this->spin_ints[index(n, m, e, i)];
                    }
                }
                
                for (int n=0; n<Nelec; n++) {
                    for (int f=Nelec; f<dim; f++) {
                        tsnew[a][i] -= ts[index(f, n)] * \
                            this->spin_ints[index(n, a, i, f)];
                    }
                }
            }

            tsnew[a][i] /= this->denominator_ai[index(a, i)];
        }
    }

    memcpy(ts, tsnew, dim * dim);
}

void CCSD::makeT2
(   double *fae, double *fmi, double *fme, double *wabef, double *wmnij, 
    double *wmbej
) {
    // make T2
    int dim = this->dimension;
    int Nelec = this->num_electron;
    double *ts = this->single_excitation;
    double *td = this->double_excitation;
    double *spin_ints = this->spin_ints;

    double tdnew[dim][dim][dim][dim];
    memset(tdnew, 0, dim * dim * dim * dim);

    auto tau = [ts, td, this](int a, int b, int i, int j) {
        return td[index(a, b, i, j)] + \
        ts[index(a, i)] * ts[index(b, j)] - \
        ts[index(b, i)] * ts[index(a, j)];
    };

    for (int a = Nelec; a < dim; a++) {
        for (int b = Nelec; b < dim; b++) {
            for (int i = 0; i < Nelec; i++) {
                for (int j = 0; j < Nelec; j++) {
                    tdnew[a][b][i][j] = spin_ints[index(i, j, a, b)];

                    for (int e = Nelec; e < dim; e++) {
                        tdnew[a][b][i][j] += \
                            td[index(a, e, i, j)] * fae[index(b, e)] - \
                            td[index(b, e, i, j)] * fae[index(a, e)];

                        for (int m = 0; m < Nelec; m++) {
                            tdnew[a][b][i][j] += 0.5 * (
                                td[index(a, e, i, j)] * ts[index(b, m)] * fme[index(m, e)] + \
                                td[index(b, e, i, j)] * ts[index(a, m)] * fme[index(m, e)]
                            );
                        }
                    }

                    for (int m = 0; m < Nelec; m++) {
                        tdnew[a][b][i][j] += \
                            -td[index(a, b, i, m)] * fmi[index(m, j)] + \
                            td[index(a, b, j, m)] * fmi[index(m, i)];

                        for (int e = Nelec; e < dim; e++) {
                            tdnew[a][b][i][j] += 0.5 * (
                                -td[index(a, b, i, m)] * ts[index(e, j)] * fme[index(m, e)] + \
                                td[index(a, b, j, m)] * ts[index(e, i)] * fme[index(m, e)]
                            );
                        }
                    }
                    
                    for (int e = Nelec; e < dim; e++) {
                        tdnew[a][b][i][j] += \
                            ts[index(e, i)] * spin_ints[index(a, b, e, j)] - \
                            ts[index(e, j)] * spin_ints[index(a, b, e, i)];
                        
                        for (int f = Nelec; f < dim; f++) {
                            tdnew[a][b][i][j] += 0.5 * tau(e, f, i, j) *
                                wabef[index(a, b, e, f)];
                        }
                    }

                    for (int m = 0; m < Nelec; m++) {
                        tdnew[a][b][i][j] += \
                            -ts[index(a, m)] * spin_ints[index(m, b, i, j)] + \
                            ts[index(b, m)] * spin_ints[index(m, a, i, j)];

                        for (int e = Nelec; e < dim; e++) {
                            tdnew[a][b][i][j] += (
                                td[index(a, e, i, m)] * wmbej[index(m, b, e, j)] - ts[index(e, i)] * ts[index(a, m)] * spin_ints[index(m, b, e, j)] + \
                                -td[index(a, e, j, m)] * wmbej[index(m, b, e, i)] + ts[index(e, j)] * ts[index(a, m)] * spin_ints[index(m, b, e, i)] + \
                                -td[index(b, e, i, m)] * wmbej[index(m, a, e, j)] + ts[index(e, i)] * ts[index(b, m)] * spin_ints[index(m, a, e, j)] + \
                                td[index(b, e, j, m)] * wmbej[index(m, a, e, i)] - ts[index(e, j)] * ts[index(b, m)] * spin_ints[index(m, a, e, i)]
                            );
                        }

                        for (int n = 0; n < Nelec; n++) {
                            tdnew[a][b][i][j] += 0.5 * tau(a, b, m, n) *
                                wmnij[index(a, b, m, n)];
                        }
                    }

                    tdnew[a][b][i][j] /= this->denominator_abij[index(a, b, i, j)];
                }
            }
        }
    }

    memcpy(td, tdnew, dim * dim * dim * dim);

}

double CCSD::update_energy() {
    int dim = this->dimension;
    int Nelec = this->num_electron;
    double *ts = this->single_excitation;
    double *td = this->double_excitation;
    double *spin_ints = this->spin_ints;

    double energy = 0.0;

    for (int i = Nelec; i < dim; i++) {
        for (int a = Nelec; a < dim; a++) {
            for (int j = 0; j < Nelec; j++) {
                for (int b = 0; b < Nelec; b++) {
                    energy += 0.25 * spin_ints[index(i, j, a, b)] * td[index(a, b, i, j)];
                    energy += 0.5 * spin_ints[index(i, j, a, b)] * ts[index(a, i)] * ts[index(b, j)];
                }
            }
        }
    }

    return energy;
}