#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <omp.h>
#include <stdio.h>
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
#define N 23000 //size of the matrix
#ifndef _OPENMP 
printf(stderr, "OpenMP is not supported – sorry!\n");
exit(0);
#endif




int main() {

    int S = N / 2;
    int i, j, k, l;
    /*create a square matrix A of size N*/
    std::vector<double> A(N * N);
    std::vector<double> A_T(N * N);
    /*randomly generating an array A of size N*/
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10; //elements from 0 to 9
    }

    // print A for debugging
    /*
    std::cout << "Matrix A:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << A[i * N + j] << " ";
        std::cout << std::endl;
    }*/

    //1 blocking level
    auto t2 = high_resolution_clock::now();  

#pragma omp parallel shared(A_T, A) private(i, j, k, l)
    {
#pragma omp for schedule(static) 
        for (i = 0; i < N; i += S) {
            for (j = 0; j < N; j += S) {
                for (k = i; k < i + S; ++k) {
                    for (l = j; l < j + S; ++l) {
                        A_T[k + l * N] = A[l + k * N];
                    }
                }
            }
        }
    }
    auto t3 = high_resolution_clock::now();

    //print A_T for debugging
    /*
    std::cout << "Matrix A transposed:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << A_T[i * N + j] << " ";
        std::cout << std::endl;
    }*/

    duration<double> time_span1 = duration_cast<duration<double>>(t3 - t2);
    std::cout << "It took me " << time_span1.count() << " seconds for single blocked level transpose on a " << N <<"x"<< N << " matrix" << std::endl;
    return 0;

}
