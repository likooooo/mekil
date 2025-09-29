#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <cpu_backend.hpp>
#include <mkl_basic_operator.h>

// Helper function to print a matrix for easy debugging
void print_matrix(const std::vector<double>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}

// Helper function for matrix-vector multiplication
std::vector<double> mat_vec_mult(const std::vector<double>& A, int rows, int cols,
                                 const std::vector<double>& v) {
    if (v.size() != (size_t)cols) {
        throw std::invalid_argument("Matrix and vector dimensions do not match.");
    }

    std::vector<double> result(rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += A[i * cols + j] * v[j];
        }
    }
    return result;
}

// Helper function for vector-vector dot product
double dot(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
}

// Helper function for vector norm
double norm(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

// Helper function for vector scaling
std::vector<double> scale_vec(const std::vector<double>& v, double scalar) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

// Helper function for vector subtraction
std::vector<double> subtract_vecs(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }
    std::vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

std::pair<std::vector<double>, std::vector<double>>
arnoldi_iteration(const std::vector<double>& A, int m, int n,
                  const std::vector<double>& b, int kmax,
                  double eps = 1e-12, bool A_is_hermitian_matrix = false) {
    // h: (kmax+1) x kmax
    std::vector<double> h((kmax + 1) * kmax, 0.0);
    // Q: m x (kmax+1)
    std::vector<double> Q(m * (kmax + 1), 0.0);

    // Normalize the input vector
    double b_norm = norm(b);
    for (int i = 0; i < m; ++i) {
        Q[i * (kmax + 1) + 0] = b[i] / b_norm;
    }

    for (int k = 1; k <= kmax; ++k) {
        std::vector<double> v(m);
        for (int i = 0; i < m; ++i) {
            v[i] = Q[i * (kmax + 1) + (k - 1)];
        }
        v = mat_vec_mult(A, m, n, v);

        int start_j = A_is_hermitian_matrix ? std::max(0, k - 2) : 0;

        for (int j = start_j; j < k; ++j) {
            std::vector<double> Q_col_j(m);
            for (int i = 0; i < m; ++i) {
                Q_col_j[i] = Q[i * (kmax + 1) + j];
            }
            h[j * kmax + (k - 1)] = dot(Q_col_j, v);
            std::vector<double> proj_v = scale_vec(Q_col_j, h[j * kmax + (k - 1)]);
            v = subtract_vecs(v, proj_v);
        }

        h[k * kmax + (k - 1)] = norm(v);
        if (h[k * kmax + (k - 1)] > eps) {
            for (int i = 0; i < m; ++i) {
                Q[i * (kmax + 1) + k] = v[i] / h[k * kmax + (k - 1)];
            }
        } else {
            return {Q, h};
        }
    }
    return {Q, h};
}

int main() {
    int n = 4;
    std::vector<double> A = {
        5, 4, 1, 1,
        4, 6, 2, 1,
        1, 2, 7, 2,
        1, 1, 2, 8
    };

    // Generate a random initial vector
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<double> b(n);
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = dis(gen);
    }

    // Run Arnoldi iteration
    auto [Q, h] = arnoldi_iteration(A, n, n, b, n, 1e-12, false);

    // Print the results
    std::cout << "A=\n";
    print_matrix(A, n, n);

    std::cout << "\nupper Hessenberg=\n";
    print_matrix(h, n + 1, n);

    std::cout << "\nH_n (n x n block of Hessenberg)=\n";
    std::vector<double> H_n(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H_n[i * n + j] = h[i * n + j];
        }
    }
    print_matrix(H_n, n, n);

    std::vector<double> w(n); 
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U',
                             n, H_n.data(), n, w.data());

    std::reverse(w.begin(), w.end());
    if (info == 0) {
        std::cout << "\nEigenvalues:\n";
        for (int i = 0; i < n; i++) {
            std::cout << w[i] << "\n";
        }
    } else {
        std::cerr << "Error: dsyev failed with info = " << info << "\n";
    }
    return 0;
}
