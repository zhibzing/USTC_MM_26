// Copyright 2026, Yumeng Liu @ USTC
// op_1: Image Compression
//
// Deps  : OpenCV, STL
// Usage : ./op1_template [image_path]
//
// TODO: Implement ImageCompression() below.

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <numeric>

// ============================================================
// TODO: Implement ImageCompression
// ============================================================
//
// Image compression using SVD.
//
// You may define any helper functions above this function.
//
// Args:
//   img         — input BGR image (CV_8UC3)
//   rank        — target rank (positive integer)
//
// Returns: resized image of rank `rank` (CV_8UC3).

// 检查矩阵是否为上双对角（除主对角线和第一条超对角线外全零）
void check_bidiagonal(const cv::Mat& B, double tol = 1e-12) {
    int m = B.rows, n = B.cols;
    int violations = 0;
    double max_val = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j || j == i + 1) continue;   // 允许 \(B(i,i)\) 和 \(B(i,i+1)\)
            double val = std::abs(B.at<double>(i, j));
            if (val > tol) {
                ++violations;
                max_val = std::max(max_val, val);
            }
        }
    }
    std::cout << "Bidiagonal check: " << violations << " off-pattern elements > " 
              << tol << ", max = " << max_val << std::endl;
}

// 检查矩阵 Q 的正交性：‖Q^T Q - I‖_F
double check_orthogonality(const cv::Mat& Q) {
    cv::Mat I = cv::Mat::eye(Q.cols, Q.cols, CV_64F);
    cv::Mat QtQ = Q.t() * Q;
    cv::Mat diff = QtQ - I;
    return cv::norm(diff, cv::NORM_L2);   // Frobenius 范数
}

void householder_bidiag_vec(const cv::Mat& A, cv::Mat& U, cv::Mat& B, cv::Mat& V) {
    int m = A.rows, n = A.cols;
    int p = std::min(m, n);
    
    A.convertTo(B, CV_64F);
    U = cv::Mat::eye(m, m, CV_64F);
    V = cv::Mat::eye(n, n, CV_64F);
    
    for (int j = 0; j < p; j++) {
        int len_l = m - j;
        cv::Mat x_l = B(cv::Rect(j, j, 1, len_l)).clone();
        
        double x0_l = x_l.at<double>(0);
        double sigma_l = 0.0;
        if (len_l > 1) {
            sigma_l = cv::norm(x_l.rowRange(1, len_l));
            sigma_l *= sigma_l;
        }
        
        if (!(sigma_l == 0.0 && x0_l >= 0)) {
            cv::Mat v_l = cv::Mat::zeros(len_l, 1, CV_64F);
            double mu_l = std::sqrt(x0_l * x0_l + sigma_l);
            
            if (x0_l <= 0.0) {
                v_l.at<double>(0) = x0_l - mu_l;
            } else {
                v_l.at<double>(0) = -sigma_l / (x0_l + mu_l);
            }
            if (len_l > 1) {
                x_l.rowRange(1, len_l).copyTo(v_l.rowRange(1, len_l));
            }
            
            double norm_v = cv::norm(v_l);
            double beta = 2.0 / (norm_v * norm_v);
            
            cv::Mat v_full = cv::Mat::zeros(m, 1, CV_64F);
            v_l.copyTo(v_full.rowRange(j, m));

            cv::Mat w_B = beta * v_full.t() * B;
            B -= v_full * w_B;

            cv::Mat w_U = beta * U * v_full;
            U -= w_U * v_full.t(); 
        }
        
        if (j >= n - 1) continue;
        
        int len_r = n - j - 1;
        cv::Mat x_r = B(cv::Rect(j + 1, j, len_r, 1)).clone();
        
        double x0_r = x_r.at<double>(0);
        double sigma_r = 0.0;
        if (len_r > 1) {
            sigma_r = cv::norm(x_r.colRange(1, len_r));
            sigma_r *= sigma_r;
        }
        
        if (!(sigma_r == 0.0 && x0_r >= 0)) {
            cv::Mat v_r = cv::Mat::zeros(1, len_r, CV_64F);
            double mu_r = std::sqrt(x0_r * x0_r + sigma_r);
            
            if (x0_r <= 0.0) {
                v_r.at<double>(0) = x0_r - mu_r;
            } else {
                v_r.at<double>(0) = -sigma_r / (x0_r + mu_r);
            }
            if (len_r > 1) {
                x_r.colRange(1, len_r).copyTo(v_r.colRange(1, len_r));
            }
            
            double norm_v = cv::norm(v_r);
            double beta = 2.0 / (norm_v * norm_v);
            
            cv::Mat v_full = cv::Mat::zeros(1, n, CV_64F);
            v_r.copyTo(v_full.colRange(j + 1, n));
            
            cv::Mat w_B = beta * B * v_full.t();
            B -= w_B * v_full;
            
            cv::Mat w_V = beta * V * v_full.t();
            V -= w_V * v_full;
        }
    }
}

void direct_qr(cv::Mat& B, cv::Mat& U, cv::Mat& V) {
    cv::Mat S, Vt;
    cv::SVD::compute(B, S, U, Vt, cv::SVD::FULL_UV);
    V = Vt.t();

    int m = B.rows, n = B.cols;
    int p = std::min(m, n);
    B = cv::Mat::zeros(m, n, CV_64F);
    for (int i = 0; i < p; i++)
        B.at<double>(i, i) = S.at<double>(i);
}

double secular_f(const double lambda, const std::vector<double>& d, const std::vector<double>& z, double off_diag) {
    double sum = 0.0;
    for (size_t i = 0; i < d.size(); i++) {
        double denom = d[i] - lambda;
        if (std::abs(denom) < 1e-12) {
            denom = (denom >= 0) ? 1e-12 : -1e-12;
        }
        sum += z[i] * z[i] / denom;
    }
    return 1.0 + off_diag * off_diag * sum;
}

double secular_df(const double lambda, const std::vector<double>& d, const std::vector<double>& z, double off_diag) {
    double sum = 0.0;
    for (size_t i = 0; i < d.size(); i++) {
        double denom = d[i] - lambda;
        if (std::abs(denom) < 1e-12) {
            denom = (denom >= 0) ? 1e-12 : -1e-12;
        }
        sum += z[i] * z[i] / (denom * denom);
    }
    return off_diag * off_diag * sum;
}

double secular_ddf(const double lambda, const std::vector<double>& d, const std::vector<double>& z, double off_diag) {
    double sum = 0.0;
    for (size_t i = 0; i < d.size(); i++) {
        double denom = d[i] - lambda;
        if (std::abs(denom) < 1e-12) {
            denom = (denom >= 0) ? 1e-12 : -1e-12;
        }
        sum += z[i] * z[i] / (denom * denom * denom);
    }
    return 2.0 * off_diag * off_diag * sum;
}

std::vector<double> solve_Secular(std::vector<double>& d, std::vector<double>& z, double off_diag) {
    int n = d.size();
    std::vector<double> lambda(n);

    // 如果输入为降序则反转为升序，求解后再反转回去
    bool is_desc = (n > 1 && d[0] > d[1]);
    std::vector<double> d_work = d;
    std::vector<double> z_work = z;
    if (is_desc) {
        std::reverse(d_work.begin(), d_work.end());
        std::reverse(z_work.begin(), z_work.end());
    }
    // 此时 d_work 为升序: d_work[0] < d_work[1] < ... < d_work[n-1]

    for (int i = 0; i < n; ++i) {
        double left, right;
        double lambda_i;

        if (i == n - 1) {                      // 最右区间 (d_work[n-1], +∞)
            left  = d_work[n-1];
            right = std::numeric_limits<double>::max();
            double sum_z2 = 0.0;
            for (double val : z_work) sum_z2 += val * val;
            lambda_i = d_work[n-1] + off_diag * off_diag * sum_z2;
        } else {                               // 内部区间 (d_work[i], d_work[i+1])
            left  = d_work[i];
            right = d_work[i+1];
            double w1 = z_work[i] * z_work[i];
            double w2 = z_work[i+1] * z_work[i+1];
            lambda_i = (d_work[i] * w2 + d_work[i+1] * w1) / (w1 + w2);
        }

        // 拉盖尔/牛顿混合迭代，带动态区间保护
        for (int iter = 0; iter < 100; ++iter) {
            double f   = secular_f(lambda_i, d_work, z_work, off_diag);
            double df  = secular_df(lambda_i, d_work, z_work, off_diag);
            double ddf = secular_ddf(lambda_i, d_work, z_work, off_diag);

            if (std::abs(f) < 1e-12) break;
            if (i < n-1 && right - left < 1e-12) break;  // 仅内部区间检查宽度

            // 动态区间收缩（函数单调递增）
            if (f < 0) left  = lambda_i;      // 根在右侧，左边界右移
            else       right = lambda_i;      // 根在左侧，右边界左移

            // 计算迭代步长（拉盖尔优先，失败则牛顿）
            double step;
            double radicand = (n - 1) * ((n - 1) * df * df - n * f * ddf);
            bool use_laguerre = (radicand >= 0.0 && std::abs(df) > 1e-12);

            if (use_laguerre) {
                double sqrt_val = std::sqrt(radicand);
                double denom = df + (df >= 0 ? sqrt_val : -sqrt_val);
                if (std::abs(denom) < 1e-12) {
                    use_laguerre = false;
                } else {
                    step = n * f / denom;
                }
            }
            if (!use_laguerre) {
                step = (std::abs(df) > 1e-12) ? (f / df) : 0.0;
            }

            double lambda_next = lambda_i - step;

            // 越界保护
            if (lambda_next <= left || lambda_next >= right) {
                if (i == n-1) {   // 最右区间无右界，若向左越界则单侧保护
                    if (lambda_next <= left)
                        lambda_next = lambda_i + std::max(0.5 * (lambda_i - left), 1.0);
                } else {          // 内部区间，直接二分
                    lambda_next = (left + right) / 2.0;
                }
            }

            // 极点避让
            if (std::abs(lambda_next - d_work[i]) < 1e-12)
                lambda_next = d_work[i] + 1e-12 * std::abs(d_work[i]);
            if (i < n-1 && std::abs(lambda_next - d_work[i+1]) < 1e-12)
                lambda_next = d_work[i+1] - 1e-12 * std::abs(d_work[i+1]);

            lambda_i = lambda_next;
        }
        lambda[i] = lambda_i;
    }

    // 如果原始输入是降序，反转回降序
    if (is_desc) {
        std::reverse(lambda.begin(), lambda.end());
    }
    return lambda;
}

void merge_blocks(cv::Mat& B, cv::Mat& U, cv::Mat& V,
                  const cv::Mat& B1, const cv::Mat& B2,
                  const cv::Mat& U1, const cv::Mat& U2,
                  const cv::Mat& V1, const cv::Mat& V2,
                  double off_diag) {
    int mid = B1.rows;
    int m = B.rows;
    int n = B.cols;
    int p = std::min(m, n);

    U = cv::Mat::zeros(m, m, CV_64F);
    V = cv::Mat::zeros(n, n, CV_64F);
    U1.copyTo(U(cv::Rect(0, 0, mid, mid)));
    U2.copyTo(U(cv::Rect(mid, mid, m - mid, m - mid)));
    V1.copyTo(V(cv::Rect(0, 0, mid, mid)));
    V2.copyTo(V(cv::Rect(mid, mid, n - mid, n - mid)));
    
    cv::Mat D = cv::Mat::zeros(m, n, CV_64F);
    for (int i = 0; i < mid; i++) {
        D.at<double>(i, i) = B1.at<double>(i, i);
    }
    for (int i = 0; i < p - mid; i++) {
        D.at<double>(mid + i, mid + i) = B2.at<double>(i, i);
    }
    D.at<double>(mid - 1, mid) = off_diag;

    cv::Mat U_sub = cv::Mat::eye(2, 2, CV_64F);
    cv::Mat V_sub = cv::Mat::eye(2, 2, CV_64F);
    cv::Mat B_sub = cv::Mat::zeros(2, 2, CV_64F);
    B_sub.at<double>(0, 0) = D.at<double>(mid - 1, mid - 1);
    B_sub.at<double>(0, 1) = D.at<double>(mid - 1, mid);
    B_sub.at<double>(1, 0) = D.at<double>(mid, mid - 1);
    B_sub.at<double>(1, 1) = D.at<double>(mid, mid);
    
    direct_qr(B_sub, U_sub, V_sub);
}

void divide_conquer_svd(cv::Mat& B, cv::Mat& U, cv::Mat& V) {
    int m = B.rows;
    int n = B.cols;
    int p = std::min(m, n);

    if (p == 0) return;
    
    if (p <= 2) {
        direct_qr(B, U, V);
        return;
    }
    
    int mid = p / 2;
    
    cv::Mat B1 = B(cv::Rect(0, 0, mid, mid)).clone();
    cv::Mat B2 = B(cv::Rect(mid, mid, n - mid, m - mid)).clone();
    double off_diag = B.at<double>(mid - 1, mid);
    
    cv::Mat U1 = cv::Mat::eye(mid, mid, CV_64F);
    cv::Mat V1 = cv::Mat::eye(mid, mid, CV_64F);
    cv::Mat U2 = cv::Mat::eye(m - mid, m - mid, CV_64F);
    cv::Mat V2 = cv::Mat::eye(n - mid, n - mid, CV_64F);
    
    divide_conquer_svd(B1, U1, V1);
    divide_conquer_svd(B2, U2, V2);
    
    merge_blocks(B, U, V, B1, B2, U1, U2, V1, V2, off_diag);
}

// std::vector<double> svd_algorithm(const cv::Mat& A, cv::Mat& U, cv::Mat& V) {
//     cv::Mat B;
//     householder_bidiag_vec(A, U, B, V);
//     cv::Mat recon_householder = U * B * V.t();
//     double err_householder = cv::norm(A - recon_householder);
//     std::cout << "Householder bidiagonalization reconstruction error: " << err_householder << std::endl;
    
//     int m = B.rows, n = B.cols;
//     int p = std::min(m, n);
    
//     cv::Mat U_svd, V_svd;
//     if (p <= 4) {
//         direct_qr(B, U_svd, V_svd);
//     } else {
//         divide_conquer_svd(B, U_svd, V_svd);
//     }

//     U = U * U_svd;
//     V = V * V_svd;

//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             if (i != j && std::abs(B.at<double>(i, j)) > 1e-10) {
//                 std::cout << "Warning: B is not perfectly diagonal. Off-diagonal element at (" << i << ", " << j << ") = " << B.at<double>(i, j) << std::endl;
//             }
//         }
//     }
//     std::cout << "Have already tested B is diagonal." << std::endl;
    
//     std::vector<double> S(p);
//     for (int i = 0; i < p; i++) {
//         S[i] = std::abs(B.at<double>(i, i));
//     }
//     cv::Mat recon_unsorted = U * B * V.t();
//     double err_unsorted = cv::norm(A - recon_unsorted);
//     std::cout << "SVD reconstruction error before sorted: " << err_unsorted << std::endl;
    
//     // sort singular values and corresponding singular vectors
//     std::vector<std::pair<double, int>> idx(p);
//     for (int i = 0; i < p; i++) {
//         idx[i] = {S[i], i};
//     }
//     std::sort(idx.begin(), idx.end(), 
//               [](auto& a, auto& b) { return a.first > b.first; });
    
//     std::vector<double> S_sorted(p);
//     cv::Mat U_sorted = U.clone();
//     cv::Mat V_sorted = V.clone();
    
//     for (int i = 0; i < p; i++) {
//         int old_idx = idx[i].second;
//         S_sorted[i] = idx[i].first;
//         U.col(old_idx).copyTo(U_sorted.col(i));
//         V.col(old_idx).copyTo(V_sorted.col(i));
//     }
    
//     U = U_sorted;
//     V = V_sorted;

//     cv::Mat S_mat = cv::Mat::zeros(m, n, CV_64F);
//     for (int i = 0; i < p; i++) {
//         S_mat.at<double>(i, i) = S_sorted[i];
//     }
//     cv::Mat recon_sorted = U * S_mat * V.t();
//     double err_sorted = cv::norm(A - recon_sorted);
//     std::cout << "SVD reconstruction error after sorted: " << err_sorted << std::endl;
    
//     return S_sorted;
// }

std::vector<double> svd_algorithm(const cv::Mat& A, cv::Mat& U, cv::Mat& V) {
    cv::Mat B;
    householder_bidiag_vec(A, U, B, V);

    // ---- 检测 1：双对角化后的矩阵形态 ----
    std::cout << "After householder bidiagonalization:" << std::endl;
    check_bidiagonal(B, 1e-10);

    // ---- 检测 2：双对角化重构误差（相对误差） ----
    cv::Mat recon_householder = U * B * V.t();
    double err_householder = cv::norm(A - recon_householder) / cv::norm(A);
    std::cout << "Householder reconstruction relative error: " << err_householder << std::endl;

    int m = B.rows, n = B.cols;
    int p = std::min(m, n);

    cv::Mat U_svd, V_svd;
    if (p <= 4) {
        direct_qr(B, U_svd, V_svd);
    } else {
        divide_conquer_svd(B, U_svd, V_svd);
    }
    // direct_qr(B, U_svd, V_svd);

    // ---- 检测 3：SVD 后 B 是否对角化 ----
    double max_off = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double val = std::abs(B.at<double>(i, j));
                if (val > 1e-10) {
                    std::cout << "Warning: B is not perfectly diagonal. Off-diagonal at (" 
                              << i << ", " << j << ") = " << val << std::endl;
                }
                if (val > max_off) max_off = val;
            }
        }
    }
    std::cout << "Max off-diagonal in B after SVD: " << max_off << std::endl;

    // ---- 检测 4：U_svd 和 V_svd 的正交性 ----
    double orthU = check_orthogonality(U_svd);
    double orthV = check_orthogonality(V_svd);
    std::cout << "SVD core orthogonality: ||U^TU - I|| = " << orthU 
              << ", ||V^TV - I|| = " << orthV << std::endl;

    // 累积全局 U/V
    U = U * U_svd;
    V = V * V_svd;

    // ---- 检测 5：累积后的全局正交性 ----
    double orthU_full = check_orthogonality(U);
    double orthV_full = check_orthogonality(V);
    std::cout << "Global orthogonality after accumulation: ||U^TU - I|| = " << orthU_full 
              << ", ||V^TV - I|| = " << orthV_full << std::endl;

    // 提取奇异值（取绝对值）
    std::vector<double> S(p);
    for (int i = 0; i < p; i++) {
        S[i] = std::abs(B.at<double>(i, i));
    }

    // 排序前重构误差
    cv::Mat recon_unsorted = U * B * V.t();
    double err_unsorted = cv::norm(A - recon_unsorted) / cv::norm(A);
    std::cout << "SVD reconstruction relative error (before sort): " << err_unsorted << std::endl;

    // 排序（保持原有逻辑不变）
    std::vector<std::pair<double, int>> idx(p);
    for (int i = 0; i < p; i++) {
        idx[i] = {S[i], i};
    }
    std::sort(idx.begin(), idx.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    std::vector<double> S_sorted(p);
    cv::Mat U_sorted = U.clone();
    cv::Mat V_sorted = V.clone();

    for (int i = 0; i < p; i++) {
        int old_idx = idx[i].second;
        S_sorted[i] = idx[i].first;
        U.col(old_idx).copyTo(U_sorted.col(i));
        V.col(old_idx).copyTo(V_sorted.col(i));
    }

    U = U_sorted;
    V = V_sorted;

    cv::Mat S_mat = cv::Mat::zeros(m, n, CV_64F);
    for (int i = 0; i < p; i++) {
        S_mat.at<double>(i, i) = S_sorted[i];
    }
    cv::Mat recon_sorted = U * S_mat * V.t();
    double err_sorted = cv::norm(A - recon_sorted) / cv::norm(A);
    std::cout << "SVD reconstruction relative error (after sort): " << err_sorted << std::endl;

    return S_sorted;
}

std::vector<std::vector<double>> calculateSVD(const cv::Mat& img, std::vector<cv::Mat>& U, std::vector<cv::Mat>& V) {
    std::vector<std::vector<double>> svdResults(3, std::vector<double>());

    cv::Mat channels[3];
    cv::split(img, channels);

    for (int c = 0; c < 3; c++) {
        cv::Mat A;
        channels[c].convertTo(A, CV_64F);

        std::vector<double> S;
        S = svd_algorithm(A, U[c], V[c]);

        svdResults[c] = S;
    }

    return svdResults;
}

// std::vector<std::vector<double>> calculateSVD(const cv::Mat& img, std::vector<cv::Mat>& U, std::vector<cv::Mat>& V) {
//     std::vector<std::vector<double>> svdResults(3);
//     cv::Mat channels[3];
//     cv::split(img, channels);
//     for (int c = 0; c < 3; c++) {
//         cv::Mat A;
//         channels[c].convertTo(A, CV_64F);
//         cv::Mat S, Vt;
//         cv::SVD::compute(A, S, U[c], Vt, cv::SVD::FULL_UV);
//         // OpenCV returns V^T, we need V
//         V[c] = Vt.t();
//         std::vector<double> sv(S.rows);
//         for (int i = 0; i < S.rows; i++) sv[i] = S.at<double>(i);
//         svdResults[c] = sv;
//     }
//     return svdResults;
// }

cv::Mat ImageCompression(const std::vector<std::vector<double>>& S, const std::vector<cv::Mat>& U, const std::vector<cv::Mat>& V, const int rank) {
    std::vector<cv::Mat> channels(3);

    for (int c = 0; c < 3; c++) {
        cv::Mat S_mat = cv::Mat::zeros(U[c].rows, V[c].cols, CV_64F);
        for (int i = 0; i < rank && i < S[c].size(); i++) {
            S_mat.at<double>(i, i) = S[c][i];
        }
        cv::Mat recon = U[c] * S_mat * V[c].t();
        recon.convertTo(channels[c], CV_8U);
    }
    cv::Mat compressedImg;
    cv::merge(channels, compressedImg);
    return compressedImg;
}

// ============================================================
// GUI
// ============================================================

static cv::Mat g_src, g_dst;
static int g_src_rank = 0, g_dst_rank = 0;

static void refresh() {
    const int PAD = 20, HDR = 36;
    int h = g_src.rows + HDR;
    int w = g_src.cols;
    if (!g_dst.empty()) {
        h = std::max(g_src.rows, g_dst.rows) + HDR;
        w = g_src.cols + g_dst.cols + PAD;
    }

    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(45, 45, 45));
    g_src.copyTo(canvas(cv::Rect(0, HDR, g_src.cols, g_src.rows)));
    cv::putText(canvas,
        "Input rank [" + std::to_string(g_src_rank) + "]",
        cv::Point(4, HDR - 10), cv::FONT_HERSHEY_SIMPLEX, 0.55,
        cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

    if (!g_dst.empty()) {
        int xoff = g_src.cols + PAD;
        g_dst.copyTo(canvas(cv::Rect(xoff, HDR, g_dst.cols, g_dst.rows)));
        cv::putText(canvas,
            "Result rank [" + std::to_string(g_dst_rank) + "]",
            cv::Point(xoff + 4, HDR - 10), cv::FONT_HERSHEY_SIMPLEX, 0.55,
            cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
    } else {
        cv::putText(canvas, "Adjust sliders then press [Space]",
            cv::Point(10, h / 2 + 10), cv::FONT_HERSHEY_SIMPLEX, 0.65,
            cv::Scalar(80, 220, 80), 2, cv::LINE_AA);
    }
    cv::imshow("Image Compression", canvas);
}

int main(int argc, char* argv[]) {
    std::string path;
    if (argc > 1) {
        path = argv[1];
        g_src = cv::imread(path, cv::IMREAD_COLOR);
    } else {
        for (const char* p : {"../figs/original.png", "../../figs/original.png", "../../../figs/original.png"}) {
            g_src = cv::imread(p, cv::IMREAD_COLOR);
            if (!g_src.empty()) { path = p; break; }
        }
        if (g_src.empty()) path = "../figs/original.png";
    }
    if (g_src.empty()) {
        std::cerr << "Cannot open image: " << path << "\n"
                  << "Usage: op2_template [image_path]\n";
        return 1;
    }

    std::vector<cv::Mat> U(3), V(3);
    std::vector<std::vector<double>> S = calculateSVD(g_src, U, V);
    g_src_rank = std::min({S[0].size(), S[1].size(), S[2].size()});

    std::cout << "Image: " << g_src.cols << " x " << g_src.rows << " px\n"
              << "Keys : [Space] run  [s] save  [r] reset  [q/Esc] quit\n";

    cv::namedWindow("Image Compression", cv::WINDOW_NORMAL);
    int win_w = std::min(g_src.cols * 2 + 140, 1600);
    cv::resizeWindow("Image Compression", win_w, g_src.rows + 120);

    cv::createTrackbar("Rank", "Image Compression", &g_dst_rank, g_src_rank);
    cv::setTrackbarPos("Rank", "Image Compression", g_src_rank);

    refresh();

    while (true) {
        int key = cv::waitKey(30) & 0xFF;
        if (key == 27 || key == 'q') break;

        if (key == ' ') {
            int tgt_rank   = std::max(1, g_dst_rank);
            std::cout << "Running: rank " << tgt_rank << " ...\n";
            g_dst = ImageCompression(S, U, V, tgt_rank);
            std::cout << "Done.\n";
            refresh();
        }

        if (key == 'r') {
            g_dst = cv::Mat();
            cv::setTrackbarPos("Rank", "Image Compression", g_src_rank);
            refresh();
        }

        if (key == 's' && !g_dst.empty()) {
            cv::imwrite("result.png", g_dst);
            std::cout << "Saved result.png\n";
        }
    }
    return 0;
}