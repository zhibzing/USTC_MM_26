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
#include <numeric>   // for std::iota

// ============================================================
// TODO: Implement ImageCompression
// ============================================================
//
// Image compression using SVD.
//
// You may define any helper functions above this function.
//
// Args:
//   img     — input BGR image (CV_8UC3)
//   rank    — target rank (positive integer)
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

// ============================================================
// 分治 SVD 合并模块（严格按照 LAPACK DLAED1-4 算法）
// ============================================================

// 计算 secular 方程的单个根（对应 DLAED4 的核心，但仅适用无压缩情况）
// 输入：d (升序极点), z (归一化向量), rho, 区间 (left, right)
// 输出：根 lambda
double solve_secular_root(const std::vector<double>& d,
                          const std::vector<double>& z,
                          double rho,
                          double left,
                          double right,
                          int max_iter = 30) {
    double lambda = (left + right) * 0.5;
    for (int iter = 0; iter < max_iter; ++iter) {
        double f = 1.0;
        double df = 0.0;
        for (size_t i = 0; i < d.size(); ++i) {
            double den = d[i] - lambda;
            double term = z[i] * z[i] / den;
            f += rho * term;
            df -= rho * z[i] * z[i] / (den * den);
        }
        if (std::abs(f) < 1e-12) break;
        double step = f / df;
        double lambda_new = lambda - step;
        // 保证新根在区间内
        if (lambda_new <= left) lambda_new = (left + right) * 0.5;
        if (lambda_new >= right) lambda_new = (left + right) * 0.5;
        // 避免触及极点
        for (size_t i = 0; i < d.size(); ++i) {
            if (std::abs(lambda_new - d[i]) < 1e-14) {
                lambda_new = d[i] + (lambda_new > d[i] ? 1e-12 : -1e-12);
            }
        }
        lambda = lambda_new;
        // 更新区间
        if (f < 0) left = lambda;
        else right = lambda;
    }
    return lambda;
}

// 合并两个子问题的 SVD（正确实现）
void merge_blocks(cv::Mat& B, cv::Mat& U, cv::Mat& V,
                  const cv::Mat& B1, const cv::Mat& B2,
                  const cv::Mat& U1, const cv::Mat& U2,
                  const cv::Mat& V1, const cv::Mat& V2,
                  double off_diag) {
    int m = B.rows, n = B.cols;
    int p = std::min(m, n);          // 有效秩
    int k = B1.rows;                 // 子问题1的维数
    int n2 = p - k;                  // 子问题2的维数

    // ---------------------------
    // 1. 构造对称三对角特征问题的数据
    // ---------------------------
    // 极点 D = [diag(B1); diag(B2)] 并升序排列
    std::vector<double> d(p);
    for (int i = 0; i < k; ++i) d[i] = B1.at<double>(i, i);
    for (int i = 0; i < n2; ++i) d[k + i] = B2.at<double>(i, i);
    std::vector<int> idx(p);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return d[i] < d[j]; });
    std::vector<double> d_sorted(p);
    for (int i = 0; i < p; ++i) d_sorted[i] = d[idx[i]];

    // 构造连接向量 z = [V1(:,k); U2(:,1)] 并按 d 的排序重排
    std::vector<double> z(p);
    cv::Mat V1_last = V1.col(k-1);
    for (int i = 0; i < k; ++i) z[i] = V1_last.at<double>(i);
    cv::Mat U2_first = U2.col(0);
    for (int i = 0; i < n2; ++i) z[k + i] = U2_first.at<double>(i);
    std::vector<double> z_sorted(p);
    for (int i = 0; i < p; ++i) z_sorted[i] = z[idx[i]];

    // 归一化 z（LAPACK 中 Z 在 DLAED2 里除以 sqrt(2)，这里直接归一化）
    double znorm = 0.0;
    for (double v : z_sorted) znorm += v * v;
    znorm = std::sqrt(znorm);
    for (double& v : z_sorted) v /= znorm;
    double rho = off_diag * off_diag;   //  secular 方程中 ρ = σ²

    // ---------------------------
    // 2. 求解各区间的新特征值（新奇异值）
    // ---------------------------
    std::vector<double> lambda(p);
    for (int i = 0; i < p; ++i) {
        double left, right;
        if (i == p - 1) {  // 最右区间
            left = d_sorted[p-1];
            double sum_z2 = 0.0;
            for (double v : z_sorted) sum_z2 += v * v;
            right = left + rho * sum_z2 * 1.5;   // 足够大的上界
        } else {
            left = d_sorted[i];
            right = d_sorted[i+1];
        }
        lambda[i] = solve_secular_root(d_sorted, z_sorted, rho, left, right);
    }

    // ---------------------------
    // 3. 构造 K×K 的特征向量矩阵 Qk
    // ---------------------------
    std::vector<std::vector<double>> Qk(p, std::vector<double>(p, 0.0));
    for (int j = 0; j < p; ++j) {
        double norm2 = 0.0;
        for (int i = 0; i < p; ++i) {
            double val = z_sorted[i] / (d_sorted[i] - lambda[j]);
            Qk[i][j] = val;
            norm2 += val * val;
        }
        double scale = 1.0 / std::sqrt(norm2);
        for (int i = 0; i < p; ++i) Qk[i][j] *= scale;
    }

    // ---------------------------
    // 4. 构建分块对角矩阵 Q_old = diag(V1, U2) 的列顺序（对应排序后的极点）
    // ---------------------------
    cv::Mat Q_old = cv::Mat::zeros(p, p, CV_64F);
    for (int i = 0; i < p; ++i) {
        int orig_col = idx[i];         // 原始极点位置
        if (orig_col < k) {            // 来自 V1
            V1.col(orig_col).copyTo(Q_old.col(i).rowRange(0, k));
        } else {                       // 来自 U2
            int sub = orig_col - k;
            U2.col(sub).rowRange(0, n2).copyTo(Q_old.col(i).rowRange(k, p));
        }
    }

    // ---------------------------
    // 5. 计算最终特征向量 Q_new = Q_old * Qk
    // ---------------------------
    cv::Mat Q_new = cv::Mat::zeros(p, p, CV_64F);
    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < p; ++i) {
            double sum = 0.0;
            for (int t = 0; t < p; ++t) {
                sum += Q_old.at<double>(i, t) * Qk[t][j];
            }
            Q_new.at<double>(i, j) = sum;
        }
    }

    // ---------------------------
    // 6. 将新奇异值排序（降序），并重排特征向量
    // ---------------------------
    std::vector<int> order(p);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return lambda[i] > lambda[j];
    });
    for (int i = 0; i < p; ++i) {
        B.at<double>(i, i) = lambda[order[i]];
        int col = order[i];
        // 右奇异向量（列空间）来自 Q_new 的前 k 行
        for (int r = 0; r < k; ++r) {
            V.at<double>(r, i) = Q_new.at<double>(r, col);
        }
        // 左奇异向量（行空间）来自 Q_new 的后 n2 行
        for (int r = 0; r < n2; ++r) {
            U.at<double>(r, i) = Q_new.at<double>(k + r, col);
        }
    }
}

void divide_conquer_svd(cv::Mat& B, cv::Mat& U, cv::Mat& V) {
    int m = B.rows, n = B.cols;
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