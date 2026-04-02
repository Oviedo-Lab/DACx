
// neuron.cpp
#include "neuron.h"

/*
 * Sections: 
 * - Helper functions
 * - Functions for computing correlations
 * - Probability distribution functions
 * - Dichotomized Gaussian helper functions
 * - Growth-transform helper functions
 * - Matrix and vector operations
 * - Neuron and network classes
 * - Neuron and network member function implementations, adjust settings
 * - Network member function implementations, build network
 * - Neuron member function implementations, basic data handling, loading
 * - Neuron and network member function implementations, analysis and simulation
 */

/*
 * ***********************************************************************************
 * Helper functions
 */

const double Inf = 1e20;

// Build sequence of numbered string prefixes
CharacterVector enum_prefix(std::string prefix, int n) {
    CharacterVector result(n);
    for (int i = 0; i < n; ++i) {
      result[i] = prefix + std::to_string(i + 1);
    }
    return result;
  }

// Rolling mean
VectorXd roll_mean(
    const VectorXd& series,         // 1D vector of points to take rolling mean
    int filter_ws                   // Size of window for taking rolling mean
  ) {
    int n = series.size();
    VectorXd series_out(n);
    for (int i = 0; i < n; i++) {
      int ws = std::min(i + 1, filter_ws);
      // Ensure iterator bounds stay within range
      auto start = series.begin() + (i - ws + 1);
      auto end = series.begin() + (i + 1);
      series_out[i] = std::accumulate(start, end, 0.0) / static_cast<double>(ws);
    } 
    return series_out;
  }
// ... overload
NumericVector roll_mean(
    const NumericVector& series,    
    int filter_ws  
  ) {
    int n = series.size();
    NumericVector series_out(n);
    for (int i = 0; i < n; i++) {
      int ws = std::min(i + 1, filter_ws);
      // Ensure iterator bounds stay within range
      auto start = series.begin() + (i - ws + 1);
      auto end = series.begin() + (i + 1);
      series_out[i] = std::accumulate(start, end, 0.0) / static_cast<double>(ws);
    } 
    return series_out;
  }
// ... overload
std::vector<double> roll_mean(
    const std::vector<double>& series,
    int filter_ws   
  ) {
    int n = series.size();
  std::vector<double> series_out(n);
    for (int i = 0; i < n; i++) {
      int ws = std::min(i + 1, filter_ws);
      // Ensure iterator bounds stay within range
      auto start = series.begin() + (i - ws + 1);
      auto end = series.begin() + (i + 1);
      series_out[i] = std::accumulate(start, end, 0.0) / static_cast<double>(ws);
    } 
    return series_out;
  }

// Return logical vector giving elements of left which match right
LogicalVector eq_left_broadcast(
    const CharacterVector& left,
    const String& right
  ) {
    int n = left.size();
    LogicalVector out(n);
    for (int i = 0; i < n; i++) {
      out[i] = left[i] == right;
    }
    return out;
  }
// ... overload
LogicalVector eq_left_broadcast(
    const std::vector<int>& left,
    const int& right
  ) {
    int n = left.size();
    LogicalVector out(n);
    for (int i = 0; i < n; i++) {
      out[i] = left[i] == right;
    }
    return out;
  }
// ... overload
LogicalVector eq_left_broadcast(
    const VectorXi& left,
    const int& right
  ) {
    int n = left.size();
    LogicalVector out(n);
    for (int i = 0; i < n; i++) {
      out[i] = left[i] == right;
    }
    return out;
  }

// Convert boolean masks to integer indexes
IntegerVector Rwhich(
    const LogicalVector& x
  ) {
    std::vector<int> indices;  // Use std::vector for efficient dynamic resizing
    for (int i = 0; i < x.size(); ++i) {
      if (x[i]) {
        indices.push_back(i);
      }
    }
    if (indices.empty()) {
      Rcpp::stop("No true values found in logical vector for Rwhich function.");
    }
    return wrap(indices);  // Convert std::vector to IntegerVector
  }

// Boolean quantifiers
bool any_true(
    const LogicalVector& x
  ) {
    for (int i = 0; i < x.size(); i++) {
      if (x[i]) {return true;}
    }
    return false;
  }

// Boolean quantifiers
bool all_true(
    const LogicalVector& x
  ) {
    for (int i = 0; i < x.size(); i++) {
      if (!x[i]) {return false;}
    }
    return true;
  }

// Convert to std::vector with doubles 
std::vector<double> to_dVec(
    const VectorXd& vec
  ) {
    std::vector<double> dVec(vec.size());
    for (int i = 0; i < vec.size(); i++) {
      dVec[i] = vec(i);
    }
    return dVec;
  }
// ... overload
std::vector<double> to_dVec(
    const NumericVector& vec
  ) {
    return Rcpp::as<std::vector<double>>(vec);
  }

// Convert to Eigen vector with doubles
VectorXd to_eVec(
    const std::vector<double>& vec
  ) {
    VectorXd VectorXd(vec.size());
    for (int i = 0; i < vec.size(); i++) {
      VectorXd(i) = vec[i];
    }
    return VectorXd;
  }
// ... overload 
VectorXd to_eVec(
    const NumericVector& vec
  ) {
    int n = vec.size();
    VectorXd VectorXd(n);
    for (int i = 0; i < n; i++) {
      VectorXd(i) = vec(i);
    }
    return VectorXd;
  }

// Convert to NumericVector 
NumericVector to_NumVec(
    const VectorXd& vec
  ) {
    NumericVector num_vec(vec.size());
    for (int i = 0; i < vec.size(); i++) {
      num_vec(i) = vec(i);
    }
    return num_vec;
  }
// ... overload 
NumericVector to_NumVec(
    const std::vector<double>& vec
  ) {
    return wrap(vec);
  }

// Convert to Eigen matrix with doubles
MatrixXd to_eMat(
    const NumericMatrix& X
  ) {
    int Xnrow = X.nrow();
    int Xncol = X.ncol();
    MatrixXd M = MatrixXd(Xnrow, Xncol);
    for (int j = 0; j < Xncol; j++) {
      for (int i = 0; i < Xnrow; i++) {
        M(i, j) = X(i, j);
      }
    }
    return M;
  }

// Convert to Eigen matrix with integers
MatrixXi to_eiMat(
    const IntegerMatrix& X
  ) {
    int Xnrow = X.nrow();
    int Xncol = X.ncol();
    MatrixXi M = MatrixXi(Xnrow, Xncol);
    for (int j = 0; j < Xncol; j++) {
      for (int i = 0; i < Xnrow; i++) {
        M(i, j) = X(i, j);
      }
    }
    return M;
  }

// Convert to NumericMatrix
NumericMatrix to_NumMat(
    const MatrixXd& M
  ) {
    int M_nrow = M.rows();
    int M_ncol = M.cols();
    NumericMatrix X(M_nrow, M_ncol);
    for (int j = 0; j < M_ncol; j++) {
      for (int i = 0; i < M_nrow; i++) {
        X(i, j) = M(i, j);
      }
    }
    return X;
  }
// ... overload
NumericMatrix to_NumMat(
    const MatrixXi& M
  ) {
    int M_nrow = M.rows();
    int M_ncol = M.cols();
    NumericMatrix X(M_nrow, M_ncol);
    for (int j = 0; j < M_ncol; j++) {
      for (int i = 0; i < M_nrow; i++) {
        X(i, j) = M(i, j);
      }
    }
    return X;
  }

// Convert to IntegerMatrix
IntegerMatrix to_IntMat(
    const MatrixXi& M
  ) {
    int M_nrow = M.rows();
    int M_ncol = M.cols();
    IntegerMatrix X(M_nrow, M_ncol);
    for (int j = 0; j < M_ncol; j++) {
      for (int i = 0; i < M_nrow; i++) {
        X(i, j) = M(i, j);
      }
    }
    return X;
  }

// Make random walk
NumericVector random_walk(
  const int& n_steps,
  const double& step_size,
  const unsigned int& seed
  ) {
    // Set up the random number generator
    pcg32 rng(seed);
    // Initialize vector to hold walk
    NumericVector walk(n_steps);
    // Start at zero
    walk(0) = 0; 
    // Take steps in walk
    for (int i = 1; i < n_steps; i++) {
      walk(i) = pcg_rnorm(walk(i - 1), step_size, rng);
    }
    // Return the random walk
    return walk;
  }

/*
 * ***********************************************************************************
 * Functions for computing correlations
 */

// Empirical Pearson correlation between two vectors
double empirical_corr(
    const VectorXd& x,
    const VectorXd& y,
    const bool& use_raw
  ) {
   
    // Check dimensions
    int n = x.size();
    if (y.size() != n) {Rcpp::stop("Vectors must be same length for empirical correlation calculation");}
    
    // Calculate components for raw correlation
    double mean_xy = x.dot(y)/(double)n;
    if (use_raw) {
      return mean_xy;
    } else {
      
      // Calculate components for Pearson correlation
      double mean_x = x.sum()/(double)n;
      double mean_y = y.sum()/(double)n;
      double mean_x2 = x.dot(x)/(double)n;
      double mean_y2 = y.dot(y)/(double)n;
      double sd_x = sqrt(mean_x2 - mean_x * mean_x);
      double sd_y = sqrt(mean_y2 - mean_y * mean_y);
      
      // Return correlation estimate
      double corr = (mean_xy - mean_x * mean_y)/(sd_x * sd_y);
      return corr;
      
    }
    
  }

// Empirical Pearson correlation between two variables sampled many times 
double empirical_corr_multisample(
    const MatrixXd& X,            // Rows as intratrial samples, columns as trials
    const MatrixXd& Y,            // Rows as intratrial samples, columns as trials
    const bool& use_raw
  ) {
    
    // Check dimensions
    int n_trials = X.cols();
    if (Y.cols() != n_trials) {Rcpp::stop("Matrices must have same number of columns (trials) for empirical correlation calculation");}
    // Calculate correlation within each trial
    double corr = 0.0;
    int n_trials_nonan = n_trials;
    for (int n = 0; n < n_trials; n++) {
      double corr_n = empirical_corr(X.col(n), Y.col(n), use_raw);
      if (std::isnan(corr_n) || std::isinf(corr_n)) {
        n_trials_nonan -= 1;
      } else {
        corr += corr_n;
      }
    }
    
    // Return mean correlation across trials
    return corr/(double)n_trials_nonan;
    
  }

// Estimate correlation across lags
VectorXd empirical_corr_lagged(
    const MatrixXd& TS_ref,       // Reference time series, rows as time points, columns as trials
    const MatrixXd& TS_cmp,       // Comparison time series (to be lagged), rows as time points, columns as trials
    const int& max_lag,
    const bool& use_raw
  ) {
    
    // Check for matching dimensions 
    const int n_ref = TS_ref.rows();
    const int n_cmp = TS_cmp.rows();
    const int N_trial = TS_ref.cols();
    if (max_lag > n_ref) {Rcpp::stop("max_lag cannot be greater than number of time points (rows) in the reference time series");}
    if (n_cmp < n_ref) {Rcpp::stop("Comparison time series must have at least as many rows as the reference series");}
    if (TS_cmp.cols() != N_trial) {Rcpp::stop("Time series must have same number of trials (columns)");}
    // Check if padding by zeros is needed
    MatrixXd TS_cmp_padded;
    if (n_cmp < n_ref + max_lag) {
      TS_cmp_padded = MatrixXd::Zero(n_ref + max_lag, N_trial);
      TS_cmp_padded(seq(0, n_cmp - 1), Eigen::all) = TS_cmp;
    } else {
      TS_cmp_padded = TS_cmp; 
    }
    
    // Initialize vector to hold correlation values
    VectorXd lagged_corr(max_lag);
    lagged_corr.setZero();
    
    // Compute correlation for all possible lags
    for (int lag = 0; lag < max_lag; lag++) {
      MatrixXd TS_cmp_shifted = TS_cmp_padded(seq(0 + lag, n_ref + lag - 1), Eigen::all);
      lagged_corr(lag) = empirical_corr_multisample(TS_ref, TS_cmp_shifted, use_raw);
    }
    
    // Return correlation 
    return(lagged_corr); 
    
  }

// Estimate raw correlation across lags, raw version (no mean subtraction, no normalization by std)
VectorXd empirical_corr_lagged_raw(
    const MatrixXd& TS1,          // Time series 1, rows as time points, columns as trials
    const MatrixXd& TS2           // Time series 2, rows as time points, columns as trials
  ) {
    
    // Check for matching dimensions 
    const int max_lag = TS1.rows();
    if (TS2.rows() != max_lag) {Rcpp::stop("Time series must have same number of time points (rows)");}
    const int N_trial = TS1.cols();
    if (TS2.cols() != N_trial) {Rcpp::stop("Time series must have same number of trials (columns)");}
    
    // Initialize vector to hold autocorrelation values
    VectorXd lagged_corr(max_lag);
    lagged_corr.setZero();
    
    // Compute correlation for all possible lags
    for (int lag = 0; lag < max_lag; lag++) {
      
      // Shift and cut
      MatrixXd TS2_shifted = TS2(seq(0 + lag, max_lag), Eigen::all);
      MatrixXd TS1_cut = TS1(seq(0, max_lag - lag), Eigen::all);
      
      // Sum over trials
      for (int n = 0; n < N_trial; n++) {
        lagged_corr(lag) += TS2_shifted.col(n).dot(TS1_cut.col(n));
      }
      // The normalization term is the number of overlapping samples at this lag. It's more transparent 
      //   to define it in terms of max_bin instead of max_lag, but in this case max_bin == max_lag.
      double normalization_term = 1.0/((double)max_lag - (double)lag);
      lagged_corr(lag) *= normalization_term;
      // Find mean proportion of co-active bins per trial
      lagged_corr(lag) *= (1.0/N_trial);
      
    }
    
    // Return correlation 
    return(lagged_corr); 
    
  }

// EDF model 
double EDF_autocorr(
    const double& lag,
    const double& A, 
    const double& tau, 
    const double& bias_term,
    const int& return_grad
  ) {
    if (return_grad == 0) {        // No gradient, return function output
      return A * exp(-lag/tau) + bias_term;
    } else if (return_grad == 1) { // gradient wrt A
      return exp(-lag/tau);
    } else if (return_grad == 2) { // gradient wrt tau
      return A * exp(-lag/tau) * (lag/(tau*tau));
    } else {
      return 0.0;
    }
  }

/*
 * ***********************************************************************************
 * Probability distribution functions
 */

// Multivariate normal CDF, upper tail
double mvnorm_cdf_uppertail(
    const NumericVector& threshold, 
    const NumericMatrix& sigma    // covariance matrix of dimension n less than 1000
  ) {
    
    if (sigma.nrow() != sigma.ncol()) {Rcpp::stop("Covariance matrix must be square");}
    if (sigma.nrow() >= 1000) {Rcpp::stop("Covariance matrix must be less than 1000x1000");}
    if (sigma.nrow() != threshold.size()) {Rcpp::stop("Matrix diagonal and threshold vector must be same length");}
    
    Function pmvnorm("pmvnorm", Environment::namespace_env("mvtnorm"));
    // ... uses Genz algorithm
    
    double prob = as<double>(
      pmvnorm( // by default, lower = -Inf, upper = Inf, and mean = 0.
        Named("lower") = threshold, 
        Named("sigma") = sigma, 
        Named("keepAttr") = false
      )
    );
    
    return prob;
    
  }

// Normal CDF, with inverse
double norm_cdf(
    const double& x,
    const double& mu,
    const double& sd,
    const bool& inverse
  ) {
    double xc = x;
    using boost::math::normal; 
    normal standard_normal(mu, sd);
    if (inverse) {
      if (xc < 1e-10) {xc = 1e-10;}
      if (xc > 1 - 1e-10) {xc = 1.0 - 1e-10;}
      return boost::math::quantile(standard_normal, xc);
    } else {
      return boost::math::cdf(standard_normal, xc);
    }
  }

// Multivariate normal random number generator
NumericMatrix mvnorm_random(
    int n,                        // Number of points to generate
    NumericVector mu,             // Mean vector, length determines dimension
    NumericMatrix sigma           // Covariance matrix, square, same dimension as mu
  ) {
   
    if (sigma.nrow() != sigma.ncol()) {Rcpp::stop("Covariance matrix must be square");}
    if (sigma.nrow() != mu.size()) {Rcpp::stop("Mean vector must have same length as sigma diagonal");}
    
    Function mvrnorm("mvrnorm", Environment::namespace_env("MASS"));
    
    // Generate random points
    NumericMatrix X = as<NumericMatrix>(
      mvrnorm(
        Named("n") = n, 
        Named("mu") = mu, 
        Named("Sigma") = sigma,
        Named("tol") = 1e-6 // default is 1e-6
      )
    );
    
    return X;
    
  } 

// Better normal distribution function, with PCG and Box-Muller
double pcg_rnorm(
    double mean, 
    double sd,
    pcg32& rng
  ) {
   
    // Sample from a uniform random distribution between 0 and 1
    int u_max = 1e9; 
    int u1i, u2i;
    do {u1i = rng(u_max);} // randomly select integer between 0 and u_max
    while (u1i == 0);
    double u1 = (double)u1i/(double)u_max; // normalize to (0, 1)
    u2i = rng(u_max);
    double u2 = (double)u2i/(double)u_max; // normalize to (0, 1)
    
    const double two_pi = 2.0 * M_PI;
    
    //compute z0 and z1
    double mag = sd * sqrt(-2.0 * log(u1));
    double z0  = mag * cos(two_pi * u2) + mean;
    //double z1  = mag * sin(two_pi * u2) + mean;
   
    //return std::make_pair(z0, z1);
    return z0; // return only one value, for now
    
  } 

/*
 * ***********************************************************************************
 * Dichotomized Gaussian helper functions
 */

// For estimating sigma for dichotomized Gaussian simulation
NumericVector dg_sigma_formula(
    const double& threshold,      // threshold for dichotomization
    const NumericVector& cov,     // desired covarance after dichotomization
    const NumericMatrix& sigma    // covariance matrix of multivariate Gaussian
  ) {
    // We know threshold and cov. By finding the sigma which sends this function 
    //   to zero, we can find the covariance needed for dichotomized Gaussian simulation
   
    // Check dimension
    int dim = sigma.nrow();
    if (dim != sigma.ncol()) {Rcpp::stop("Covariance matrix must be square");}
    if (dim != cov.size()) {Rcpp::stop("Covariance vector must have the same length as sigma diagonal");}
    
    // Find probability of a point being above the threshold along all dimensions
    NumericVector threshold_vec = Rcpp::rep(threshold, dim); 
    double Phi2_upper = mvnorm_cdf_uppertail(
      threshold_vec, 
      sigma
    );
   
    // Find probability of a point being below the threshold along one dimension
    double Phi = norm_cdf(
      threshold, 
      0.0,     // mean
      1.0,     // sd
      false    // return inverse? No, return cdf
    );
    
    // desired sigma will be the one which sends all elements to zero
    //  Formula: cov = Phi2_upper - (1 - Phi) * (1 - Phi) is derived as follows: 
    //   By definition of cov, cov = E[X1*X2] - E[X1]*E[X2].
    //   In this case, E[X1] = E[X2] = P(X > threshold) = 1 - Phi.
    //   X1*X2 != 0 only if both X1 and X2 > threshold, which occurs with probability Phi2_upper.
    NumericVector residuals(dim);
    for (int i = 0; i < dim; i++) {
      residuals[i] = cov[i] - Phi2_upper + (1 - Phi) * (1 - Phi);
    }
    
    return residuals;
    
  }

// Wrapper for use with find-root-bisection algorithm 
double dg_sigma_formula_scalar(
    const double& threshold,      // threshold for dichotomization
    const double& cov,            // desired covarance after dichotomization
    const double& sigma           // Gaussian covariance
  ) {
    
    // Construct covariance matrix sigma (2x2)
    NumericMatrix sigmaMat(2, 2);
    sigmaMat(_,0) = NumericVector::create(1.0, sigma);
    sigmaMat(_,1) = NumericVector::create(sigma, 1.0);
    
    // Include self-covariance on front, which is the variance, sd^2
    //   The Gaussian is normal, so mean is zero and sd is 1, so variance is 1
    NumericVector cov1 = {1.0, cov};
    
    // Evaluate formula and return second value
    NumericVector residual = dg_sigma_formula(threshold, cov1, sigmaMat);
    // Return only the second element, corresponding to cov
    return residual[1]; 
    
  }

// Function to find sigma by root bisection 
double dg_find_sigma_RootBisection(
    const double& threshold,      // threshold for dichotomization
    const double& cov             // desired covarance after dichotomization
  ) {
    
    // Set search parameters 
    const int max_iter = 50; 
    const double tol = 1e-4;
    
    // Initiate sigmas
    double sigma_lower = -0.999;
    double sigma_upper = 0.999;
    
    // Evaluate formula
    double fx_lower = dg_sigma_formula_scalar(threshold, cov, sigma_lower);
    double fx_upper = dg_sigma_formula_scalar(threshold, cov, sigma_upper);
    
    // Run checks 
    if (abs(fx_lower) < tol) {return sigma_lower;}
    else if (abs(fx_upper) < tol) {return sigma_upper;}
    else if (fx_lower * fx_upper > tol) {return 0.0;} // Both initial covariance values lie on same side of zero crossing
    
    // Run bisection
    double fx = Inf;
    double sigma_mid;
    int iter = 0;
    while (abs(fx) > tol && iter < max_iter) {
      
      // Find midpoint
      sigma_mid = (sigma_lower + sigma_upper)/2.0;
      fx = dg_sigma_formula_scalar(threshold, cov, sigma_mid);
      
      // Update bounds
      if (fx > 0.0) {
        sigma_lower = sigma_mid;
      } else {
        sigma_upper = sigma_mid;
      }
      
      // Update iteration
      iter++;
      
    }
    
    return sigma_mid; 
    
  }

/*
 * ***********************************************************************************
 * Growth-transform helper functions
 */

// Membrane potential barrier function
VectorXd v_barrier(
    const VectorXd& v_input,        // Column vector of membrane potentials for a network of neurons at one time step
    const VectorXd& threshold,      // Spike threshold, in unit_potential, for each neuron in network
    const VectorXd& I_out           // Spike current, in unit_current, for each neuron in network
  ) {
    // Initialize output vector
    VectorXd output(v_input.size());
    // Loop through each neuron in the network
    for (int i = 0; i < v_input.size(); i++) {
      if (v_input[i] < threshold[i]) { 
        // If v_input is below the threshold, return zero
        output[i] = 0.0;
      } else {
        // Otherwise, return output current
        output[i] = I_out[i];
      }
    }
    return output;
  } 

// Create lagged voltage trace matrix to simulate transmission delays
MatrixXd lagged_traces(
    int n,                // Current step index
    const MatrixXi& lag,  // Pairwise lags, in time steps, for signal to get from neuron (row) i to j. 
    const MatrixXd& v     // Membrane potential traces
  ) {
    const int n_neuron = v.rows();
    MatrixXd v_lagged(n_neuron, n_neuron);
    
    for (int j = 0; j < n_neuron; ++j) {
      for (int i = 0; i < n_neuron; ++i) {
        int time_index = n - lag(i, j);
        if (time_index < 0) time_index = 0; 
        v_lagged(i, j) = v(i, time_index); // Neuron i's membrane potential as seen by neuron j. 
      }
    }
    return v_lagged;
    
  }

// Gradient of total dissipated metabolic power in network, w.r.t. membrane potential
VectorXd network_power_dissipation_gradient(
    const MatrixXd& v_traces_lagged,  // n_neuron x n_neuron matrix giving membrane potentials, in unit_potential, with each column j giving the membrane potentials of all neurons as seen by neuron j at this time step
    const VectorXd& v_traces,         // n_neuron x 1 matrix (column vector) of membrane potentials, in unit_potential, from which to calculate derivative
    const VectorXd& stimulus_current, // n_neuron x 1 matrix (column vector) of stimulus currents, in unit_current, from which to calculate derivative
    const MatrixXd& transconductance, // n_neuron x n_neuron transconductance matrix, giving connections between neurons
    const VectorXd& I_spike,          // spike current, in unit_current
    const VectorXd& threshold         // spike threshold, in unit_potential
  ) {  
    // Change dH in total dissipated metabolic power in network (a current) from small change dv in membrane potential, 
    //  given the membrane potential at time step n, for each neuron in network
    //  ... Notice that this function implies that row indices represent post-synaptic neurons, column indices represent pre-synaptic neurons
    VectorXd lagged_power_dissipation = (transconductance.array() * v_traces_lagged.transpose().array()).rowwise().sum();
    // ... transconductance(i, j) = conductance from neuron j to neuron i
    // ... v_traces_lagged(i, j) = neuron i's membrane potential as seen by neuron j at this time step
    // ... v_traces_lagged.transpose()(i, j) = neuron j's membrane potential as seen by neuron i at this time step
    // ... so, row-wise sum gives power dissipation from input into i
    VectorXd dHdv = 
      lagged_power_dissipation -                        // power dissipation (electrical current) from coupling between neurons
      stimulus_current +                                // power injected into the system (electrical current) from external stimulation
      v_barrier(v_traces, threshold, I_spike);          // power dissipated (electrical current) from neural responses (namely, spikes)
    return dHdv;
    
    /*
     * transconductance * v_traces_lagged >>>
     *      (rows are post-synaptic neuron, columns are pre-synaptic neuron) >>>
     *        transconductance row i * v_traces_lagged col j = input into neuron i from all other neurons.
     * ... so, need v_traces_lagged to be a matrix, with each column j giving the membrane potentials of all neurons as seen by neuron j at this time step.
     * ... then the relevant output is the diagonal of the output matrix. 
     *      so, compute only (transconductance.cwiseProduct(v_traces_lagged.transpose())).rowwise().sum()
     * ... How do I make the v_traces_lagged matrix? 
     * ... Need to know, for each neuron i, how many time steps it takes the soma potential of neuron j to reach neuron i (for all j). 
     * ... Time for i to reach j, lag(i, j) = distance(i, j)/conduction_velocity(i), rounded to nearest time step.
     * ... v_traces_lagged(n).col(j)(i) = neuron i's membrane potential at time step n - lag(i, j)
     * ... v_traces_lagged(n).col(j)(i) = v_traces(i, n - lag(i, j));
     */
    
  }

/*
 * ***********************************************************************************
 * Matrix and vector operations
 */

// Create Toeplitz matrix
NumericMatrix toeplitz(
    const std::vector<double>& first_col, 
    const std::vector<double>& first_row
  ) {
    
    int rows = first_col.size();
    int cols = first_row.size();
    NumericMatrix T(rows, cols);
    
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        T(i, j) = (i >= j) ? first_col[i - j] : first_row[j - i];
      }
    }
    
    return T;
    
  }

// Function to make a matrix positive definite
NumericMatrix makePositiveDefinite(
    const NumericMatrix& NumX
  ) {
    
    MatrixXd X = to_eMat(NumX);
    SelfAdjointEigenSolver<MatrixXd> solver(X);
    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXd eigenvectors = solver.eigenvectors();
    
    // Ensure all eigenvalues are positive
    for (int i = 0; i < eigenvalues.size(); ++i) {
      if (eigenvalues(i) < 1e-10) {  // Adjust small or negative values
        eigenvalues(i) = 1e-10;
      }
    }
    
    // Reconstruct the matrix
    return to_NumMat(MatrixXd(eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose()));
    
  }

// Find pairwise Euclidean distances for a set of points
MatrixXd pairwise_distances(
    const MatrixXd& points   // Rows as points, columns as dimensions: columns z (patch), y (layer), x (column)
  ) {
    int N = points.rows();
    MatrixXd D(N, N);
    for (int i = 0; i < N; ++i) {
      const auto vi = points.row(i);
      D(i,i) = 0.0;
      for (int j = i + 1; j < N; ++j) {
        const auto vj = points.row(j);
        const double dz = vi[0] - vj[0];
        const double dy = vi[1] - vj[1];
        const double dx = vi[2] - vj[2];
        const double d = std::sqrt(dz*dz + dy*dy + dx*dx);
        D(i,j) = d;
        D(j,i) = d;
      }
    }
    return D;
  }

// Find pairwise Euclidean distances for a set of points and convert directly into integer lags
MatrixXi pairwise_lags(
    const MatrixXd& coordinates_spatial,      // N x 3 (rows = neurons), columns z (patch), y (layer), x (column)
    const VectorXd& neuron_transmission_velocity,
    double dt
  ) {
    const int N = coordinates_spatial.rows();
    MatrixXi pair_lags(N, N);
    
    // Precompute reciprocals
    const VectorXd inv_vel = neuron_transmission_velocity.cwiseInverse();
    const double inv_dt = 1.0 / dt;
    
    for (int i = 0; i < N; ++i) {
      const auto vi = coordinates_spatial.row(i);
      pair_lags(i, i) = 0;
      
      for (int j = i + 1; j < N; ++j) {
        const auto vj = coordinates_spatial.row(j);
        
        const double dz = vi[0] - vj[0];
        const double dy = vi[1] - vj[1];
        const double dx = vi[2] - vj[2];
        const double dist_ij = std::sqrt(dz*dz + dy*dy + dx*dx);
        
        const double lag_ij = dist_ij * inv_vel[i] * inv_dt;
        const double lag_ji = dist_ij * inv_vel[j] * inv_dt;
        
        pair_lags(i, j) = static_cast<int>(std::round(lag_ij));
        pair_lags(j, i) = static_cast<int>(std::round(lag_ji));
      }
    }
    
    return pair_lags;
  }

/*
 * ***********************************************************************************
 * Neuron and network classes
 */

// Constructor, neuron
neuron::neuron(
    const int id_num, 
    const std::string recording_name, 
    const std::string type, 
    const std::string genotype,
    const std::string sex,
    const std::string hemi,
    const std::string region,
    const std::string age,
    const bool sim, 
    const std::string unit_time, 
    const std::string unit_sample_rate, 
    const std::string unit_data, 
    const double t_per_bin, 
    const double sample_rate
  ) : id_num(id_num), 
    recording_name(recording_name), 
    type(type), 
    genotype(genotype),
    sex(sex),
    hemi(hemi), 
    region(region),
    age(age),
    sim(sim), 
    unit_time(unit_time), 
    unit_sample_rate(unit_sample_rate), 
    unit_data(unit_data), 
    t_per_bin(t_per_bin), 
    sample_rate(sample_rate)
  { 
      // No initialization operations
  }

// Constructor, motif
motif::motif(
    const std::string motif_name
  ) : motif_name(motif_name)
  { 
      // No initialization operations
  }

// Constructor, network
network::network(
    const std::string network_name, 
    const std::string recording_name, 
    const std::string type, 
    const std::string genotype,
    const std::string sex,
    const std::string hemi,
    const std::string region,
    const std::string age,
    const std::string unit_time, 
    const std::string unit_sample_rate, 
    const std::string unit_potential, 
    const std::string unit_current,
    const std::string unit_conductance,
    const std::string unit_distance,
    const double t_per_bin, 
    const double sample_rate
  ) : network_name(network_name), 
    recording_name(recording_name), 
    type(type), 
    genotype(genotype),
    sex(sex),
    hemi(hemi), 
    region(region),
    age(age),
    unit_time(unit_time), 
    unit_sample_rate(unit_sample_rate), 
    unit_potential(unit_potential), 
    unit_current(unit_current),
    unit_conductance(unit_conductance),
    unit_distance(unit_distance),
    t_per_bin(t_per_bin), 
    sample_rate(sample_rate)
  { 
      // No initialization operations
  }

// Lookup table for known cell types
std::unordered_map<std::string, cell_type> cell_types;

/*
 * To use or modify cell types: 
 * 
 *   const auto& ct = cell_types.at("PV");
 *.  double cutoff = ct.temporal_modulation_amplitude;
 *.  cell_types["PV"].temporal_modulation_timeconstant = 0.03;
 */

// Known cell types
// [[Rcpp::export]]
void init_known_celltypes() {
  /*
   * Format: 
   * 
   *  std::string type_name;
   *  int valence;                          // valence of each neuron type, +1 for excitatory, -1 for inhibitory
   *  double temporal_modulation_bias;      // temporal modulation time (in unit_time) bias for each neuron type
   *  double temporal_modulation_timeconstant;     // temporal modulation time (in unit_time) step for each neuron type
   *  double temporal_modulation_amplitude;        // temporal modulation time (in unit_time) cutoff for each neuron type
   *  double transmission_velocity;         // transmission velocity, in unit_distance/unit_time, for each neuron type (microns/ms)
   *  double v_bound;                       // potential bound, in unit_potential (mV), such that -v_bound <= v_traces <= v_bound for all neurons in network
   *  double dHdv_bound;                    // bound on derivative of metabolic energy wrt potential, such that dHdv_bound > abs(dHdv), in unit_current, for each neuron in the network, based on its type (mA)
   *  double I_spike;                       // spike current, in unit_current (mA)
   *  double coupling_scaling_factor;       // Controls how energy used in synaptic transmission compares to that used in spiking
   *  double spike_potential;               // Magnitude of each spike, in unit_potential (mV)
   *  double resting_potential;             // resting potential, in unit_potential (mV)
   *  double threshold;                     // spike threshold, in unit_potential (mV)
   */
  // Defaults 
  double temporal_modulation_bias = 1e-3;   // primarily affects firing rate
  double temporal_modulation_timeconstant = 1e0;
  double temporal_modulation_amplitude = 5e-3;
  double transmission_velocity = 30e3;      // microns/ms ... 30 m/s = 30e6 micron/s = 30e6 micron/ 1e3 ms = 30e3 micron/ms
  double v_bound = 85.0;                  
  double dHdv_bound = 1.05e-6;
  double I_spike = 1e-6; 
  double coupling_scaling_factor = 1e-7;
  double spike_potential = 35.0;
  double resting_potential = -70.0; 
  double threshold = -55.0;  
  int process_node_count = 10;
  int axon_branch_count = 10;
  int dendrite_branch_count = 10;
  // Define excitatory cells
  cell_types["principal"] = cell_type{
    "principal", 1,
    temporal_modulation_bias, temporal_modulation_timeconstant,
    temporal_modulation_amplitude * 0.0, // No bursting
    transmission_velocity, 
    v_bound, dHdv_bound, I_spike,
    coupling_scaling_factor,
    spike_potential, resting_potential, threshold,
    process_node_count, axon_branch_count, dendrite_branch_count
  };
  // Define inhibitory cells
  cell_types["PV"] = cell_type{
    "PV", -1,
    temporal_modulation_bias, temporal_modulation_timeconstant,
    temporal_modulation_amplitude,
    transmission_velocity * 1.2,
    v_bound, dHdv_bound, I_spike,
    coupling_scaling_factor,
    spike_potential, resting_potential, threshold,
    process_node_count, axon_branch_count, dendrite_branch_count
  };
  cell_types["SST"] = cell_type{
    "SST", -1,
    temporal_modulation_bias, temporal_modulation_timeconstant,
    temporal_modulation_amplitude,
    transmission_velocity * 0.8,
    v_bound, dHdv_bound, I_spike,
    coupling_scaling_factor,
    spike_potential, resting_potential, threshold,
    process_node_count, axon_branch_count, dendrite_branch_count
  };
  cell_types["VIP"] = cell_type{
    "VIP", -1,
    temporal_modulation_bias, temporal_modulation_timeconstant,
    temporal_modulation_amplitude,
    transmission_velocity,
    v_bound, dHdv_bound, I_spike,
    coupling_scaling_factor,
    spike_potential, resting_potential, threshold,
    process_node_count, axon_branch_count, dendrite_branch_count
  };
}

// Print known cell types 
// [[Rcpp::export]]
void print_known_celltypes() {
  Rcpp::Rcout << "Known cell types:" << std::endl;
  for (const auto& pair : cell_types) {
    const cell_type& ct = pair.second;
    Rcpp::Rcout << "\nType: " << ct.type_name << std::endl
                << "  Valence: " << ct.valence << std::endl
                << "  Temporal modulation bias: " << ct.temporal_modulation_bias << std::endl
                << "  Temporal modulation time constant: " << ct.temporal_modulation_timeconstant << std::endl
                << "  Temporal modulation amplitude: " << ct.temporal_modulation_amplitude << std::endl
                << "  Transmission velocity: " << ct.transmission_velocity << std::endl
                << "  Potential bound (mV): " << ct.v_bound << std::endl
                << "  Metabolic energy derivative dHdv bound (mA): " << ct.dHdv_bound << std::endl
                << "  Spike current (mA): " << ct.I_spike << std::endl
                << "  Coupling scaling factor: " << ct.coupling_scaling_factor << std::endl
                << "  Spike potential (mV): " << ct.spike_potential << std::endl
                << "  Resting potential (mV): " << ct.resting_potential << std::endl
                << "  Threshold (mV): " << ct.threshold << std::endl
                << "  Process node density: " << ct.process_node_count << std::endl
                << "  Axon branch density: " << ct.axon_branch_count << std::endl
                << "  Dendrite branch density: " << ct.dendrite_branch_count << std::endl;
  }
}

// Fetch cell type parameters 
// [[Rcpp::export]]
List fetch_cell_type_params(const std::string& type_name) {
  auto it = cell_types.find(type_name);
  if (it == cell_types.end()) {
    Rcpp::stop("Cell type not found in known cell types");
  } else {
    const cell_type& ct = (*it).second;
    return List::create(
      Named("type_name") = ct.type_name,
      Named("valence") = ct.valence,
      Named("temporal_modulation_bias") = ct.temporal_modulation_bias,
      Named("temporal_modulation_timeconstant") = ct.temporal_modulation_timeconstant,
      Named("temporal_modulation_amplitude") = ct.temporal_modulation_amplitude,
      Named("transmission_velocity") = ct.transmission_velocity,
      Named("v_bound") = ct.v_bound,
      Named("dHdv_bound") = ct.dHdv_bound,
      Named("I_spike") = ct.I_spike,
      Named("coupling_scaling_factor") = ct.coupling_scaling_factor,
      Named("spike_potential") = ct.spike_potential,
      Named("resting_potential") = ct.resting_potential,
      Named("threshold") = ct.threshold,
      Named("process_node_count") = ct.process_node_count,
      Named("axon_branch_count") = ct.axon_branch_count,
      Named("dendrite_branch_count") = ct.dendrite_branch_count
    );
  }
}

// Make new cell type
// [[Rcpp::export]]
void add_cell_type(
    const std::string& type_name,
    const int& valence,
    const double& temporal_modulation_bias,
    const double& temporal_modulation_timeconstant,
    const double& temporal_modulation_amplitude,
    const double& transmission_velocity,
    const double& v_bound,                      // potential bound, in unit_potential
    const double& dHdv_bound,                   // bound on dHdv, in unit_current
    const double& I_spike,                      // spike current, in unit_current
    const double& coupling_scaling_factor,      // Controls how energy used in synaptic transmission
    const double& spike_potential,              // Magnitude of each spike, in unit_potential
    const double& resting_potential,            // resting potential, in unit_potential
    const double& threshold,                    // spike threshold, in unit_potential
    const int& process_node_count,              // Expected number of process nodes over the length of one process branch
    const int& axon_branch_count,               // Expected number of axon branches 
    const int& dendrite_branch_count            // Expected number of dendrite branches 
  ) {
    if (cell_types.find(type_name) != cell_types.end()) {
      Rcpp::stop("Cell type already exists in known cell types");
    } else {
      cell_types[type_name] = cell_type{
        type_name, valence,
        temporal_modulation_bias, temporal_modulation_timeconstant,
        temporal_modulation_amplitude,
        transmission_velocity,
        v_bound, dHdv_bound, I_spike,
        coupling_scaling_factor,
        spike_potential, resting_potential, threshold,
        process_node_count, axon_branch_count, dendrite_branch_count
      };
    }
  }

// Modify cell type parameters 
// [[Rcpp::export]]
void modify_cell_type(
    const std::string& type_name,
    const int& valence,
    const double& temporal_modulation_bias,
    const double& temporal_modulation_timeconstant,
    const double& temporal_modulation_amplitude,
    const double& transmission_velocity,
    const double& v_bound,                      // potential bound, in unit_potential
    const double& dHdv_bound,                   // bound on dHdv, in unit_current
    const double& I_spike,                      // spike current, in unit_current
    const double& coupling_scaling_factor,      // Controls how energy used in synaptic transmission compares to that used in spiking
    const double& spike_potential,              // Magnitude of each spike, in unit_potential
    const double& resting_potential,            // resting potential, in unit_potential
    const double& threshold,                    // spike threshold, in unit_potential
    const int& process_node_count,              // Expected number of process nodes over the length of one process branch
    const int& axon_branch_count,               // Expected number of axon branches 
    const int& dendrite_branch_count            // Expected number of dendrite branches 
  ) {
    if (cell_types.find(type_name) != cell_types.end()) {
      cell_types[type_name].valence = valence;
      cell_types[type_name].temporal_modulation_bias = temporal_modulation_bias;
      cell_types[type_name].temporal_modulation_timeconstant = temporal_modulation_timeconstant;
      cell_types[type_name].temporal_modulation_amplitude = temporal_modulation_amplitude;
      cell_types[type_name].transmission_velocity = transmission_velocity;
      cell_types[type_name].v_bound = v_bound;
      cell_types[type_name].dHdv_bound = dHdv_bound;
      cell_types[type_name].I_spike = I_spike;
      cell_types[type_name].coupling_scaling_factor = coupling_scaling_factor;
      cell_types[type_name].spike_potential = spike_potential;
      cell_types[type_name].resting_potential = resting_potential;
      cell_types[type_name].threshold = threshold;
      cell_types[type_name].process_node_count = process_node_count;
      cell_types[type_name].axon_branch_count = axon_branch_count;
      cell_types[type_name].dendrite_branch_count = dendrite_branch_count;
    } else {
      Rcpp::stop("Cell type not found in known cell types");
    }
  }

/*
 * ***********************************************************************************
 * Neuron and network member function implementations, adjust settings
 */

void neuron::set_edf_initials(
    double a0, 
    double t0
  ) {
    A0 = a0;
    tau0 = t0;
  };

void neuron::set_edf_termination(
    double ct, 
    int me
  ) {
    ctol = ct;
    max_evals = me;
  };

void network::set_network_structure(
    CharacterVector nrn_types,
    CharacterVector lyr_names,
    int n_lyr,
    int n_cls,
    int n_pch,
    double lyr_height,
    double cls_diameter,
    double lyr_separation_factor,
    double cls_separation_factor,
    double pch_separation_factor,
    IntegerMatrix nrn_per_node,
    List recur_factors,
    double pruning_thresh_factor
  ) {
    
    // Check layer names (needed for motifs)
    if (lyr_names.size() != n_lyr) {
      Rcpp::Rcout << "lyr_names size: " << lyr_names.size() << ", n_layers: " << n_lyr << std::endl;
      Rcpp::stop("Length of lyr_names must equal n_layers");
    }
    
    // Convert recurrence factors from R List to std::vector<MatrixXd>
    std::vector<MatrixXd> rec_factors_vec;
    for (int i = 0; i < recur_factors.size(); i++) {
      NumericMatrix rec_mat_r = recur_factors[i];
      recurrence_factors.push_back(to_eMat(rec_mat_r));
    }
    
    // Load cell types 
    for (String nt : nrn_types) {
      std::string nts = nt;
      auto it = cell_types.find(nts);
      if (it == cell_types.end()) Rcpp::stop("Unknown neuron type: %s", nts);
      neuron_types.push_back((*it).second);
    }
   
    // Set other network parameters
    layer_names = lyr_names;
    n_layers = n_lyr;
    n_columns = n_cls;
    n_patches = n_pch;
    layer_height = lyr_height;
    column_diameter = cls_diameter;
    layer_separation_factor = lyr_separation_factor;
    column_separation_factor = cls_separation_factor;
    patch_separation_factor = pch_separation_factor;
    neurons_per_node = to_eiMat(nrn_per_node);
    pruning_threshold_factor = pruning_thresh_factor;
    
    // Set network components
    n_neuron_types = neuron_types.size();
    int n_nodes = n_layers * n_columns * n_patches;
    n_neurons = 0; // Compute total number of neurons as we go
    node_range_ends.assign(n_nodes, 0);
    node_coordinates_spatial.resize(n_nodes, 3);
    std::vector<double> neuron_temporal_modulation_bias;
    std::vector<double> neuron_temporal_modulation_timeconstant;
    std::vector<double> neuron_temporal_modulation_amplitude;
    std::vector<double> neuron_transmission_velocity_tmp;
    for (int p = 0; p < n_patches; p++) {
      for (int l = 0; l < n_layers; l++) {
        for (int c = 0; c < n_columns; c++) {
          int node_idx = p * (n_layers * n_columns) + l * n_columns + c;
          // Set global spatial coordinates for this node
          node_coordinates_spatial(node_idx, 0) = p * column_diameter/2.0 * patch_separation_factor;   // z
          node_coordinates_spatial(node_idx, 1) = l * layer_height/2.0 * layer_separation_factor;      // y
          node_coordinates_spatial(node_idx, 2) = c * column_diameter/2.0 * column_separation_factor;  // x
          // ... was c = 0, l = 1, p = 2
          for (int t = 0; t < n_neuron_types; t++) {
            // Randomly select neuron numbers for each node
            int n = (int)R::rpois(neurons_per_node(l,t));
            // Keep track of the number of cells assigned so far
            n_neurons += n; 
            // Keep track of the types of these cells and their intrinsic properties
            for (int i = 0; i < n; i++) {
              neuron_type_name.push_back(neuron_types[t].type_name);
              neuron_type_num.push_back(t);
              neuron_temporal_modulation_bias.push_back(neuron_types[t].temporal_modulation_bias);
              neuron_temporal_modulation_timeconstant.push_back(neuron_types[t].temporal_modulation_timeconstant);
              neuron_temporal_modulation_amplitude.push_back(neuron_types[t].temporal_modulation_amplitude);
              neuron_transmission_velocity_tmp.push_back(neuron_types[t].transmission_velocity);
            }
          }
          // Save end-point index for this node
          node_range_ends[node_idx] = n_neurons - 1;
        }
      }
    }
    
    // Grab cell type parameters and convert into vectors of length n_neurons
    v_bound = VectorXd::Zero(n_neurons);
    dHdv_bound = VectorXd::Zero(n_neurons);
    I_spike = VectorXd::Zero(n_neurons);
    spike_potential = VectorXd::Zero(n_neurons);
    resting_potential = VectorXd::Zero(n_neurons);
    threshold = VectorXd::Zero(n_neurons);
    for (int i = 0; i < n_neurons; i++) {
      int type_idx = neuron_type_num[i];
      v_bound(i) = neuron_types[type_idx].v_bound;
      dHdv_bound(i) = neuron_types[type_idx].dHdv_bound;
      I_spike(i) = neuron_types[type_idx].I_spike;
      spike_potential(i) = neuron_types[type_idx].spike_potential;
      resting_potential(i) = neuron_types[type_idx].resting_potential;
      threshold(i) = neuron_types[type_idx].threshold;
    }
    
    // Set length of the vectors holding cell processes
    arbors.resize(n_neurons);
    
    // Convert neuron temporal modulation to Eigen matrix
    neuron_temporal_modulation = MatrixXd::Zero(n_neurons, 3);
    neuron_temporal_modulation.col(0) = Map<VectorXd>(neuron_temporal_modulation_bias.data(), neuron_temporal_modulation_bias.size());
    neuron_temporal_modulation.col(1) = Map<VectorXd>(neuron_temporal_modulation_timeconstant.data(), neuron_temporal_modulation_timeconstant.size());
    neuron_temporal_modulation.col(2) = Map<VectorXd>(neuron_temporal_modulation_amplitude.data(), neuron_temporal_modulation_amplitude.size());
    
    // Convert neuron transmission delay to Eigen vector
    neuron_transmission_velocity = Map<VectorXd>(neuron_transmission_velocity_tmp.data(), neuron_transmission_velocity_tmp.size());
    
    // Resize network coordinate components 
    coordinates_spatial = MatrixXd::Zero(n_neurons, 3); 
    coordinates_node = MatrixXi::Zero(n_neurons, 3); // patch (z), layer (y), column (x)
    
  };

/*
 * ***********************************************************************************
 * Network member function implementations, build network
 */

void motif::load_projection(
    const Projection& proj,
    const int& max_up,
    const int& max_down,
    const double& c_strength
  ) {
    projections.push_back(proj);
    max_col_shift_up.push_back(max_up);
    max_col_shift_down.push_back(max_down);
    connection_strength.push_back(c_strength);
    n_projections++;
  }

// Function to make axons and dendrites 
void network::make_arbor_branch(
    const int& cell_idx,                    // Number of neuron for which to make processes
    int n_segments,                         // Expected number of process segments on longest branch
    const bool& is_axon,                    // Whether to make axon (true) or dendrite (false)
    double segment_divisor,                 // Specifies expected length of segments in terms of column diameter and layer height, or distance to attractor point
    int parent_branch_idx,                  // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
    const Eigen::Matrix<double, 3, 1> attractor_point
  ) {
    
    // Check expected number of segments
    if (n_segments < 2) {n_segments = 2;}
    // ... and check segment divisor
    if (segment_divisor <= 0.0) {segment_divisor = n_segments;}
    // ... randomly select the number of segments, ensuring at least 1
    n_segments = R::rpois(n_segments - 1) + 1;
    
    // Check attractor point
    bool use_attractor = false;
    if (attractor_point(0,0) != 0.0 ||
        attractor_point(1,0) != 0.0 ||
        attractor_point(2,0) != 0.0) {
      use_attractor = true;
    }
    
    // Initialize new cell_arbors structure
    cell_arbors& arbor = arbors[cell_idx];
    
    // Set parent flag 
    bool has_parent = (parent_branch_idx >= 0);
    
    // Find initial point 
    Vector3d initial_point;
    if (has_parent) {
      // If child of parent branch, make sure parent exists
      if (parent_branch_idx >= arbor.axon.size()) {
        Rcpp::Rcout << "Parent branch index: " << parent_branch_idx << ", number of branches in arbor: " << arbor.axon.size() << std::endl;
        Rcpp::stop("Parent branch index exceeds number of branches in arbor");
      }
      // ... and check axon flag
      if (is_axon != arbor.axon[parent_branch_idx]) {
        Rcpp::stop("Parent branch type (axon vs dendrite) does not match specified branch type for new branch");
      }
      // ... and randomly select branch point 
      int parent_branch_length = arbor.coordinates[parent_branch_idx].size();
      int branch_point = Rcpp::sample(parent_branch_length, 1)[0] - 1; // Rcpp::sample is 1-indexed, so subtract 1 for 0-indexing
      // ... and set as initial point
      initial_point = arbor.coordinates[parent_branch_idx][branch_point];
      // ... and save as parent of next node
      arbor.parents[parent_branch_idx].push_back(branch_point);
    } else {
      // Set axon flag for new process arbor
      if (is_axon) {
        arbor.axon.push_back(true);
      } else {
        arbor.axon.push_back(false);
      }
      // Set initial point as the soma location
      initial_point = coordinates_spatial.row(cell_idx);
      // Initialize new coordinates vector (of Vector3d) with first row as spatial coordinates of the cell body 
      arbor.coordinates.push_back({initial_point});
      // ... and initialize new vector to track node parents
      arbor.parents.push_back({-1});
      // ... initialize empty leaf vector 
      arbor.leafs.emplace_back();
      // ... set as parent branch 
      parent_branch_idx = arbor.axon.size() - 1;
    }
    
    // Set the expected segment length
    double zx_segment_length = (column_diameter/2.0)/segment_divisor; 
    double y_segment_length = (layer_height/2.0)/segment_divisor;
    // ... adjust if using attractor 
    if (use_attractor) {
      double z_bias = (attractor_point(0,0) - arbor.coordinates[parent_branch_idx].back()[0]);
      double y_bias = (attractor_point(1,0) - arbor.coordinates[parent_branch_idx].back()[1]);
      double x_bias = (attractor_point(2,0) - arbor.coordinates[parent_branch_idx].back()[2]);
      double bias_magnitude = std::sqrt(z_bias*z_bias + y_bias*y_bias + x_bias*x_bias);
      bias_magnitude = (bias_magnitude > 0) ? bias_magnitude : 1.0; // Avoid division by zero
      zx_segment_length *= bias_magnitude / column_diameter;
      y_segment_length *= bias_magnitude / layer_height;
    }
    
    // Make branch
    Vector3d last_node = initial_point;
    for (int s = 0; s < n_segments; s++) {
      
      // Make random component of the step
      double z_step = R::rnorm(0.0, zx_segment_length);
      double y_step = R::rnorm(0.0, y_segment_length);
      double x_step = R::rnorm(0.0, zx_segment_length);
      double step_magnitude = std::sqrt(z_step*z_step + y_step*y_step + x_step*x_step);
      
      if (use_attractor) {
        
        // Make directed component of the step 
        double z_bias = (attractor_point(0,0) - last_node[0]);
        double y_bias = (attractor_point(1,0) - last_node[1]);
        double x_bias = (attractor_point(2,0) - last_node[2]);
        
        // ... normalize so that it's the same magnitude as the random component
        double bias_magnitude = std::sqrt(z_bias*z_bias + y_bias*y_bias + x_bias*x_bias);
        bias_magnitude = (bias_magnitude > 0) ? bias_magnitude : 1.0; // Avoid division by zero
        z_bias = (z_bias / bias_magnitude) * step_magnitude;
        y_bias = (y_bias / bias_magnitude) * step_magnitude;
        x_bias = (x_bias / bias_magnitude) * step_magnitude;
        
        // Randomly select a weight between 0 and 1
        double weight = R::runif(0.0, 1.0);
        
        // Make weighted combination of the step and bias 
        z_step = weight * z_step + (1 - weight) * z_bias;
        y_step = weight * y_step + (1 - weight) * y_bias;
        x_step = weight * x_step + (1 - weight) * x_bias;
        
      }
      
      // Add the step to the previous segment's coordinates to get the new segment's coordinates, and add to arbor coordinates
      Vector3d new_node(
        last_node[0] + z_step,
        last_node[1] + y_step,
        last_node[2] + x_step
      );
      arbor.coordinates[parent_branch_idx].push_back(new_node);
      // ... and update last_node 
      last_node = new_node;
      
      // Add new node as child of previous node in parents vector
      if (!has_parent || s > 0) {
        arbor.parents[parent_branch_idx].push_back(arbor.coordinates[parent_branch_idx].size() - 2);
      }
      
    }
    
    // Make the last row number in coordinates as a leaf for this branch
    arbor.leafs[parent_branch_idx].push_back(arbor.coordinates[parent_branch_idx].size() - 1);
    
  }

void network::make_arbor(
    const int& cell_idx,                    // Number of neuron for which to make processes
    int n_segments,                         // Expected number of process segments on longest branch
    int n_branches,                         // Expected number of branches, including the main process 
    const bool& is_axon,                    // Whether to make axon (true) or dendrite (false)
    double segment_divisor,                 // Specifies expected length of segments in terms of column diameter and layer height, or distance to attractor point
    int parent_branch_idx,                  // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
    const Eigen::Matrix<double, 3, 1> attractor_point
  ) {
    
    // Find number of existing branches 
    int n_existing_arbors = arbors[cell_idx].axon.size();
    
    // Randomly set number of branches 
    if (n_branches < 2) {n_branches = 2;}
    n_branches = R::rpois(n_branches - 1) + 1; // Ensure at least 1 branch
    
    // Make branch structure
    std::vector<int> parent_branch_idx_list; 
    if (parent_branch_idx == -1) {
      // Starting fresh, no matter what other branches already exist
      parent_branch_idx_list.push_back(-1);
      for (int b = 1; b < n_branches; b++) {
        parent_branch_idx_list.push_back(n_existing_arbors); 
      }
    } else {
      // Starting from existing branch, so make sure to build off of that branch and its children
      if (parent_branch_idx >= n_existing_arbors) {
        Rcpp::Rcout << "Parent branch index: " << parent_branch_idx << ", number of existing branches: " << n_existing_arbors << std::endl;
        Rcpp::stop("Parent branch index exceeds number of branches in arbor");
      }
      parent_branch_idx_list.push_back(parent_branch_idx);
      for (int b = 1; b < n_branches; b++) {
        parent_branch_idx_list.push_back(n_existing_arbors);
      }
    }
    
    for (int b = 0; b < n_branches; ++b) {
      // Grab parent branch index for this branch
      int parent_branch_idx_b = parent_branch_idx_list[b];
      // Make arbor branch
      make_arbor_branch(
        cell_idx,
        n_segments,
        is_axon,
        segment_divisor,
        parent_branch_idx_b,
        attractor_point
      );
    }
    
  }

// Function to set transconductances and spatial coordinates for all local nodes 
void network::make_local_nodes() {
    
    if (edge_types.size() != 0) {
      Rcpp::Rcout << "Edge types have already been set; cannot run make_local_nodes twice; returning." << std::endl;
      return;
    }
    
    // Initialize vectors to track local edge coordinates
    std::vector<int> local_edges_pre; 
    std::vector<int> local_edges_post;
    
    // Initialize local transconductance matrix
    MatrixXd local_transconductances = MatrixXd::Zero(n_neurons, n_neurons);
    
    // Patch index of the local node 
    for (int p = 0; p < n_patches; p++) {
      
      // Layer index of the local node
      for (int l = 0; l < n_layers; l++) {
        
        // Get recurrence factor matrix for this layer
        MatrixXd recurrence_factor_matrix = recurrence_factors[l];
        
        // Column index of the local node
        for (int c = 0; c < n_columns; c++) {
          
          // Get node ID number
          int node_idx = p * (n_layers * n_columns) + l * n_columns + c;
          // Get spatial position of this node
          double node_z = node_coordinates_spatial(node_idx, 0); // z / p
          double node_y = node_coordinates_spatial(node_idx, 1); // y / l
          double node_x = node_coordinates_spatial(node_idx, 2); // x / c
          // Get the range of neuron ID numbers for this node
          int node_range_start = (node_idx == 0) ? 0 : node_range_ends[node_idx - 1] + 1;
          int node_range_end = node_range_ends[node_idx];
          
          // For all cells in this node
          for (int idx = node_range_start; idx <= node_range_end; idx++) {
            
            // Set spatial coordinates
            coordinates_spatial(idx, 0) = node_z + R::rnorm(0.0, column_diameter/2.0);
            coordinates_spatial(idx, 1) = node_y + R::rnorm(0.0, layer_height/2.0);
            coordinates_spatial(idx, 2) = node_x + R::rnorm(0.0, column_diameter/2.0);
            
            // Set node coordinates
            coordinates_node(idx, 0) = p;
            coordinates_node(idx, 1) = l;
            coordinates_node(idx, 2) = c;
            
            // Get neuron types 
            int t_num = neuron_type_num[idx];
            int process_node_count = neuron_types[t_num].process_node_count;
            int axon_branch_count = neuron_types[t_num].axon_branch_count;
            int dendrite_branch_count = neuron_types[t_num].dendrite_branch_count;
            
            // Create local axon arbor
            make_arbor(idx, process_node_count, axon_branch_count, true);
            
            // Create local dendrite arbor
            make_arbor(idx, process_node_count, dendrite_branch_count, false);
            
          }
          
          // For all combinations of pre- and post-synaptic neurons in this node
          for (int idx_pre = node_range_start; idx_pre <= node_range_end; idx_pre++) {
            
            // Get neuron types for pre-synaptic neurons
            int t_pre = neuron_type_num[idx_pre];
            
            // Get spike current potential, and power for pre-synaptic cells
            double I_spike = neuron_types[t_pre].I_spike;
            double spike_potential = neuron_types[t_pre].spike_potential;
            double spike_H = I_spike * spike_potential;
            
            // Get coupling scaling factor for pre-synaptic cells
            double coupling_scaling_factor = neuron_types[t_pre].coupling_scaling_factor;
            
            // Set transconductance into post-synaptic cells
            for (int idx_post = node_range_start; idx_post <= node_range_end; idx_post++) {
              
              // Get neuron types for post-synaptic neurons
              int t_post = neuron_type_num[idx_post];
              
              // Get neuron valences for pre-synaptic neurons
              double val_pre = neuron_types[t_pre].valence;
              
              // Get recurrence factor for this connection type
              double rec_factor = recurrence_factor_matrix(t_post, t_pre);
              double transductance_bias = rec_factor * spike_H * coupling_scaling_factor;
              double pruning_threshold = pruning_threshold_factor * transductance_bias;
              
              // Set transductance 
              double trans = R::runif(0.0, 2.0) * transductance_bias;
              if (trans > pruning_threshold) {
                local_transconductances(idx_post, idx_pre) = val_pre * trans;
                // Save edge coordinate
                local_edges_pre.push_back(idx_pre);
                local_edges_post.push_back(idx_post);
              }
              
            }
            
          }
          
        }
      }
      
    }
   
    // Save to transconductance matrix
    transconductances.push_back(local_transconductances);
    
    // Collect local edge coordinates in matrix
    int n_local_edges = local_edges_pre.size();
    MatrixXi local_edges(n_local_edges, 2); 
    local_edges.col(0) = Eigen::Map<VectorXi>(local_edges_pre.data(), n_local_edges);
    local_edges.col(1) = Eigen::Map<VectorXi>(local_edges_post.data(), n_local_edges);
    
    // Save to edge types
    edge_types.push_back(local_edges);
    
  }

// Function to apply circuit motif
void network::apply_circuit_motif(
    const motif& cmot
  ) {
   
    if (edge_types.size() < 1) {
      Rcpp::stop("Must set local edges before applying any circuit motifs.");
    }
    
    // Initialize vectors to track motif edge coordinates
    std::vector<int> motif_edges_pre; 
    std::vector<int> motif_edges_post;
    
    // Initialize motif transconductance matrix
    MatrixXd motif_transconductances = MatrixXd::Zero(n_neurons, n_neurons);
    
    // For each projection in the motif
    for (int p = 0; p < cmot.n_projections; p++) {
      
      // Grab projection
      Projection proj = cmot.projections[p];
      
      // Grab pre- and post-synaptic cell types for this projection
      std::string pre_type_name = proj.pre_type;
      std::string post_type_name = proj.post_type;
      cell_type pre_type = cell_types.at(pre_type_name);
      cell_type post_type = cell_types.at(post_type_name);
      // Get indices for neuron_types in this network
      CharacterVector type_names(neuron_types.size());
      for (int i = 0; i < neuron_types.size(); i++) {type_names[i] = neuron_types[i].type_name;}
      LogicalVector pre_type_exists = eq_left_broadcast(type_names, pre_type_name);
      LogicalVector post_type_exists = eq_left_broadcast(type_names, post_type_name);
      if (!(any_true(pre_type_exists) && any_true(post_type_exists))) {
        Rcpp::Rcout << "Projection " << p << " in motif " << cmot.motif_name << " has pre- or post-synaptic type that does not exist in this network; skipping this projection." << std::endl;
        continue;
      }
      int t_pre = Rwhich(pre_type_exists)[0];
      int t_post = Rwhich(post_type_exists)[0];
      // ... and make masks for neurons in this network
      LogicalVector pre_type_mask = eq_left_broadcast(neuron_type_num, t_pre);
      LogicalVector post_type_mask = eq_left_broadcast(neuron_type_num, t_post);
      
      // Grab pre-synaptic projection strength and set pruning threshold
      // ... get spike current potential, and power for pre-synaptic cell
      double I_spike = neuron_types[t_pre].I_spike;
      double spike_potential = neuron_types[t_pre].spike_potential;
      double spike_H = I_spike * spike_potential;
      // ... get coupling scaling factor for pre-synaptic cell
      double coupling_scaling_factor = neuron_types[t_pre].coupling_scaling_factor;
      // ... get recurrence factor for this connection type
      double proj_strength = cmot.connection_strength[p];
      double transductance_bias = proj_strength * spike_H * coupling_scaling_factor;
      double pruning_threshold = pruning_threshold_factor * transductance_bias;
      
      // Grab pre-synaptic valence
      int val_pre = neuron_types[t_pre].valence;
      
      // Grab pre and post layers
      LogicalVector pre_layer_exists = eq_left_broadcast(layer_names, proj.pre_layer);
      LogicalVector post_layer_exists = eq_left_broadcast(layer_names, proj.post_layer);
      if (!(any_true(pre_layer_exists) && any_true(post_layer_exists))) {
        Rcpp::Rcout << "Projection " << p << " in motif " << cmot.motif_name << " has pre- or post-synaptic layer that does not exist in this network; skipping this projection." << std::endl;
        continue;
      }
      int layer_pre = Rwhich(pre_layer_exists)[0];
      int layer_post = Rwhich(post_layer_exists)[0];
      // ... and make masks 
      LogicalVector pre_layer_mask = eq_left_broadcast(coordinates_node(Eigen::all,1), layer_pre); // column 1 is the layer
      LogicalVector post_layer_mask = eq_left_broadcast(coordinates_node(Eigen::all,1), layer_post);
      
      // Grab pre and post densities
      double density_pre = proj.pre_density;
      double density_post = proj.post_density;
      
      // Grab max shifts
      int max_up = cmot.max_col_shift_up[p];
      int max_down = cmot.max_col_shift_down[p];
      
      // Build column range
      VectorXi col_range(max_up + max_down + 1);
      for (int i = 0; i < col_range.size(); i++) col_range[i] = -max_down + i;
      
      // Pre-make all column masks 
      LogicalMatrix column_masks(n_neurons, n_columns);
      for (int c = 0; c < n_columns; c++) {
        column_masks(_, c) = eq_left_broadcast(coordinates_node(Eigen::all,2), c); // column 2 is the column
      }
      
      // Apply projection to each column 
      for (int c = 0; c < n_columns; c++) {
        
        // Get pre-synaptic column mask
        LogicalVector pre_column_mask = column_masks(_, c);
        
        // Shift range to this column 
        VectorXi col_range_shifted = col_range.array() + c;
        
        // For each target column
        for (int tc : col_range_shifted) {
          
          // Check if target column is valid
          if (tc >= 0 && tc < n_columns) {
            
            // Don't make local connections 
            if (layer_pre != layer_post || c != tc) {
              
              // Find distance off home column 
              int col_offset = std::abs(tc - c);
              double offset_factor = 1.0 / (double)(col_offset + 1.0);
              
              // Get post-synaptic column mask
              LogicalVector post_column_mask = column_masks(_, tc);
              
              // Sample pre-synaptic cells 
              LogicalVector pre_mask = pre_type_mask & pre_layer_mask & pre_column_mask;
              if (!any_true(pre_mask)) {continue;} // Skip if no pre-synaptic cells of the right type in this column and layer
              IntegerVector pre_indices = Rwhich(pre_mask);
              int n_pre = R::rpois(pre_indices.size() * density_pre * offset_factor);
              n_pre = std::min(n_pre, 1);
              IntegerVector pre_sampled = Rcpp::sample(pre_indices, n_pre, false);
              
              // Prepare for repeated sampling of post-synaptic cells
              LogicalVector post_mask = post_type_mask & post_layer_mask & post_column_mask;
              if (!any_true(post_mask)) {continue;} // Skip if no post-synaptic cells of the right type in this column and layer
              IntegerVector post_indices = Rwhich(post_mask);
              
              for (int pre_c : pre_sampled) {
                
                // Sample post-synaptic cells
                int n_post = R::rpois(post_indices.size() * density_post * offset_factor);
                n_post = std::min(n_post, 1);
                IntegerVector post_sampled = Rcpp::sample(post_indices, n_post, false);
                
                // Set transductances 
                for (int post_c : post_sampled) {
                  double trans = R::runif(0.0, 2.0) * transductance_bias;
                  if (trans > pruning_threshold) {
                    motif_transconductances(post_c, pre_c) = val_pre * trans;
                    // Save edge coordinate
                    motif_edges_pre.push_back(pre_c);
                    motif_edges_post.push_back(post_c);
                  }
                  
                }
                
              }
              
            } 
            
          }
          
        }
        
      }
      
    }
    
    // Save to transconductance matrix vector 
    transconductances.push_back(motif_transconductances);
    
    // Collect local edge coordinates in matrix
    int n_motif_edges = motif_edges_pre.size();
    MatrixXi motif_edges(n_motif_edges, 2); 
    motif_edges.col(0) = Eigen::Map<VectorXi>(motif_edges_pre.data(), n_motif_edges);
    motif_edges.col(1) = Eigen::Map<VectorXi>(motif_edges_post.data(), n_motif_edges);
    
    // Save to edge types
    edge_types.push_back(motif_edges);
    
    // Add motif name
    edge_type_names.push_back(cmot.motif_name);
    
  }

/*
 * ***********************************************************************************
 * Neuron member function implementations, basic data handling, loading
 */

// Loading trial data (Eigen matrix)
void neuron::load_trial_data(const MatrixXd& td) {
    
    // Set trial data
    trial_data = td;
    
    // Compute mean neuron value (e.g., firing rate)
    lambda = trial_data.sum()/(double)(trial_data.rows() * trial_data.cols());
    lambda_bin = lambda * t_per_bin;
    
    // Compute standard deviation
    spike_sd = sqrt(trial_data.array().square().sum()/(double)(trial_data.rows() * trial_data.cols()) - lambda * lambda);
    
    // Make spike raaster
    if (unit_data == "spike") {
      infer_raster();
    }
    
  } 

// Loading trial data (Rcpp matrix)
void neuron::load_trial_data_R(const NumericMatrix& td) {
    
    // Set trial data
    trial_data = as<MatrixXd>(td);
    
    // Compute mean neuron value (e.g., firing rate)
    lambda = trial_data.sum()/(double)(trial_data.rows() * trial_data.cols());
    lambda_bin = lambda * t_per_bin;
    
    // Compute standard deviation
    spike_sd = sqrt(trial_data.array().square().sum()/(double)(trial_data.rows() * trial_data.cols()) - lambda * lambda);
    
    // Make spike raster
    if (unit_data == "spike") {
      infer_raster();
    }
    
  } 

// Loading spike raster (Eigen matrix)
void neuron::load_spike_raster(
    const MatrixXd& sr,
    const int& min_duration,
    const int& max_displacement
  ) {
    
    // Check input form
    if (unit_data != "spike") {Rcpp::stop("Spike raster can only be loaded if unit_data is 'spike'");}
    if (sr.cols() != 2) {Rcpp::stop("Spike raster must have two columns");}
   
    // Save spike raster
    spike_raster = sr;
    
    // Infer trial data
    infer_trial(min_duration, max_displacement);
    
    // Compute mean neuron value (e.g., firing rate)
    lambda = trial_data.sum()/(double)(trial_data.rows() * trial_data.cols());
    lambda_bin = lambda * t_per_bin;
    
    // Compute standard deviation
    spike_sd = sqrt(trial_data.array().square().sum()/(double)(trial_data.rows() * trial_data.cols()) - lambda * lambda);
    
  } 

// Loading spike raster (Rcpp matrix)
void neuron::load_spike_raster_R(
    const NumericMatrix& sr,
    const int& min_duration,
    const int& max_displacement
  ) {
    
    // Check input form
    if (unit_data != "spike") {Rcpp::stop("Spike raster can only be loaded if unit_data is 'spike'");}
    if (sr.ncol() != 2) {Rcpp::stop("Spike raster must have two columns");}
    CharacterVector sr_colnames = colnames(sr); 
    if (sr_colnames[0] != "time" || sr_colnames[1] != "trial") {Rcpp::stop("Spike raster must have columns named 'time' and 'trial'");}
    
    // Save spike raster
    spike_raster = as<MatrixXd>(sr);
    
    // Infer trial data
    infer_trial(min_duration, max_displacement);
    
    // Compute mean neuron value (e.g., firing rate)
    lambda = trial_data.sum()/(double)(trial_data.rows() * trial_data.cols());
    lambda_bin = lambda * t_per_bin;
    
    // Compute standard deviation
    spike_sd = sqrt(trial_data.array().square().sum()/(double)(trial_data.rows() * trial_data.cols()) - lambda * lambda);
    
  } 

/*
 * ***********************************************************************************
 * Neuron and network member function implementations, basic data handling, fetching
 */

// Fetching trial data (Eigen matrix)
MatrixXd neuron::fetch_trial_data() const {return trial_data;} 

// Fetching trial data (Rcpp matrix)
NumericMatrix neuron::fetch_trial_data_R() const {
    NumericMatrix trial_data_R = wrap(trial_data);
    colnames(trial_data_R) = enum_prefix("trial", trial_data_R.ncol());
    rownames(trial_data_R) = enum_prefix("time", trial_data_R.nrow());
    return trial_data_R;
  } 

// Fetching spike raster (Eigen matrix)
MatrixXd neuron::fetch_spike_raster() const {return spike_raster;} 

// Fetching spike raster (Eigen matrix)
NumericMatrix neuron::fetch_spike_raster_R() const {
    NumericMatrix spike_raster_R = wrap(spike_raster);
    colnames(spike_raster_R) = CharacterVector::create("time", "trial");
    return spike_raster_R;
  } 

// Method to return fields with neuron ID information
List neuron::fetch_id_data() const {
    
    return List::create(
      _["id_num"] = id_num,
      _["recording_name"] = recording_name,
      _["type"] = type,
      _["genotype"] = genotype,
      _["sex"] = sex,
      _["hemi"] = hemi,
      _["region"] = region,
      _["age"] = age,
      _["sim"] = sim,
      _["unit_time"] = unit_time,
      _["unit_sample_rate"] = unit_sample_rate,
      _["unit_data"] = unit_data,
      _["t_per_bin"] = t_per_bin,
      _["sample_rate"] = sample_rate
    );
    
  }

// Method to return firing rate
NumericVector neuron::fetch_lambda() const {
    NumericVector lambdas = {lambda, lambda_bin};
    lambdas.names() = CharacterVector({"lambda", "lambda_bin"});
    return lambdas;
  }

// Method to fetch network components 
List network::fetch_network_components(
    const bool& include_arbors
  ) const {
   
    // Convert transconductances into list of NumericMatrix
    List transconductance_matrices(transconductances.size());
    for (int tci = 0; tci < transconductances.size(); tci++) {
      MatrixXd tc = transconductances[tci];
      NumericMatrix tc_r = to_NumMat(tc);
      transconductance_matrices[tci] = tc_r;
    } 
    
    // Convert edge_types into list of NumericMatrix
    List edge_type_matrices(edge_types.size());
    CharacterVector emn = CharacterVector::create("pre_neuron_idx", "post_neuron_idx");
    for (int eti = 0; eti < edge_types.size(); eti++) {
      MatrixXi et = edge_types[eti];
      NumericMatrix et_r = to_NumMat(et);
      for (double &v : et_r) v++; // put into 1-indexed form for R
      colnames(et_r) = emn;
      edge_type_matrices[eti] = et_r;
    }
    
    // Convert arbors into list of numeric matrices
    List arbor_list;
    if (include_arbors) {
      for (int n = 0; n < n_neurons; n++) {
        int n_arbors = arbors[n].axon.size();
        int n_segments = 0;
        for (int a = 0; a < n_arbors; a++) {
          n_segments += arbors[n].coordinates[a].size();
        }
        NumericMatrix arbor_r(n_segments, 6);
        int last_seg_idx = 0;
        for (int a = 0; a < n_arbors; a++) {
          int n_segs = arbors[n].coordinates[a].size();
          double arbor_type = arbors[n].axon[a] ? 1.0 : 0.0; // 1 for axon, 0 for dendrite
          NumericVector parents_r = wrap(arbors[n].parents[a]);
          for (double &v : parents_r) v++; // put into 1-indexed form for R
          IntegerVector leafs_r_idx = wrap(arbors[n].leafs[a]);
          NumericVector leafs_r = Rcpp::rep(0.0, n_segs);
          for (int i = 0; i < leafs_r_idx.size(); i++) {leafs_r[leafs_r_idx[i]] = 1.0;} // Set leafs to 1
          for (int i = 0; i < n_segs; i++) { 
            arbor_r(i + last_seg_idx, 0) = arbor_type;
            arbor_r(i + last_seg_idx, 1) = parents_r[i];
            arbor_r(i + last_seg_idx, 2) = leafs_r[i];
            arbor_r(i + last_seg_idx, 3) = arbors[n].coordinates[a][i][0]; // z
            arbor_r(i + last_seg_idx, 4) = arbors[n].coordinates[a][i][1]; // y
            arbor_r(i + last_seg_idx, 5) = arbors[n].coordinates[a][i][2]; // x
          }
          last_seg_idx += n_segs;
        }
        colnames(arbor_r) = CharacterVector::create("is_axon", "parent_idx", "is_leaf", "z", "y", "x");
        arbor_list.push_back(arbor_r);
      }
    } else {
      arbor_list = R_NilValue;
    }
    
    // Add labels 
    NumericMatrix coordinates_node_R = to_NumMat(coordinates_node);
    colnames(coordinates_node_R) = CharacterVector::create("patch_idx", "layer_idx", "column_idx");
    NumericMatrix coordinates_spatial_R = to_NumMat(coordinates_spatial);
    colnames(coordinates_spatial_R) = CharacterVector::create("z", "y", "x");
    NumericMatrix node_coordinates_spatial_R = to_NumMat(node_coordinates_spatial);
    colnames(node_coordinates_spatial_R) = CharacterVector::create("z", "y", "x");
    
    // Put into 1-indexed form
    for (double &v : coordinates_node_R) v++;
    
    return List::create(
      _["network_name"] = network_name,
      _["n_neurons"] = n_neurons,
      _["n_neuron_types"] = n_neuron_types,
      _["layer_names"] = layer_names, 
      _["transconductances"] = transconductance_matrices,
      _["node_coordinates_spatial"] = node_coordinates_spatial_R,
      _["coordinates_spatial"] = coordinates_spatial_R,
      _["coordinates_node"] = coordinates_node_R,
      _["neuron_type_name"] = neuron_type_name,
      _["neuron_type_num"] = neuron_type_num,
      _["node_range_ends"] = node_range_ends,
      _["edge_idx_by_type"] = edge_type_matrices, 
      _["edge_type_names"] = edge_type_names,
      _["sim_dt"] = sim_dt,
      _["arbor_list"] = arbor_list, 
      _["units"] = List::create(
        _["time"] = unit_time,
        _["sample_rate"] = unit_sample_rate,
        _["potential"] = unit_potential,
        _["current"] = unit_current,
        _["conductance"] = unit_conductance,
        _["distance"] = unit_distance
      )
    );
   
  }

// Methods to fetch SGT simulation results 
NumericMatrix network::fetch_sim_traces_R() const {return to_NumMat(sim_traces);}
NumericVector network::fetch_spike_counts_R() const {return to_NumVec(spike_counts);}

// Infer trial data from spike raster
void neuron::infer_trial(
    const int& min_duration,
    const int& max_displacement
  ) {
    
    // Find number of trials
    int num_trials = static_cast<int>(spike_raster.col(1).maxCoeff());
    
    // Find duration
    int duration = static_cast<int>(spike_raster.col(0).maxCoeff() - spike_raster.col(0).minCoeff());
    duration = ((duration + 9) / 10) * 10; // round up to nearest 10
    if (duration < min_duration) {duration = min_duration;}
    if (duration <= 0) {Rcpp::stop("Spike raster has zero or negative duration");}
    
    // Find displacement
    int displacement = static_cast<int>(std::round(spike_raster.col(0).minCoeff()));
    if (displacement > max_displacement) {displacement = max_displacement;}
    
    // Resize trial_data and set to zero
    trial_data.resize(duration, num_trials);
    trial_data.setZero();
    
    // Fill trial data with recorded spikes 
    for (int r = 0; r < spike_raster.rows(); r++) {
      int time_idx = std::max(0, static_cast<int>(std::round(spike_raster(r, 0))) - displacement - 1);
      int trial_idx = static_cast<int>(std::round(spike_raster(r, 1))) - 1;
      trial_data(time_idx, trial_idx) += 1.0;
    }
    
  }

// Infer spike raster from trial data
void neuron::infer_raster() {
    
    int num_spikes = static_cast<int>(trial_data.sum());
    spike_raster.resize(num_spikes, 2);
    int ctr = 0;
    
    for (int r = 0; r < trial_data.rows(); r++) {
      for (int c = 0; c < trial_data.cols(); c++) {
        if (trial_data(r, c) > 0.0) {
          spike_raster(ctr, 0) = static_cast<double>(r + 1);
          spike_raster(ctr, 1) = static_cast<double>(c + 1);
          ctr++;
        }
      }
    }
    
  }

// Fetching autocorrelation 
VectorXd neuron::fetch_autocorr() const {return autocorr;}

// Fetching autocorrelation, R
NumericVector neuron::fetch_autocorr_R() const {return wrap(autocorr);}
NumericVector neuron::fetch_autocorr_edf_R() const {return wrap(autocorr_edf);}
NumericVector neuron::fetch_sigma_gauss_R() const {return wrap(sigma_gauss);}

// Fetch fitted EDF parameters 
NumericVector neuron::fetch_EDF_parameters() const {
    NumericVector results = {A, tau, bias_term};
    results.names() = CharacterVector({"A", "tau", "bias_term"});
    return results;
  }

/*
 * ***********************************************************************************
 * Neuron and network member function implementations, analysis and simulation
 */

// Compute cross-correlation of this neuron with another neuron
VectorXd neuron::compute_crosscorrelation(
    const neuron& nrn_compare,
    const std::string& bin_count_action,
    const int& max_lag,
    const bool& use_raw,
    const bool& verbose
  ) {
    
    if (verbose) {
      Rcpp::Rcout << "Computing cross-correlation between neuron " << id_num << " (ref) and neuron " << nrn_compare.id_num << " (comparison)" << std::endl;
    }
    
    // Compare time units 
    if (unit_time != nrn_compare.unit_time) {
      Rcpp::stop("Both neurons must have the same unit_time for cross-correlation calculation");
    } else if (verbose) {
      Rcpp::Rcout << "Both neurons have unit_time = " << unit_time << std::endl;
    }
    
    // Compare bin size 
    if (t_per_bin != nrn_compare.t_per_bin) {
      Rcpp::stop("Both neurons must have the same t_per_bin for cross-correlation calculation");
    } else if (verbose) {
      Rcpp::Rcout << "Both neurons have t_per_bin = " << t_per_bin << " " << unit_time << std::endl;
    }
    
    // Compare trial lengths
    const int trial_length_ref = trial_data.rows();
    const int trial_length_comp = nrn_compare.trial_data.rows();
    if (verbose) {
      Rcpp::Rcout << "Reference neuron trial length: " << trial_length_ref << " " << unit_time << std::endl;
      Rcpp::Rcout << "Comparison neuron trial length: " << trial_length_comp << " " << unit_time << std::endl;
    }
    if (trial_length_comp < trial_length_ref) {Rcpp::stop("Comparison neuron must have trial length at least as long as reference neuron for cross-correlation calculation");}
    if (max_lag > trial_length_ref) {Rcpp::stop("max_lag must be less than or equal to the trial length of the reference neuron for cross-correlation calculation");}
    
    // Check time unit of the trial matrix, home cell
    bool convert_to_bins = false;
    int max_lag_bin = max_lag;
    int trial_length_ref_bin = trial_length_ref;
    int trial_length_comp_bin = trial_length_comp;
    if (unit_time != "bin") {
      convert_to_bins = true;
      max_lag_bin = (int) max_lag/t_per_bin;
      trial_length_ref_bin = (int) trial_length_ref/t_per_bin;
      trial_length_comp_bin = (int) trial_length_comp/t_per_bin;
      if (verbose) {
        Rcpp::Rcout << "Converting to bins for cross-correlation calculation" << std::endl;
        Rcpp::Rcout << "max_lag in bins: " << max_lag_bin << std::endl;
        Rcpp::Rcout << "Reference neuron trial length in bins: " << trial_length_ref_bin << std::endl;
        Rcpp::Rcout << "Comparison neuron trial length in bins: " << trial_length_comp_bin << std::endl;
      }
    }
    
    // Check binning action
    if (bin_count_action != "sum" && bin_count_action != "boolean" && bin_count_action != "mean") {
      Rcpp::stop("bin_count_action must be 'sum', 'boolean', or 'mean'");
    } else if (verbose) {
      Rcpp::Rcout << "Using bin_count_action = " << bin_count_action << std::endl;
    }
    
    // Initialize matrices for data manipulation 
    const int N_trial = std::min(trial_data.cols(), nrn_compare.trial_data.cols());
    MatrixXd binned_data_ref(trial_length_ref_bin, N_trial);
    MatrixXd binned_data_comp(trial_length_comp_bin, N_trial);
    
    // Collapse bins if needed
    if (convert_to_bins) {
      
      for (int n = 0; n < N_trial; n++) {
        for (int b = 0; b < trial_length_ref_bin; b++) {
          // Sum spikes in bin
          double bin_count_ref = trial_data(seq(b*(int)t_per_bin, (b*(int)t_per_bin) + t_per_bin - 1), n).sum();
          // Apply binning action
          if (bin_count_action == "mean") {
            binned_data_ref(b, n) = bin_count_ref/(double)t_per_bin;
          } else if (bin_count_action == "sum") {
            binned_data_ref(b, n) = bin_count_ref;
          } else if (bin_count_action == "boolean") {
            binned_data_ref(b, n) = (bin_count_ref > 0.0) ? 1.0 : 0.0;
          }
        }
        for (int b = 0; b < trial_length_comp_bin; b++) {
          // Sum spikes in bin
          double bin_count_comp = nrn_compare.trial_data(seq(b*(int)t_per_bin, (b*(int)t_per_bin) + t_per_bin - 1), n).sum();
          // Apply binning action
          if (bin_count_action == "mean") {
            binned_data_comp(b, n) = bin_count_comp/(double)t_per_bin;
          } else if (bin_count_action == "sum") {
            binned_data_comp(b, n) = bin_count_comp;
          } else if (bin_count_action == "boolean") { 
            binned_data_comp(b, n) = (bin_count_comp > 0.0) ? 1.0 : 0.0;
          }
        } 
      }
      
    } else {
      // No binning
      binned_data_ref = trial_data;
      binned_data_comp = nrn_compare.trial_data;
    }
    
    // How many times should the lag be stepped across the data?
    int n_lag_steps = trial_length_ref_bin - max_lag_bin;
    int n_lag_steps_good = n_lag_steps;
    
    // compute cross-correlation as average of lagged correlations across all possible lag steps
    if (verbose) {
      Rcpp::Rcout << "Computing cross-correlation vector of length " << max_lag_bin << " bins" << std::endl;
    }
    VectorXd crosscorr(max_lag_bin);
    crosscorr.setZero();
    for (int lag_step = 0; lag_step <= n_lag_steps; lag_step++) {
      
      // Subset data to current lag step
      int ref_end = lag_step + max_lag_bin - 1;
      MatrixXd binned_data_ref_cut = binned_data_ref(seq(lag_step, ref_end), Eigen::all);
      // Comparison data should be longer, as it is shifted by max_lag_bin
      int comp_end = std::min(ref_end + max_lag_bin, trial_length_comp_bin);
      MatrixXd binned_data_comp_cut = binned_data_comp(seq(lag_step, comp_end), Eigen::all);
      // Find lagged correlation for this step
      VectorXd step_crosscorr = empirical_corr_lagged(binned_data_ref_cut, binned_data_comp_cut, max_lag_bin, use_raw);
      
      // If lagged correlation is well defined, add to running sum
      if ((step_crosscorr.array().isNaN()).any() || (step_crosscorr.array().isInf()).any()) {
        n_lag_steps_good -= 1;
      } else {
        crosscorr += step_crosscorr;
      }
    }
    // Reduce running sum to average
    crosscorr /= (double)(n_lag_steps + 1);
    
    if (verbose) {
      Rcpp::Rcout << "Possible lag steps: " << n_lag_steps << ", good lag steps: " << n_lag_steps_good << std::endl;
      Rcpp::Rcout << "Cross-correlation calculation complete" << std::endl;
    }
    
    crosscorr = roll_mean(crosscorr, t_per_bin); // smooth with rolling mean
    
    // Return 
    return crosscorr;
    
  }

// Wrapper
NumericVector neuron::compute_crosscorrelation_R(
    const neuron& nrn_compare,
    const std::string& bin_count_action,
    const int& max_lag,
    const bool& use_raw,
    const bool& verbose
  ) {
    VectorXd cc = compute_crosscorrelation(nrn_compare, bin_count_action, max_lag, use_raw, verbose);
    return to_NumVec(cc);
  }

// Compute autocorrelation of trial data
void neuron::compute_autocorrelation(
    const std::string& bin_count_action,
    int max_lag,
    const bool& use_raw
  ) {
    
    /*
     * Based on Neophytou et al 2022, "Differences in temporal processing ... "
     *   https://doi.org/10.1371/journal.pbio.3001803
     *   equation: x_n(t) = trial_matrix[t,n]
     */
    
    // Check time unit of the trial matrix
    bool convert_to_bins = false;
    int T_n = trial_data.rows();
    if (max_lag > T_n) {
      std::cout << "T_n: " << T_n << std::endl;
      std::cout << "max_lag: " << max_lag << std::endl;
      Rcpp::stop("max_lag must be less than or equal to the trial length");
    }
    if (unit_time != "bin") {
      convert_to_bins = true;
      max_lag = (int) max_lag/t_per_bin;
      T_n = (int) T_n/t_per_bin;
    }
    
    // Check binning action
    if (bin_count_action != "sum" && bin_count_action != "boolean" && bin_count_action != "mean") {
      Rcpp::stop("bin_count_action must be 'sum', 'boolean', or 'mean'");
    }
    
    // Initialize matrix for data manipulation 
    const int N_trial = trial_data.cols();
    MatrixXd data(T_n, N_trial);
    
    // Collapse bins if needed
    if (convert_to_bins) {
      
      for (int n = 0; n < N_trial; n++) {
        for (int b = 0; b < T_n; b++) {
          double bin_count = trial_data(seq(b*(int)t_per_bin, (b*(int)t_per_bin) + t_per_bin - 1), n).sum();
          if (bin_count_action == "mean") {
            data(b, n) = bin_count/(double)t_per_bin;
          } else if (bin_count_action == "sum") {
            data(b, n) = bin_count;
          } else if (bin_count_action == "boolean") {
            data(b, n) = (bin_count > 0.0) ? 1.0 : 0.0;
          }
        }
      }
      
      // Compute standard deviation, used below, in dg_parameters
      spike_sd_bin = sqrt(data.array().square().sum()/(double)(data.rows() * data.cols()) - lambda_bin * lambda_bin);
      
    } else {
      data = trial_data;
      spike_sd_bin = spike_sd;
    }
    
    // Find autocorrelation
    autocorr = empirical_corr_lagged(data, data, max_lag, use_raw);
    // ... smooth with rolling mean
    autocorr = roll_mean(autocorr, t_per_bin); 
    
  }

// Objective function for fitting EDF model to autocorrelation
double neuron::bounded_MSE_EDF_autocorr(
    const std::vector<double>& x, // 0 is A, 1 is tau
    std::vector<double>& grad,
    void* data                    // neuron object (this)
  ) {
    
    /*
     * EDF: autocor = A*exp(-lag/tau) + bias_term
     */
    
    // Grab neuron and estimated autocorrelation 
    neuron* nrn = static_cast<neuron*>(data);
    VectorXd est_autocorr = nrn->autocorr;
    int max_lag = est_autocorr.size();
    
    // Find mean squared error of autocorr prediction from these parameters x
    double bias_term = nrn->bias_term;
    double mse = 0.0;
    for (int i = 1; i < max_lag; i++) {
      // i = 1 because we throw out the first bin, which dominates
      double lag = (double)i;
      double pred = EDF_autocorr(lag, x[0], x[1], bias_term, 0);
      double err = est_autocorr(i) - pred;
      mse += err * err;
    }
    mse = mse/(double)(max_lag - 1.0);
    
    // Apply penalty to keep both parameters positive
    double penalty_multiple = nrn->penalty_multiple;
    double p1 = penalty_multiple/(x[0] * x[0]);
    double p2 = penalty_multiple/(x[1] * x[1]);
    
    // Compute gradient if needed
    if (!grad.empty()) {
      
      // Compute gradient of mse
      std::vector<double> grd = {0.0, 0.0};
      for (int i = 1; i < max_lag; i++) {
        // i = 1 because we throw out the first bin, which dominates
        double lag = (double)i;
        double fx = est_autocorr(i) - EDF_autocorr(lag, x[0], x[1], bias_term, 0);
        double gA = -EDF_autocorr(lag, x[0], x[1], bias_term, 1);
        double gtau = -EDF_autocorr(lag, x[0], x[1], bias_term, 2);
        grd[0] += 2.0 * fx * gA;
        grd[1] += 2.0 * fx * gtau;
      }
      grd[0] = grd[0]/(double)(max_lag - 1.0);
      grd[1] = grd[1]/(double)(max_lag - 1.0);
      
      // Compute gradient of penalty
      grd[0] = grd[0] - (2.0 * penalty_multiple * x[0])/(x[0] * x[0] * x[0] * x[0]);
      grd[1] = grd[1] - (2.0 * penalty_multiple * x[1])/(x[1] * x[1] * x[1] * x[1]);
      
      grad.assign({grd[0], grd[1]});
    } 
    
    return mse + p1 + p2;
    
  }

// Fit EDF model to autocorrelation
void neuron::fit_autocorrelation() { 
   
    // Grab initial parameters
    std::vector<double> x = {A0, tau0};
    size_t n = x.size();
    
    // Set penalty multiple 
    int max_lag = autocorr.size();
    double sum_err0_sq = 0.0;
    for (int i = 1; i < max_lag; i++) {
      // i = 1 because we throw out the first bin, which dominates
      double lag = (double)i;
      double pred0 = EDF_autocorr(lag, A0, tau0, bias_term, 0);
      double err0 = autocorr(i) - pred0;
      sum_err0_sq += err0 * err0;
    }
    double mse0 = sum_err0_sq/(double)(max_lag - 1.0);
    
    // Set bias term
    bias_term = lambda * t_per_bin;
    bias_term = bias_term * bias_term;
    
    // When A is 1.0 and tau is 10.0, want p1 + p2 = mse0 * penalty_weight
    const double penalty_weight = 0.25;
    penalty_multiple = (mse0 * penalty_weight)/(1.0/(A0 * A0) + 1.0/(tau0 * tau0));
    
    // Set up NLopt optimizer
    nlopt::srand(static_cast<unsigned long>(bias_term*1e6));
    nlopt::opt opt(nlopt::LD_LBFGS, n); 
    opt.set_min_objective(neuron::bounded_MSE_EDF_autocorr, this);
    opt.set_ftol_rel(ctol);       // stop when iteration changes objective fn value by less than this fraction 
    opt.set_maxeval(max_evals);   // Maximum number of evaluations to try
    
    // Fit model
    int success_code = 0;
    double min_fx;
    try {
      nlopt::result sc = opt.optimize(x, min_fx);
      success_code = static_cast<int>(sc);
    } catch (std::exception& e) { 
      if (false) {
        Rcpp::Rcout << "Optimization failed: " << e.what() << std::endl;
      } 
      success_code = 0;
    } 
    
    // Compute final modelled autocorrelation 
    autocorr_edf.resize(max_lag);
    for (int i = 0; i < max_lag; i++) {
      // Want i = 0 here for the construction of toeplitz sigma matrix
      double lag = (double)i;
      autocorr_edf(i) = EDF_autocorr(lag, x[0], x[1], bias_term, 0);
    }
    
    // Save optimization results, in proper time units
    A = x[0];
    tau = x[1] * t_per_bin;
    
  } 

// Find parameters for dichotomized Gaussian simulation
void neuron::dg_parameters(
    const bool& use_raw,
    const bool& verbose
  ) {
    
    if (verbose) {Rcpp::Rcout << "Finding dichotomized Gaussian parameters for neuron " << id_num << ", " << recording_name << " ..." << std::endl;}
    
    // Check that autocorrelation has been modeled
    int max_lag = autocorr_edf.size();
    if (max_lag == 0) {
      Rcpp::stop("autocorr_edf must be computed before dg_parameters");
    }
    
    // Compute gamma (dichotomizing threshold) needed to simulate firing rate lambda 
    gamma = norm_cdf(
      1 - lambda_bin,
      0.0,   // mean
      1.0,   // sd
      true   // return inverse
    );
    
    // Resize sigma_gauss
    sigma_gauss.resize(max_lag);
    
    // Find the covariance sigma_gauss for each lag
    if (use_raw) {
      for (int i = 0; i < max_lag; i++) {
        // When raw, autocorr_edf = E[X1*X2], so need to subtract off E[X1]*E[X2] = lambda^2 to get covariance
        // Want lambda_bin (not lambda) because these parameters are going to simulations, which have time units of bin
        sigma_gauss[i] = dg_find_sigma_RootBisection(gamma, autocorr_edf[i] - lambda_bin*lambda_bin);
      }
    } else {
      for (int i = 0; i < max_lag; i++) {
        // When using Pearson correlation, autocorr_edf = cov/(sd1*sd2), so need to multiply by sd1*sd2 to get covariance
        // Want spike_sd_bin (not lambda) because these parameters are going to simulations, which have time units of bin
        sigma_gauss[i] = dg_find_sigma_RootBisection(gamma, autocorr_edf[i] * spike_sd_bin*spike_sd_bin);
      }
    }
    
  }

// Make dichotomized Gaussian simulation of neuron
neuron neuron::dg_simulation(
    const int& trials,
    const bool& verbose
  ) {
   
    if (verbose) {Rcpp::Rcout << "Running dichotomized Gaussian simulation of neuron " << id_num << ", " << recording_name << " ..." << std::endl;}
    
    // Check that the covariance matrix sigma_guass has been computed
    int max_lag = sigma_gauss.size();
    if (max_lag == 0) {Rcpp::stop("sigma_gauss must be computed before dg_simulation");}
    
    // Make random draws
    NumericMatrix sigma_gauss_matrix = makePositiveDefinite(toeplitz(sigma_gauss, sigma_gauss));
    NumericVector mu = rep(0.0, max_lag); // Drawing from MVN with mean = 0 and sd = 1
    NumericMatrix simulated_trials_transpose = mvnorm_random(
      trials, 
      mu,
      sigma_gauss_matrix // Must be covariance matrix (positive definite)
    ); 
    
    // Take transpose, mvrnorm puts "points" (trials) as rows
    MatrixXd simulated_trials = to_eMat(simulated_trials_transpose).transpose();
    
    // Dichotomize
    for (int j = 0; j < simulated_trials.cols(); j++) {
      for (int i = 0; i < simulated_trials.rows(); i++) {
        simulated_trials(i, j) = (simulated_trials(i, j) < gamma) ? 0.0 : 1.0;
      }
    }
    
    // Make a copy of this neuron 
    neuron my_sim = neuron(*this);
    
    // Load with simulated trials
    my_sim.load_trial_data(simulated_trials);
    my_sim.sim = true;
    my_sim.unit_time = "bin";
    my_sim.t_per_bin = 1.0;
    
    // return simulation 
    return my_sim;
    
  }

// Estimate autocorrelation parameters for a neuron with dichotomized Gaussian simulations
NumericMatrix neuron::estimate_autocorr_params(
    const int& trials_per_sim, 
    const int& num_sims,
    int max_lag,
    const std::string& bin_count_action,
    const double& A0,
    const double& tau0,
    const double& ctol,
    const int& max_evals,
    const bool& use_raw,
    const bool& verbose
  ) {
    
    if (verbose) {
      Rcpp::Rcout << "Estimating autocorrelation parameters for neuron " << id_num << ", " << recording_name << " ..." << std::endl;
    }
    
    // Pre-process base neuron, if needed
    if (autocorr_edf.size() == 0) {
      // ... set exponential decay function (EDF) parameters 
      set_edf_initials(A0, tau0);
      set_edf_termination(ctol, max_evals);
      // ... compute autocorrelation
      compute_autocorrelation(bin_count_action, max_lag, use_raw);
      // ... fit autocorrelation 
      fit_autocorrelation();
    }
    
    // Find parameters for dichotomized Gaussian simulation
    dg_parameters(
      use_raw,
      false // verbose?
    );
    
    // Initialize matrix to hold results 
    NumericMatrix sim_results(num_sims, 9); 
    
    for (int s = 0; s < num_sims; s++) {
      
      // Simulate neuron with dichotomized Gaussian
      neuron my_sim = dg_simulation(
        trials_per_sim, 
        false // verbose?
      );
      
      // Set exponential decay function (EDF) parameters 
      my_sim.set_edf_initials(A0, tau0);
      my_sim.set_edf_termination(ctol, max_evals);
      
      // Compute autocorrelation
      my_sim.compute_autocorrelation(bin_count_action, max_lag, use_raw);
      
      // Fit autocorrelation 
      my_sim.fit_autocorrelation();
      
      // Fetch results 
      VectorXd autocorr_edf_tail_eigen = my_sim.autocorr_edf.tail(my_sim.autocorr_edf.size() - 1);
      NumericVector autocorr_edf_tail = to_NumVec(autocorr_edf_tail_eigen);
      NumericVector sim_results_row = {
        my_sim.lambda, 
        my_sim.lambda_bin, 
        my_sim.A, 
        my_sim.tau * t_per_bin, // convert back to original time units
        my_sim.bias_term,
        my_sim.autocorr_edf[0],
        max(autocorr_edf_tail),
        mean(autocorr_edf_tail),
        min(autocorr_edf_tail)
      };
      
      sim_results.row(s) = sim_results_row;
      
    }
    
    return sim_results;
    
  }

// Simulate network responses to input current using Growth Transform model
void network::SGT(
    const NumericMatrix& stimulus_current_R, // matrix of stimulus currents, in unit_current, n_neurons x n_steps
    const double& dt                         // time step length, in unit_time
  ) {
    
    // Save dt
    sim_dt = dt;
    
    // Convert stimulus current to Eigen matrix
    MatrixXd stimulus_current = to_eMat(stimulus_current_R);
   
    // Check size of stimulus current matrix 
    if (stimulus_current.rows() != n_neurons) {Rcpp::stop("stimulus_current must have n_neurons rows");}
    
    // Find number of time steps to simulate
    const int n_steps = stimulus_current.cols();
    
    // Collapse the transconductances into a single matrix
    //   ... rows as post-synaptic, cols as pre-synaptic
    MatrixXd transconductances_sum = MatrixXd::Zero(n_neurons, n_neurons);
    for (const auto& m : transconductances) {transconductances_sum += m;}
    
    // Find pairwise distances between all neurons and convert into timestep lag matrix (rows as pre-synaptic, cols as post-synaptic)
    MatrixXi pair_lags = pairwise_lags(coordinates_spatial, neuron_transmission_velocity, dt);
    
    // Extract temporal modulation values 
    VectorXd neuron_temporal_modulation_bias = neuron_temporal_modulation.col(0);
    VectorXd neuron_temporal_modulation_timeconstant = neuron_temporal_modulation.col(1);
    VectorXd neuron_temporal_modulation_amplitude = neuron_temporal_modulation.col(2);
    
    // Resize matrix to hold simulated spike traces (membrane potential plus spike)
    sim_traces.resize(n_neurons, n_steps);
    sim_traces.setZero();
    sim_traces.col(0) = resting_potential;
    
    // Initialize matrix to hold simulated sub-threshold membrane potential traces (without spike)
    MatrixXd v_traces = MatrixXd::Zero(n_neurons, n_steps);
    v_traces.col(0) = resting_potential;
    
    // Resize spike_counts vector
    spike_counts.resize(n_neurons);
    spike_counts.setZero();
    
    // Initialize count to keep track of bursting
    VectorXd burst_step_counter = VectorXd::Zero(n_neurons);
    
    // Simulate each time step after the initial
    for (int t = 1; t < n_steps; t++) {
      
      // Compute each cell's membrane potential state (rows) as seen by each other cell (columns)
      MatrixXd v_traces_lagged = lagged_traces(t, pair_lags, v_traces);
      
      // Compute rate of change for total metabolic power dissipation in the network, w.r.t. each neuron
      // ... units of dHdv are power/voltage, i.e., Watts/mV = mA
      // ... key idea? if a change dv in voltage in any one cell causes a spike, then H increases as well
      VectorXd dHdv = network_power_dissipation_gradient(
        v_traces_lagged,
        v_traces.col(t - 1), 
        stimulus_current.col(t - 1), 
        transconductances_sum, 
        I_spike, 
        threshold
      );
      /*
       * Note: The definition of network_power_dissipation_gradient seems to imply that it's 
       * the _subthreshold_ voltage of inputs i which determines the power used by j for synaptic transductance ... 
       * but, shouldn't a subthreshold voltage in a presynaptic cell i mean that the postsynaptic cell j expends no 
       * energy on synaptic transduction? 
       */
      
      // For each neuron in network, at this time step, 
      // ... compute power to initiate a spike:
      VectorXd spike_initiation_power = dHdv_bound.array() * v_traces.col(t - 1).array();
      VectorXd rest_maintenance_power = dHdv.array() * v_bound.array();
      VectorXd spike_cost = spike_initiation_power - rest_maintenance_power;  
      // ... compute max power to initiate a spike
      VectorXd spike_initiation_power_from_rest = dHdv_bound.array() * v_bound.array();
      VectorXd maintenance_power = dHdv.array() * v_traces.col(t - 1).array();
      VectorXd max_spike_cost = spike_initiation_power_from_rest - maintenance_power; 
      // ... normalize spike cost
      VectorXd normalized_spike_cost = spike_cost.array()/max_spike_cost.array(); 
      
      // Multiple potential bound by normalized spike cost ... units: mV * W/W = mV
      VectorXd v_bound_fraction = v_bound.array() * normalized_spike_cost.array();
      
      // Set dv_dt based on v_bound fraction
      VectorXd dv_dt_unmodulated = v_bound_fraction - v_traces.col(t - 1);
      
      // Find temporal modulation for this time step with vectorized operations
      VectorXd neuron_temporal_modulation =
        neuron_temporal_modulation_bias.array() +
        neuron_temporal_modulation_amplitude.array() * (-burst_step_counter.array() / neuron_temporal_modulation_timeconstant.array()).exp();
      // ... update burst step counter
      burst_step_counter.array() += dt;
      // ... check if reset is needed
      burst_step_counter = (neuron_temporal_modulation.array() < neuron_temporal_modulation_bias.array() * 1.01).select(0.0, burst_step_counter);
      // ... rescale temporal modulation with dt
      VectorXd temporal_modulation_dt = neuron_temporal_modulation/dt; 
      
      // Find dv_dt by dividing by the temporal modulation
      VectorXd dv_dt = dv_dt_unmodulated.array() / temporal_modulation_dt.array();
      
      // Find new subthreshold membrane potential
      VectorXd v_subthreshold = v_traces.col(t - 1) + dv_dt; 
      // ... save for next step
      v_traces.col(t) = v_subthreshold;
      
      // Divide spike_potential (minus threshold) by spike current to get transimpedance value necessary for that spike potential
      VectorXd transimpedance = (spike_potential - threshold).array()/I_spike.array();
      
      // Find spike value
      VectorXd barrier_values = v_barrier(v_subthreshold, threshold, I_spike);
      VectorXd spike = transimpedance.array() * barrier_values.array(); 
      // ... update spike counts
      spike_counts += (barrier_values.array() / I_spike.array()).matrix();
      
      // Add spike to raw membrane potential and save to spike traces 
      sim_traces.col(t) = v_subthreshold + spike;
      
    }
    
  }

/*
 * RCPP_MODULE to expose class to R
 */

RCPP_EXPOSED_CLASS(neuron)
RCPP_MODULE(neuron) {
  class_<neuron>("neuron")
  .constructor<int, std::string, std::string, std::string, std::string, std::string, std::string, std::string, bool, std::string, std::string, std::string, double, double>()
  .method("set_edf_initials", &neuron::set_edf_initials)
  .method("set_edf_termination", &neuron::set_edf_termination)
  .method("load_trial_data_R", &neuron::load_trial_data_R)
  .method("load_spike_raster_R", &neuron::load_spike_raster_R)
  .method("fetch_trial_data_R", &neuron::fetch_trial_data_R)
  .method("fetch_spike_raster_R", &neuron::fetch_spike_raster_R)
  .method("fetch_autocorr_R", &neuron::fetch_autocorr_R)
  .method("fetch_autocorr_edf_R", &neuron::fetch_autocorr_edf_R)
  .method("fetch_sigma_gauss_R", &neuron::fetch_sigma_gauss_R)
  .method("fetch_EDF_parameters", &neuron::fetch_EDF_parameters)
  .method("fetch_id_data", &neuron::fetch_id_data)
  .method("fetch_lambda", &neuron::fetch_lambda)
  .method("compute_autocorrelation", &neuron::compute_autocorrelation)
  .method("compute_crosscorrelation_R", &neuron::compute_crosscorrelation_R)
  .method("fit_autocorrelation", &neuron::fit_autocorrelation)
  .method("dg_parameters", &neuron::dg_parameters)
  .method("dg_simulation", &neuron::dg_simulation)
  .method("estimate_autocorr_params", &neuron::estimate_autocorr_params);
}

RCPP_EXPOSED_CLASS(motif)
RCPP_MODULE(motif) {
  class_<motif>("motif")
  .constructor<std::string>()
  .method("load_projection", &motif::load_projection);
}

RCPP_EXPOSED_CLASS(network)
RCPP_MODULE(network) {
  class_<network>("network")
  .constructor<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, double, double>()
  .method("set_network_structure", &network::set_network_structure)
  .method("make_local_nodes", &network::make_local_nodes)
  .method("apply_circuit_motif", &network::apply_circuit_motif)
  .method("fetch_network_components", &network::fetch_network_components)
  .method("fetch_sim_traces_R", &network::fetch_sim_traces_R)
  .method("fetch_spike_counts_R", &network::fetch_spike_counts_R)
  .method("SGT", &network::SGT);
}

RCPP_EXPOSED_CLASS(Projection)
RCPP_MODULE(Projection) {
  class_<Projection>("Projection")
  .constructor()
  .field("pre_type",      &Projection::pre_type)
  .field("pre_layer",     &Projection::pre_layer)
  .field("pre_density",   &Projection::pre_density)
  .field("post_type",     &Projection::post_type)
  .field("post_layer",    &Projection::post_layer)
  .field("post_density",  &Projection::post_density);
}