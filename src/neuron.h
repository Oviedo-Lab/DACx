
// neuron.h
#ifndef NEURON_H
#define NEURON_H

// Rcpp
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <nlopt.hpp>
#include <boost/math/distributions/normal.hpp>
#include "pcg_random.hpp"
using namespace Rcpp;
using namespace Eigen;

/*
 * ***********************************************************************************
 * Helper functions
 */

// Build sequence of numbered string prefixes
CharacterVector enum_prefix(
    std::string prefix, 
    int n
  );

// Rolling mean
VectorXd roll_mean(
    const VectorXd& series,    // 1D vector of points to take rolling mean
    int filter_ws              // Size of window for taking rolling mean
  );
// ... overload
NumericVector roll_mean(
    const NumericVector& series, 
    int filter_ws 
  );
// ... overload
std::vector<double> roll_mean(
    const std::vector<double>& series, 
    int filter_ws  
  );

// Return logical vector giving elements of left which match right
LogicalVector eq_left_broadcast(const CharacterVector& left, const String& right);
// ... overload
LogicalVector eq_left_broadcast(const std::vector<int>& left, const int& right);
// ... overload 
LogicalVector eq_left_broadcast(const VectorXi& left, const int& right);

// Convert boolean masks to integer indexes
IntegerVector Rwhich(const LogicalVector& x);

// Convert between vector types
std::vector<double> to_dVec(const VectorXd& vec);
std::vector<double> to_dVec(const NumericVector& vec);
VectorXd to_eVec(const std::vector<double>& vec);
VectorXd to_eVec(const NumericVector& vec);
NumericVector to_NumVec(const VectorXd& vec);
NumericVector to_NumVec(const std::vector<double>& vec);
MatrixXd to_eMat(const NumericMatrix& X);
MatrixXi to_eiMat(const IntegerMatrix& X);
NumericMatrix to_NumMat(const MatrixXd& M);
NumericMatrix to_NumMat(const MatrixXi& M);
IntegerMatrix to_IntMat(const MatrixXi& M);

// Make random walk
NumericVector random_walk(
    const int& n_steps,
    const double& step_size,
    const unsigned int& seed
  );

/*
 * ***********************************************************************************
 * Functions for computing correlations
 */

// Empirical Pearson correlation between two vectors
double empirical_corr(
    const VectorXd& x,
    const VectorXd& y,
    const bool& use_raw
  );

// Empirical Pearson correlation between two variables sampled many times 
double empirical_corr_multisample(
    const MatrixXd& X,    // Rows as intratrial samples, columns as trials
    const MatrixXd& Y,    // Rows as intratrial samples, columns as trials
    const bool& use_raw
  );
  
// Estimate Pearson correlation across lags
VectorXd empirical_corr_lagged(
    const MatrixXd& TS1,  // Time series 1, rows as time points, columns as trials
    const MatrixXd& TS2,  // Time series 2, rows as time points, columns as trials
    const int& max_lag,
    const bool& use_raw
  );

// Estimate raw correlation across lags, raw version (no mean subtraction, no normalization by std)
VectorXd empirical_corr_lagged_raw(
    const MatrixXd& TS1,  // Time series 1, rows as time points, columns as trials
    const MatrixXd& TS2   // Time series 2, rows as time points, columns as trials
  );

// Exponential decay function (with gradients) for modelling
double EDF_autocorr(
    const double& lag, 
    const double& A, 
    const double& tau, 
    const double& bias_term, 
    const int& return_grad // 0 = function output, 1 = gradient wrt A, 2 = gradient wrt tau
  );

/*
 * ***********************************************************************************
 * Probability distribution functions
 */

// multivariate normal CDF, upper tail
double mvnorm_cdf_uppertail(
    const NumericVector& threshold, 
    const NumericMatrix& sigma
  );

// Normal CDF, with inverse
double norm_cdf(
    const double& x, 
    const double& mu, 
    const double& sd,
    const bool& inverse
  );

// Multivariate normal random number generator
MatrixXd mvnorm_random(
    int n, 
    VectorXd mu, 
    MatrixXd sigma
  );

// Better normal distribution function, with PCG and Box-Muller
double pcg_rnorm(
    double mean, 
    double sd,
    pcg32& rng
  );

/*
 * ***********************************************************************************
 * Dichotomized Gaussian helper functions
 */

// Formula for estimating sigma for dichotomized Gaussian simulation
NumericVector dg_sigma_formula(
    const double& threshold,      // threshold for dichotomization
    const NumericVector& cov,     // desired covarance after dichotomization
    const NumericMatrix& sigma    // covariance matrix of multivariate Gaussian
  );

// Wrapper for use with find-root-bisection algorithm 
double dg_sigma_formula_scalar(
    const double& threshold,      // threshold for dichotomization
    const double& cov,            // desired covarance after dichotomization
    const double& sigma           // Gaussian covariance
  );

// Function to find sigma by root bisection 
double dg_find_sigma_RootBisection(
    const double& threshold,      // threshold for dichotomization
    const double& cov             // desired covarance after dichotomization
  );

/*
 * ***********************************************************************************
 * Growth-transform helper functions
 */

// Membrane potential barrier function
VectorXd v_barrier(
    const VectorXd& v_input,      // Column vector of membrane potentials for a network of neurons at one time step
    const VectorXd& threshold,    // Spike threshold, in unit_potential
    const VectorXd& I_out         // Spike current, in unit_current
  );

// Create lagged voltage trace matrix to simulate transmission delays
MatrixXd lagged_traces(
    int n,
    const MatrixXi& lag,
    const MatrixXd& v
  );

// Gradient of total dissipated metabolic power in network, w.r.t. membrane potential
VectorXd network_power_dissipation_gradient(
    const MatrixXd& v_traces_lagged,  // n_neuron x n_steps matrix of membrane potentials, in unit_potential, from which to calculate derivative
    const VectorXd& v_traces_current, // n_neuron x 1 matrix (column vector) of membrane potentials, in unit_potential, from which to calculate derivative
    const VectorXd& stimulus_current, // n_neuron x 1 matrix (column vector) of stimulus currents, in unit_current, from which to calculate derivative
    const MatrixXd& transconductance, // n_neuron x n_neuron transconductance matrix, giving connections between neurons
    const double& I_spike,            // spike current, in unit_current
    const double& threshold           // spike threshold, in unit_potential
  );

/*
 * ***********************************************************************************
 * Matrix and vector operations
 */

// Create Toeplitz matrix
NumericMatrix toeplitz(
    const std::vector<double>& first_col, 
    const std::vector<double>& first_row
  );

// Function to make a matrix positive definite
NumericMatrix makePositiveDefinite(
    const NumericMatrix& NumX
  );

// Find pairwise Euclidean distances for a set of points
MatrixXd pairwise_distances(
    const MatrixXd& points   // Rows as points, columns as dimensions
  );

/*
 * ***********************************************************************************
 * Neuron and network classes
 */

class neuron {
  
  // private: Eventually move some of the public stuff in here? 
  
  // public:

  public:
    
    // Variables *********************************
    
    // ID parameters
    int id_num = 0;                               // Fixed ID number for neuron
    std::string recording_name = "not_provided";  // Recording (if any) on which this neuron is based
    std::string type = "generic";                 // Modeled electrophysiology of neuron, e.g. "generic", "blackbox" "LIF", "McCullochPitts", "excitatory", "inhibitory", etc.
    std::string genotype = "WT";                  // Genotype of animal, e.g. "WT", "KO", "MECP2", "transgenic", etc.
    std::string sex = "not_provided";             // Sex of animal
    std::string hemi = "not_provided";            // Hemisphere of neuron, e.g. "left", "right"
    std::string region = "not_provided";          // Brain region of neuron, e.g. "V1", "M1", "CA1", "PFC", etc.
    std::string age = "not_provided";             // Age of animal, e.g. "P0", "P7", "P14", "adult", etc.
    bool sim = false;                             // Whether this neuron is simulated or based on recorded data
    
    // Unit specifications
    std::string unit_time = "ms";                 // Unit of time, e.g., "ms", "bin", "sample"
    std::string unit_sample_rate = "Hz";          // Unit of recording sample rate, e.g., "Hz", "kHz"
    std::string unit_data = "mV";                 // Unit of data, e.g., "mV", "spike count"
    
    // Unit conversions 
    double t_per_bin = 1.0;                       // Time (in above units) per bin, e.g., 1 ms per bin
    double sample_rate = 1e4;                     // Sample rate (in above units), e.g., 10000 Hz
    
    // Data fields
    MatrixXd trial_data;                          // NxM matrix of doubles, rows as recording times (in "unit_time"), columns as trials, data values in "unit_data"
    MatrixXd spike_raster;                        // Nx2 matrix, each row one spike, columns as time (in "unit_time") and trial number
    double lambda;                                // Mean value of neuron, in "unit_data" per "unit_time"
    double lambda_bin;                            // Mean value of neuron, in "unit_data" per bin
    double spike_sd;                              // Standard deviation of "unit_data" (presumably spike counts) across trials
    double spike_sd_bin;                          // Standard deviation of "unit_data" (presumably spike counts) across trials, in time units of bin
    
    // Analysis fields
    VectorXd autocorr;                            // Estimated (observed) autocorrelation of trial_data
    VectorXd autocorr_edf;                        // Estimated autocorrelation of trial_data using EDF model
    std::vector<double> sigma_gauss;              // Covariance values for Gaussian simulation, in time units of bin
    int max_evals = 500;                          // Max number of evals when fitting EDF to autocorrelation
    double ctol = 1e-7;                           // Convergence tolerance when fitting EDF to autocorrelation
    double A0 = 0.1;                              // Initial amplitude of EDF model of autocorrelation 
    double tau0 = 10.0;                           // Initial time constant of EDF model of autocorrelation
    double A;                                     // Fitted amplitude of EDF model of autocorrelation
    double tau;                                   // Fitted time constant of EDF model of autocorrelation
    double bias_term;                             // Bias term for EDF model of autocorrelation
    double penalty_multiple;                      // For scaling boundary penalty terms when fitting EDF model of autocorrelation
    double gamma;                                 // Threshold for dichotomized Gaussian simulation (in time units of bin)
    
    // Functions *********************************
    
    // Constructor and Destructor
    neuron(
      const int id_num = 0, 
      const std::string recording_name = "not_provided", 
      const std::string type = "generic", 
      const std::string genotype = "WT",
      const std::string sex = "not_provided",
      const std::string hemi = "not_provided",
      const std::string region = "not_provided",
      const std::string age = "not_provided",
      const bool sim = false, 
      const std::string unit_time = "ms", 
      const std::string unit_sample_rate = "Hz", 
      const std::string unit_data = "mV", 
      const double t_per_bin = 1.0, 
      const double sample_rate = 1e4
    );
    virtual ~neuron() {};
    
    // Copy method 
    neuron(const neuron& other) = default;
    
    // Member functions for adjusting settings
    void set_edf_initials(double a0, double t0);
    void set_edf_termination(double ct, int me);
    
    // Member functions for loading data
    void load_trial_data(const MatrixXd& td);
    void load_trial_data_R(const NumericMatrix& td);
    void load_spike_raster(const MatrixXd& sr, const int& min_duration, const int& max_displacement);
    void load_spike_raster_R(const NumericMatrix& sr, const int& min_duration, const int& max_displacement);
    void infer_raster();
    void infer_trial(const int& min_duration, const int& max_displacement);
    
    // Member functions for fetching data
    MatrixXd fetch_trial_data() const;
    NumericMatrix fetch_trial_data_R() const;
    MatrixXd fetch_spike_raster() const;
    NumericMatrix fetch_spike_raster_R() const;
    List fetch_id_data() const;
    NumericVector fetch_lambda() const;
    
    // Member functions for fetching analysis results
    VectorXd fetch_autocorr() const;
    NumericVector fetch_autocorr_R() const;
    NumericVector fetch_autocorr_edf_R() const;
    NumericVector fetch_sigma_gauss_R() const;    // continuous (normal) equivalent of autocorr_edf
    NumericVector fetch_EDF_parameters() const;
    
    // Member functions for data analysis
    VectorXd compute_crosscorrelation(
      const neuron& nrn_compare,
      const std::string& bin_count_action,
      const int& max_lag,
      const bool& use_raw,
      const bool& verbose
    );
    NumericVector compute_crosscorrelation_R(
      const neuron& nrn_compare,
      const std::string& bin_count_action,
      const int& max_lag,
      const bool& use_raw,
      const bool& verbose
    );
    void compute_autocorrelation(
      const std::string& bin_count_action,        // action must be 'sum', 'boolean', or 'mean'
      int max_lag,                                // maximum lag (in unit_time) for autocorrelation
      const bool& use_raw                         // whether to use raw autocorrelation (true) or standard centered and normalized correlation (false)
    ); 
    static double bounded_MSE_EDF_autocorr(
      // Objective function for fitting EDF model to autocorrelation
      const std::vector<double>& x,               // 0 is A, 1 is tau
      std::vector<double>& grad,
      void* data                                  // neuron object (this)
    );
    void fit_autocorrelation();
    void dg_parameters(const bool& use_raw, const bool& verbose);
    neuron dg_simulation(const int& trials, const bool& verbose);
    NumericMatrix estimate_autocorr_params(
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
    );
    
  };

struct cell_type {
    std::string type_name;
    int valence;                         // valence of each neuron type, +1 for excitatory, -1 for inhibitory
    double temporal_modulation_bias;     // temporal modulation time (in unit_time) bias for each neuron type
    double temporal_modulation_timeconstant;     // temporal modulation time (in unit_time) step for each neuron type
    double temporal_modulation_amplitude;        // temporal modulation time (in unit_time) cutoff for each neuron type
    double transmission_velocity;        // transmission velocity (in unit_distance/unit_time) for each neuron type
    double v_bound;                      // potential bound, in unit_potential
    double dHdv_bound;                   // bound the derivative of metabolic energy wrt potential, in unit_current
    double I_spike;                      // spike current, in unit_current
    double coupling_scaling_factor;      // Controls how energy used in synaptic transmission compares to that used in spiking
    double spike_potential;              // Magnitude of each spike, in unit_potential
    double resting_potential;            // resting potential, in unit_potential
    double threshold;                    // spike threshold, in unit_potential
    int process_node_count;              // Sets n_segments in make_arbor, in terms of expected number of process nodes per process length
    int axon_branch_count;               // Sets n_branches in make_arbor, in terms of expected number of branches per process length
    int dendrite_branch_count;           // Sets n_branches in make_arbor, in terms of expected number of branches per process length
  };

struct Projection {
    std::string pre_type;
    std::string pre_layer;
    double pre_density;
    std::string post_type;
    std::string post_layer;
    double post_density;
  };

struct cell_arbors {
    std::vector<bool> axon;                            // axon[i] = whether arbor i is axon (true) or dendrite (false)
    std::vector<std::vector<Vector3d>> coordinates;    // Rows as process nodes (including soma coordinates); Columns z, y, x
    std::vector<std::vector<int>> parents;             // parents[i] = the idx in coordinates of the parent of node i in coordinates, with -1 for the soma
    std::vector<std::vector<int>> leafs;               // leafs[i] = coordinates idx of leaf i in coordinates matrix
  };

class motif {
  
  /*
   * Motifs are recipes for building internode projections within a neural network. They are 
   *   "columnar", in the sense that they are repeated across cortical columns. 
   */
  
  // private: Eventually move some of the public stuff in here? 
  
  // public:
  
  public:
    
    // Variables *********************************
    
    std::string motif_name = "not_provided";      // Name of motif
    std::vector<Projection> projections;
    std::vector<int> max_col_shift_up;            // Maximum number of columns to shift up when applying motif
    std::vector<int> max_col_shift_down;          // Maximum number of columns to shift down when applying motif
    std::vector<double> connection_strength;      // Strength of connection for each projection
    int n_projections = 0;                        // Number of projections in motif
    
    // Functions *********************************
    
    // Constructor and Destructor
    motif(
      const std::string motif_name = "not_provided"
    );
    virtual ~motif() {};
    
    // Copy method 
    motif(const motif& other) = default;
    
    // Load projection into motif
    void load_projection(
      const Projection& proj,
      const int& max_up = 0,
      const int& max_down = 0,
      const double& c_strength = 1.0
    );
    
  };

class network {
  
  /*
   * Networks are points (representing neurons) connected by directed edges. Within the growth-transform (GT) model
   *   framework, these edges have transconductance values representing synaptic connections between neurons.
   * 
   * Point types: Points can be grouped by types, which affect 
   *   their behavior and connectivity. Within the GT model framework, these types each have their own 
   *   temporal modulation constants (determining, e.g., whether the cell bursts or fires singular spikes) and 
   *   valence (excitatory or inhibitory).
   * 
   * Global structure: Modelling the mammalian cortex, networks are assumed to divide into a coarse-grained 
   *   two-dimensional coordinate system of layers (rows) and columns (columns). Each point is assigned to a layer-column
   *   coordinate (called a "node"), having both local x-y coordinates within that node and a global x-y coordinate within the network. 
   *   
   * Local structure: Each layer-column coordinate defines a "node" containing a number of points determined by layer and type. 
   *   Connections (edges) within a node are determined by a local recurrence factor matrix determining the transconductance between 
   *   points of each type. These edges are called "local". 
   *   
   * Long-range projections: Connections (edges) between points in different nodes are determined by a long-range projection motif and 
   *   labelled with the name of that motif. 
   * 
   */
  
  // private: Eventually move some of the public stuff in here? 
  
  // public:
  
  public:
    
    // Variables *********************************
    
    // ID parameters
    std::string network_name = "not_provided";    // Name of network
    std::string recording_name = "not_provided";  // Recording (if any) on which this network is based
    std::string type = "Growth_Transform";        // Type of network, only "Growth_Transform" currently supported
    std::string genotype = "WT";                  // Genotype of animal, e.g. "WT", "KO", "MECP2", "transgenic", etc.
    std::string sex = "not_provided";             // Sex of animal
    std::string hemi = "not_provided";            // Hemisphere of neuron, e.g. "left", "right"
    std::string region = "not_provided";          // Brain region of neuron, e.g. "V1", "M1", "CA1", "PFC", etc.
    std::string age = "not_provided";             // Age of animal, e.g. "P0", "P7", "P14", "adult", etc.
    
    // Unit specifications
    std::string unit_time = "ms";                 // Unit of time, e.g., "ms", "bin", "sample"
    std::string unit_sample_rate = "Hz";          // Unit of recording sample rate, e.g., "Hz", "kHz"
    std::string unit_potential = "mV";            // Unit of membrane potential, e.g., "mV"
    std::string unit_current = "mA";              // Unit of current, e.g., "mA", "nA"
    std::string unit_conductance = "mS";          // Unit of conductance, e.g., "mS", "uS"
    std::string unit_distance = "micron";         // Unit of distance, e.g., "micron", "mm"
    
    // Unit conversions 
    double t_per_bin = 1.0;                       // Time (in above units) per bin, e.g., 1 ms per bin
    double sample_rate = 1e4;                     // Sample rate (in above units), e.g., 10000 Hz
    
    // Network structure
    std::vector<cell_type> neuron_types;          // Types of neurons in network, e.g., "principal", "PV", "SST", "VIP"
    CharacterVector layer_names;                  // Names of layers in the network
    int n_layers = 1;                             // number of layers in the network
    int n_columns = 1;                            // number of columns in the network
    int n_patches = 1;                            // number of patches (rows of columns, i.e., n_layers x n_columns sheets) in the network
    double layer_height = 1.0;                    // sd of the normal distribution for local y coordinates of the neurons
    double column_diameter = 1.0;                 // sd of the normal distribution for local x coordinates of the neurons
    double layer_separation_factor = 1.25;        // factor to multiply layer height by to get the distance between layers
    double column_separation_factor = 1.5;        // factor to multiply column diameter by to get the distance between columns
    double patch_separation_factor = 1.5;         // factor to multiply column diameter by to get the distance between patches (rows of columns)
    MatrixXi neurons_per_node;                    // mean number of neurons in each layer (rows) by type (columns)
    std::vector<MatrixXd> recurrence_factors;     // Vector of matrices of sd of the normal distribution for local transconductances between neurons of each type, one matrix per layer
    double pruning_threshold_factor = 0.1;        // transconductances below this fraction of the recurrence factor set to zero
    
    // Network components 
    int n_neurons;                                // Total number of neurons in the network
    int n_neuron_types;                           // Number of different neuron types in the network
    std::vector<cell_arbors> arbors;
    std::vector<MatrixXd> transconductances;      // Vector of square matrices, each giving the transconductance between each neuron in the network, rows are post-synaptic, columns are pre-synaptic
    MatrixXd node_coordinates_spatial;            // Mx3 matrix giving the (z,y,x) spatial coordinates of each node in the network
    MatrixXd coordinates_spatial;                 // Nx3 matrix giving the (z,y,x) spatial coordinates of each neuron in the network
    MatrixXi coordinates_node;                    // Nx3 matrix giving the (patch, layer, column) node coordinates of each neuron in the network
    VectorXd v_bound;                             // Vector giving potential bound, such that -v_bound <= v_traces <= v_bound, in unit_potential, for each neuron in the network, based on its type
    VectorXd dHdv_bound;                          // Vector giving bound on derivative of metabolic energy wrt potential, such that dHdv_bound > abs(dHdv), in unit_current, for each neuron in the network, based on its type
    VectorXd I_spike;                             // Vector giving spike current, in unit_current, for each neuron in the network, based on its type
    VectorXd spike_potential;                     // Vector giving magnitude of each spike, in unit_potential, for each neuron in the network, based on its type
    VectorXd resting_potential;                   // Vector giving resting potential, in unit_potential, for each neuron in the network, based on its type
    VectorXd threshold;                           // Vector giving spike threshold, in unit_potential, for each neuron in the network, based on its type
    MatrixXd neuron_temporal_modulation;          // Nx3 matrix giving the temporal modulation time (in unit_time) bias, step, and cutoff for each neuron in the network, based on its type
    VectorXd neuron_transmission_velocity;        // Vector giving the transmission delay (in unit_time) for each neuron in the network, based on its type
    CharacterVector neuron_type_name;             // Vector giving the type of each neuron in the network, as a string
    std::vector<int> neuron_type_num;             // Vector giving the type of each neuron in the network, as an integer index
    std::vector<int> node_range_ends;             // Vector giving the ending neuron index for each node in the network
    std::vector<MatrixXi> edge_types;             // Vector of integer matrices giving all transconductance matrix coordinates for each edge type 
    CharacterVector edge_type_names = {"local connections"};  // Names of elements in edge_types
    
    // Data fields 
    double sim_dt;                                // Time step for simulation, in unit_time
    MatrixXd sim_traces;                          // NxT matrix of doubles, each column giving the simulated membrane potential of a neuron, each row giving a time-step in the simulation
    VectorXd spike_counts;                        // Vector of length N, giving the number of spikes for each neuron in the network during a simulation
    
    // Functions *********************************
    
    // Constructor and Destructor
    network(
      const std::string network_name = "not_provided", 
      const std::string recording_name = "not_provided", 
      const std::string type = "Growth_Transform", 
      const std::string genotype = "WT",
      const std::string sex = "not_provided",
      const std::string hemi = "not_provided",
      const std::string region = "not_provided",
      const std::string age = "not_provided",
      const std::string unit_time = "ms", 
      const std::string unit_sample_rate = "Hz", 
      const std::string unit_potential = "mV", 
      const std::string unit_current = "mA",
      const std::string unit_conductance = "mS",
      const std::string unit_distance = "micron",
      const double t_per_bin = 1.0, 
      const double sample_rate = 1e4
    );
    virtual ~network() {};
    
    // Copy method 
    network(const network& other) = default;
    
    // Member functions for adjusting settings
    void set_network_structure(
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
    );
    
    // Member functions for building network
    void make_arbor_branch(
      const int& cell_idx,                    // Number of neuron for which to make processes
      int n_segments,                         // Expected number of process segments on longest branch
      const bool& is_axon,                    // Whether to make axon (true) or dendrite (false)
      double segment_divisor = 0.0,           // Specifies expected length of segments in terms of column diameter and layer height, or distance to attractor point
      int parent_branch_idx = -1,             // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
      const Eigen::Matrix<double, 3, 1> attractor_point = {0.0, 0.0, 0.0}
    );
    void make_arbor(
      const int& cell_idx,                    // Number of neuron for which to make processes
      int n_segments,                         // Expected number of process segments on longest branch
      int n_branches,                         // Expected number of branches, including the main process 
      const bool& is_axon,                    // Whether to make axon (true) or dendrite (false)
      double segment_divisor = 0.0,           // Specifies expected length of segments in terms of column diameter and layer height, or distance to attractor point
      int parent_branch_idx = -1,             // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
      const Eigen::Matrix<double, 3, 1> attractor_point = {0.0, 0.0, 0.0}
    );
    void make_local_nodes(); 
    void apply_circuit_motif(const motif& cmot);
    
    // Member functions for fetching data 
    List fetch_network_components(const bool& include_arbors = false) const;
    NumericMatrix fetch_sim_traces_R() const;
    NumericVector fetch_spike_counts_R() const;
    
    // Member functions for analysis and simulation 
    void SGT(
      const NumericMatrix& stimulus_current,     // matrix of stimulus currents, in unit_current, n_neurons x n_steps
      const double& dt                           // time step length, in unit_time
    );
    
  };

#endif
