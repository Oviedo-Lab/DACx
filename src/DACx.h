
// neuron.h
#ifndef DACX_H
#define DACX_H

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

// Return logical vector giving elements of left which match right
LogicalVector eq_left_broadcast(const CharacterVector& left, const String& right);
// ... overload
LogicalVector eq_left_broadcast(const std::vector<int>& left, const int& right);
// ... overload 
LogicalVector eq_left_broadcast(const VectorXi& left, const int& right);

// Convert boolean masks to integer indexes
IntegerVector Rwhich(const LogicalVector& x);

// Boolean quantifiers
bool any_true(const LogicalVector& x);
bool all_true(const LogicalVector& x);

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

// Better normal distribution function, with PCG and Box-Muller
double pcg_rnorm(
    double mean, 
    double sd,
    pcg32& rng
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

// Function to make a matrix positive definite
NumericMatrix makePositiveDefinite(
    const NumericMatrix& NumX
  );

// Find pairwise Euclidean distances for a set of points
MatrixXd pairwise_distances(
    const MatrixXd& points   // Rows as points, columns as dimensions
  );

// Find pairwise Euclidean distances for a set of points and convert directly into integer lags
MatrixXi pairwise_lags(
    const MatrixXd& coordinates_spatial,      // N x 3 (rows = neurons), columns z (patch), y (layer), x (column)
    const VectorXd& neuron_transmission_velocity,
    double dt
  ); 

/*
 * ***********************************************************************************
 * Network and related classes
 */

// Cell types used in the network
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

// Meso-scale axonal and dendritic projections
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
