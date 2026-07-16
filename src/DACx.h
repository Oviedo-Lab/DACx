
// neuron.h
#ifndef DACX_H
#define DACX_H

// Rcpp
// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <nlopt.hpp>
using namespace Rcpp;
using namespace Eigen;

/*
 * ***********************************************************************************
 * Helper functions
 */

// Return logical vector giving elements of left which match right
LogicalVector Rmask(const CharacterVector& left, const String& right);
// ... overload 
LogicalVector Rmask(const CharacterVector& left, const std::string& right);
// ... overload
LogicalVector Rmask(const std::vector<int>& left, const int& right);
// ... overload 
LogicalVector Rmask(const VectorXi& left, const int& right);

// Convert boolean masks to integer indexes
IntegerVector Rwhich(const LogicalVector& x);
// ... overload 
IntegerVector Rwhich(const std::vector<bool>& x);

// Boolean quantifiers
bool any_true(const LogicalVector& x);
bool any_true(const std::vector<bool>& x);
bool all_true(const LogicalVector& x);
bool all_true(const std::vector<bool>& x);

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

/*
 * ***********************************************************************************
 * Growth-transform helper functions
 */

// Membrane potential barrier function
VectorXd v_barrier(
    const VectorXd& v_input,      // Column vector of membrane potentials for a network of neurons at one time step
    const VectorXd& threshold,    // Spike threshold, in mV
    const VectorXd& I_out         // Spike current, in pA
  );

// Create lagged voltage trace matrix to simulate transmission delays
MatrixXd lagged_traces(
    int n,
    const MatrixXi& lag,
    const MatrixXd& v
  );

// Gradient of total dissipated metabolic power in network, w.r.t. membrane potential
VectorXd network_power_dissipation_gradient(
    const MatrixXd& v_traces_lagged,  // n_neuron x n_steps matrix of membrane potentials, in mV, from which to calculate derivative
    const VectorXd& v_traces_current, // n_neuron x 1 matrix (column vector) of membrane potentials, in mV, from which to calculate derivative
    const VectorXd& membrane_current, // n_neuron x 1 matrix (column vector) of membrane currents, in pA, from which to calculate derivative
    const MatrixXd& transconductance, // n_neuron x n_neuron transconductance matrix, giving connections between neurons, in nS
    const VectorXd& I_spike,          // spike current, in pA
    const VectorXd& threshold         // spike threshold, in mV
  );

// Derivative of depressive weight, from Schiff & Reyes 2012 (https://doi.org/10.1152/jn.00208.2011) 
VectorXd dWdt(
    const VectorXd& W,                  // Vector of depressive weights, one per cell
    const VectorXd& recent_spike_count, // Vector of counts of recent spikes, per cell; conceptually combines firing rate and depressive factor, FR * DF
    const VectorXd& tau_STD_recovery    // Vector giving time constant for recovery from depression, per cell
  );

/*
 * ***********************************************************************************
 * Matrix and vector operations
 */

// Find first neighbor 
std::vector<int> find_first_neighbor(
    const std::vector<Vector3d>& b_active,     // Branch searching for neighbor
    const std::vector<Vector3d>& b_all,        // Branch being searched
    const double& neighborhood_radius,
    const bool& skip_origin = true
  );

/*
 * ***********************************************************************************
 * Cell types and related functions
 */

// Cell types used in the network
struct cell_type {
    // ID information
    std::string type_name;
    // Excitatory or inhibitory?
    int valence;                         // valence of each neuron type, +1 for excitatory, -1 for inhibitory
    // Membrane kinetics (burst control)
    double temporal_modulation_bias;     // temporal modulation time (ms) bias for each neuron type
    double temporal_modulation_timeconstant;     // temporal modulation time (ms) step for each neuron type
    double temporal_modulation_amplitude;        // temporal modulation time (ms) cutoff for each neuron type
    double spike_recovery_rate;          // Number of spikes which can be "cleared" per ms
    double tau_STD_recovery;             // Time constant for recovery from short-term depression (STD), in spikes/ms; requires tau_STD_recovery < spike_recovery_rate
    // Intercell transmission
    double transmission_velocity;        // transmission velocity (in micron/ms) for each neuron type
    double spine_density;                // Scale between 0 and 1: 0 = no nodes have spines, 1 = all nodes have spines
    std::string axon_target;             // "spine", "dendrite_shaft", "soma", and "axon_shaft"
    // Membrane characteristics
    double I_spike;                      // spike current, in pA; absolute value plus a little bit used as dHdv_bound
    double spike_potential;              // Magnitude of each spike, in mV
    double resting_potential;            // resting potential, in mV; absolute value plus a little bit used as v_bound
    double threshold;                    // spike threshold, in mV
    double leak_conductance;             // conductance controlling the leak current, I_leak = leak_conductance (resting_potential - v), in nS
    // Process size and structure parameters
    int axon_branch_count;               // Sets expected number n_branches in make_arbor for axons
    int dendrite_branch_count;           // Sets expected number n_branches in make_arbor for dendrites
    double branch_independence;          // Scale between 0 and 1; 0 = all branches connect to soma from single segment, 1 = all branches connect directly to soma
    double branch_spread;                // Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 = straight line away from soma
    // Apical dendrite parameters 
    std::string apical_target_layer;     // Layer to which apical dendrite is expected to grow, if any; if none, "none"
  };

// Singleton accessor for the cell type registry
std::unordered_map<std::string, cell_type>& get_cell_types();

// Internal helper: construct a cell_type from a named Rcpp List
cell_type build_cell_type_from_list(const List& params);

// Print known cell types 
void print_known_celltypes();

// Fetch cell type parameters 
List fetch_cell_type_params(const std::string& type_name);

// Add new cell type (params is a fully-specified named List)
void add_cell_type(const List& params);

// Modify existing cell type (params is a fully-specified named List)
void modify_cell_type(const std::string& type_name, const List& params);

/*
 * ***********************************************************************************
 * Network and related classes
 */

// Meso-scale axonal and dendritic projections
struct Projection {
    std::string pre_type;
    std::string pre_layer;
    std::string post_type;
    std::string post_layer;
  };

// Node-tree description of process arbors, one structure instance per cell
struct cell_arbors {
    std::vector<int> arbor_id;                         // arbor_id[i] = unique id for arbor i
    std::vector<bool> axon;                            // axon[i] = whether arbor i is axon (true) or dendrite (false)
    std::vector<std::vector<Vector3d>> coordinates;    // coordinates[i][j] j = coordinates z, y, x of process node j on arbor i (including soma coordinates for j = 0)
    std::vector<std::vector<std::string>> node_type;   // node_type[i][j] = "soma", "dendrite_shaft", "axon_shaft", or "spine" for node j in arbor i
    std::vector<std::vector<int>> parents;             // parents[i][j] = the node number (idx in coordinates) of the parent of node j in arbor i, with -1 for the soma
    std::vector<std::vector<int>> leafs;               // leafs[i][j] = 1 if node j in arbor i is a leaf, 0 otherwise
    std::vector<std::vector<int>> synapses;            // synapses[i][j] = number of synapses on node j in arbor i, with 0 for non-synaptic nodes
  };

// Cortical projection motif
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
    std::vector<Projection> projections;          // List of projection descriptions
    std::vector<int> max_col_shift_up;            // Maximum number of columns to shift up when applying motif
    std::vector<int> max_col_shift_down;          // Maximum number of columns to shift down when applying motif
    std::vector<double> projection_conductance;   // Strength of connection for each projection
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
      const int& max_up,
      const int& max_down,
      const double& proj_conductance
    );
    
  };

// Cortical network model
class network {
  
  // Suggest using values in units of ms (time), mV (potential), pA (current), nS (conductance), micron (distance)
  
  // private: Eventually move some of the public stuff in here? 
  
  // public:
  
  public:
    
    // Variables *********************************
    
    // Network structure
    std::vector<cell_type> neuron_types;             // Types of neurons in network, e.g., "pyramidal", "PV", "SST", "VIP"
    CharacterVector        layer_names;              // Names of layers in the network
    int                    n_layers  = 1;            // Number of layers in the network
    int                    n_columns = 1;            // Number of columns in the network
    int                    n_patches = 1;            // Number of patches (rows of columns, i.e., n_layers x n_columns sheets) in the network
    double                 layer_height;             // sd of the normal distribution for local y coordinates of the neurons
    double                 column_diameter;          // sd of the normal distribution for local x coordinates of the neurons
    double                 segment_length;           // Expected length of segments in process arbors, in micron
    double                 layer_separation_factor;  // Factor to multiply layer height by to get the distance between layers
    double                 column_separation_factor; // Factor to multiply column diameter by to get the distance between columns
    double                 patch_separation_factor;  // Factor to multiply column diameter by to get the distance between patches (rows of columns)
    double                 synaptic_neighborhood;    // Radius of synapse-forming neighborhood; axon-dendrite node pairs within this distance initialize as synapses
    MatrixXi               neurons_per_node;         // Mean number of neurons in each layer (rows) by type (columns)
    std::vector<MatrixXd>  local_conductance;        // Vector of matrices of sd of the normal distribution for local transconductances between neurons of each type, one matrix per layer, in nS
    
    // Network components 
    int                      n_neurons;              // Total number of neurons in the network
    int                      n_neuron_types;         // Number of different neuron types in the network
    MatrixXi                 synapse_arbor_idx;      // n_neuron x n_neuron matrix of synapse indexes, with -1 for no synapse and otherwise synapse_arbor_idx[i,j] = index ax in arbors[i].coordinates[ax] of the axon of cell i holding the synapse into cell j
    MatrixXi                 synapse_node_idx;       // n_neuron x n_neuron matrix of synapse indexes, with -1 for no synapse and otherwise synapse_node_idx[i,j] = index k in arbors[i].coordinates[ax][k] of the synapse from neuron i to neuron j
    std::vector<cell_arbors> arbors;                 // Vector of length n_neurons
    std::vector<MatrixXd>    transconductances;      // Vector of square matrices, each giving the transconductance between each neuron in the network, rows are post-synaptic, columns are pre-synaptic, in nS
    MatrixXd                 node_coordinates_spatial;  // Mx3 matrix giving the (z,y,x) spatial coordinates of each node in the network
    MatrixXd                 coordinates_spatial;    // n_neurons x 3 matrix giving the (z,y,x) spatial coordinates of each neuron in the network
    MatrixXi                 coordinates_node;       // n_neurons x 3 matrix giving the (patch, layer, column) node coordinates of each neuron in the network
    VectorXd                 v_bound;                // Vector giving potential bound, such that -v_bound <= v_traces <= v_bound, in mV, for each neuron in the network, based on its type
    VectorXd                 dHdv_bound;             // Vector giving bound on derivative of metabolic energy wrt potential, such that dHdv_bound > abs(dHdv), in pA, for each neuron in the network, based on its type
    VectorXd                 I_spike;                // Vector giving spike current, in pA, for each neuron in the network, based on its type
    VectorXd                 spike_potential;        // Vector giving magnitude of each spike, in mV, for each neuron in the network, based on its type
    VectorXd                 resting_potential;      // Vector giving resting potential, in mV, for each neuron in the network, based on its type
    VectorXd                 threshold;              // Vector giving spike threshold, in mV, for each neuron in the network, based on its type
    VectorXd                 leak_conductance;       // Vector giving conductance controlling the leak current, I_leak = leak_conductance (resting_potential - v), in nS, for each neuron in the network, based on its type
    MatrixXd                 tau_components;         // n_neurons x 3 matrix giving the temporal modulation time (ms) bias, step, and cutoff for each neuron in the network, based on its type
    VectorXd                 spike_recovery_rate;    // Vector giving the number of spikes which can be "cleared" per ms, for each neuron in the network, based on its type
    VectorXd                 tau_STD_recovery;       // Vector giving the rate at which each cell in the network recovers from short-term depression (STD); units match spike_recovery_rate (spikes per ms), and STD requires tau_STD_recovery < spike_recovery_rate
    VectorXd                 transmission_velocity;  // Vector giving the transmission delay (ms) for each neuron in the network, based on its type
    CharacterVector          neuron_type_name;       // Vector giving the type of each neuron in the network, as a string
    std::vector<int>         neuron_type_num;        // Vector giving the type of each neuron in the network, as an integer index
    std::vector<int>         node_range_ends;        // Vector giving the ending neuron index for each node in the network
    std::vector<MatrixXi>    edge_types;             // Vector of integer matrices giving all transconductance matrix coordinates for each edge type, in nS
    CharacterVector          edge_type_names = {"local connections"};  // Names of elements in edge_types
    
    // Data fields 
    double   sim_dt;                                 // Time step for simulation, in ms
    MatrixXd sim_traces;                             // NxT matrix of doubles, each column giving the simulated membrane potential of a neuron, each row giving a time-step in the simulation
    VectorXd spike_counts;                           // Vector of length n_neurons, giving the number of spikes for each neuron in the network during a simulation
    
    // Functions *********************************
    
    // Constructor and Destructor
    network();
    virtual ~network() {};
    
    // Copy method 
    network(const network& other) = default;
    
    // Member functions for adjusting settings
    void set_network_structure(
      CharacterVector nrn_types,
      CharacterVector lyr_names,
      int             n_lyr,
      int             n_cls,
      int             n_pch,
      double          lyr_height,
      double          cls_diameter,
      double          seg_length, 
      double          lyr_separation_factor,
      double          cls_separation_factor,
      double          pch_separation_factor,
      double          synaptic_neighborhood_radius,
      IntegerMatrix   nrn_per_node,
      List            local_conductance
    );
    
    // Expand cell type parameters into per-neuron vectors (called once after n_neurons is known)
    void resize_neuron_params();
    
    // Member functions for building network
    void make_arbor_branch(
      int             cell_idx,                           // Number of neuron for which to make processes
      bool            is_axon,                            // Whether to make axon (true) or dendrite (false)
      int             parent_branch_idx = -1,             // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
      const Vector3d& attractor_point = {0.0, 0.0, 0.0}
    );
    void make_arbor(
      int                          cell_idx,                   // Number of neuron for which to make processes
      int                          n_branches,                 // Expected number of branches, including the main process 
      bool                         is_axon,                    // Whether to make axon (true) or dendrite (false)
      int                          parent_branch_idx = -1,     // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
      const std::vector<Vector3d>& attractor_points = {{0.0, 0.0, 0.0}}
    );
    double find_synapse( // ... and return its transduction while setting a number of other important synaptic properties
      int    idx_pre,
      int    idx_post,
      double val_pre,
      double transductance_bias
    );
    double compute_expected_node_radius();
    void make_local_nodes(); 
    void apply_circuit_motif(const motif& cmot, bool verbose = true);
    
    // Member functions for fetching data 
    List fetch_network_components(bool include_arbors = false) const;
    NumericMatrix fetch_sim_traces_R() const;
    NumericVector fetch_spike_counts_R() const;
    
    // Member functions for analysis and simulation 
    MatrixXi find_pairwise_lags_by_axon(
      double dt                                  // time step length, in ms
    );
    void SGT(
      const NumericMatrix& stimulus_current_R,   // matrix of stimulus currents, in pA, n_neurons x n_steps
      double dt,                                 // time step length, in ms
      double initial_potential                   // start all neurons with this membrane potential
    );
    
  };

#endif
