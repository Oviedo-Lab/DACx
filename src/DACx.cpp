
// DACx.cpp

// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <nlopt.hpp>
#include <random>
#include <algorithm>
#include <fstream>
using namespace Rcpp;
using namespace Eigen;

/*
 * Sections: 
 * - Templates
 * - Type and class definitions
 * - Helper functions
 * - Growth-transform helper functions
 * - Cell types and related functions
 * - Network (and related) member function implementations
 */

// Define aliases for convenience
using Pnt3 = std::vector<Vector3d>;
using Vint = std::vector<int>;
using Vdbl = std::vector<double>;
using Vboo = std::vector<bool>;
using Vstr = std::vector<std::string>; 
using VCV  = std::vector<CharacterVector>;

/*
 * ***********************************************************************************
 * Templates
 */

// Return index of first element equal to val, or -1 if not found.
// Templated to work on CharacterVector, std::vector<std::string>, std::vector<int>, etc.
template<typename Vec, typename Val>
int find_first(
    const Vec& vec, 
    const Val& val
  ) {
    for (int i = 0; i < (int)vec.size(); ++i) { if (vec[i] == val) return i; }
    return -1;
  }

// find_first_by: like find_first but uses a lambda accessor rather than operator[].
template<typename Accessor, typename Val>
int find_first_by(
    int        n, 
    Accessor   accessor, 
    const Val& val
  ) {
    for (int i = 0; i < n; ++i) { if (accessor(i) == val) return i; }
    return -1;
  }

/*
 * ***********************************************************************************
 * Type and class definitions
 */

// Cell types used in the network
struct cell_type {
    // ID information
    std::string type_name;
    // Membrane kinetics
    double      tau_fast;                // time constant (ms) of the fast sodium (Na+) current: time to flow in.[1]
    double      tau_slow;                // time constant (ms) of the slow calcium (Ca2+) current: time to pump out.[1]
    double      tau_Vs;                  // time constant (ms) for restoring pre-synaptic vesicles, i.e., recovery from short-term depression (STD)
    double      dCdr;                    // slow-current molecule (Ca2+) influx as concentration per spike (concentration/spike).[5]
    double      dVdr;                    // utilization ratio (concentration/spike) of vesicles per spike
    double      max_spike_rate;          // constant (spikes/ms) controlling estimation of spike rate and its max value
    double      g_leak;                  // conductance (nS) controlling the leak current.[2]
    // Intercell transmission
    double      spike_velocity;          // transmission velocity (micron/ms) along axon
    double      spine_density;           // scale controlling percentage of dendrite nodes with spines: zero means none, one means all.
    std::string axon_target;             // "spine", "dendrite_shaft", "soma", and "axon_shaft"
    // Spiking
    double      I_spike;                 // spike current (pA)
    double      dHdv_bound;              // Scale factor giving the bound on derivative of metabolic energy wrt potential.[3]
    double      v_spike;                 // peak potential during a spike spike (mV)
    double      tau_spike;               // time spike activates synapse (ms) before decay
    double      v_threshold;             // spike v_threshold (mV)
    Vdbl        v_eq;                    // induced potential at which no current naturally flows across membrane (mV), given the neurotransmitters of each possible pre-synaptic cell type
    // Membrane characteristics
    double      v_rest;                  // resting potential (mV)
    double      v_bound;                 // multiplier on abs(v_rest) giving the membrane potential barrier (mirrors dHdv_bound).[4]
    Vdbl        g_syn;                   // conductance of cell's synapses to neurotransmitters of each possible pre-synaptic cell type
    Vdbl        tau_syn;                 // decay time constant (ms) of the post-synaptic current, given the neurotransmitter of each possible pre-synaptic cell type
    // Neurite structure
    int         axon_branch_count;       // expected number of axon branches
    int         dendrite_branch_count;   // expected number of dendrite branches
    double      branch_independence;     // scale controlling branch independence: zero means all branches connect to soma from single segment, one means all branches connect directly to soma
    double      branch_spread;           // scale controlling branch spread: zero means no tendency to extend away from soma, one means straight line away from soma
    std::string apical_target_layer;     // layer to which apical dendrite is expected to grow, if any; if none, "none"
    // Dendritic computing 
    double      dendrite_velocity;       // transmission velocity (microns/ms) along dendrite
    double      Ta;                      // scalar [0, 1] giving the strength of the supra-threshold, sub-additive effect on synaptic integration across dendrites
    double      tA;                      // scalar [0, 1] giving the strength of the sub-threshold, supra-additive effect on synaptic integration across dendrites
    
    /*
     * Comments: 
     * [1] Na+ and Ca2+ influx are inward, i.e. negative under the outward-positive convention. 
     * [2] I_leak = g_leak * (v - v_rest) (outward-positive)
     * [3] dHdv_bound * I_spike > abs(dHdv) 
     * [4] v_bound * abs(v_rest)
     * [5] "dC/dr" = derivative of Ca2+ concentration w.r.t. spike rate
     */
    
  };

// Meso-scale axonal and dendritic projections
struct Prj {
    std::string pre_type;
    std::string pre_layer;
    std::string post_type;
    std::string post_layer;
    int         hem_shift  = 0;     // 0 = same hemisphere; 1 = contralateral
    bool        via_apical = false; // Does the pre_type cell connect to the post_type cell via the post_type's apical dendrite? 
  };

// Node-tree description of neurite arbors, one structure instance per cell
struct cell_arbors {
    Vint              arbor_id;     // arbor_id[i]         = unique id for arbor i
    Vint              axon_idx;     // indices of the arbors which are axons 
    Vint              motifs = {1}; // motifs[i]           = 1 if motif i was used in construction of the arbor, 0 otherwise; initializes with {1} for local connections
    Vboo              axon;         // axon[i]             = whether arbor i is axon (true) or dendrite (false)
    Vboo              apical;       // apical[i]           = whether arbor i is an apical dendrite (true) or not (false)
    std::vector<Pnt3> coordinates;  // coordinates[i][j]   = coordinates z, y, x of neurite node j on arbor i (including soma coordinates for j = 0)
    std::vector<Vstr> node_type;    // node_type[i][j]     = "soma", "dendrite_shaft", "axon_shaft", or "spine" for node j in arbor i
    std::vector<Vint> parents;      // parents[i][j]       = the node number (idx in coordinates) of the parent of node j in arbor i, with -1 for the soma
    std::vector<Vint> leafs;        // leafs[i][j]         = 1 if node j in arbor i is a leaf, 0 otherwise
    std::vector<Vint> synapses;     // synapses[i][j]      = number of synapses on node j in arbor i, with 0 for non-synaptic nodes
  };

// Network structure 
struct ntw_struct {
    VCV      hsl_names; 
    Vint     n = {1, 0, 1, 1, 1}; 
    Vdbl     sep_factors;        
    double   lyr_height;             // sd of the normal distribution for local y coordinates of the neurons
    double   cls_diameter;           // sd of the normal distribution for local x coordinates of the neurons
    double   seg_length;             // expected length of segments in process arbors (microns)
    double   synaptic_neighborhood;  // radius of synapse-forming neighborhood; axon-dendrite node pairs within this distance initialize as synapses
    double   expected_node_radius; 
    MatrixXi nrn_per_node;
    /*
     * hsl_names = names of: 
     *      hem_names: hemispheres                     (e.g., left and right)
     *      sub_names: subcortical layers              (e.g., thalamus)
     *      lyr_names: cortical layers                 (e.g., L1, L2, L3, etc.)
     *      
     * n = number of: 
     *      n_hem: hemispheres                         (must be 1 or 2)
     *      n_sub: subcortical (e.g., thalamic) layers (can be zero)
     *      n_lyr: cortical layers                     (must be > 0)
     *      n_cls: laminar and subcortical columns     (must be > 0)
     *      n_pch: laminar and subcortical patches     (rows of columns, i.e., n_lyr x n_cls sheets)
     *      
     * sep_factors = factors by which to multiply: 
     *      hem_separation_factor: column diameter to get distance between hemispheres
     *      sub_separation_factor: layer height    to get distance to subcortical layers
     *      lyr_separation_factor: layer height    to get distance between layers
     *      cls_separation_factor: column diameter to get distance between columns
     *      pch_separation_factor: column diameter to get distance between patches (rows of columns)
     * 
     * nrn_per_node = mean number of neurons in each cortical and subcortical layer (rows) by type (columns). 
     *      Total number of rows must equal either the sum of cortical and subcortical layers, or (if n_hem = 2) 2x this sum. 
     *      If n_hem = 2 and the number of rows is only the sum of cortical and subcortical layers, the rows will be reused in both hemispheres.
     */
  }; 

// per-neuron cell-type based GT parameters
struct per_nrn_params {
    Vint     neuron_type_num;        // vector giving the type of each neuron in the network, as an integer index
    ArrayXd  v_bound;                // vector giving potential bound, in mV, for each neuron
    ArrayXd  dHdv_bound;             // vector giving scale factor (w.r.t. I_spike) of bound on derivative of metabolic energy wrt potential, for each neuron
    ArrayXd  I_spike;                // vector giving spike current, in pA, for each neuron
    ArrayXd  v_spike;                // vector giving magnitude of each spike, in mV, for each neuron
    ArrayXd  tau_spike;              // vector giving time a spike activates a synapse, in ms, for each neuron
    ArrayXd  v_rest;                 // vector giving resting potential, in mV, for each neuron
    ArrayXd  v_threshold;            // vector giving spike v_threshold, in mV, for each neuron
    ArrayXd  g_leak;                 // vector giving leak conductance, in nS, for each neuron
    ArrayXd  max_spike_rate;         // vector giving the number of spikes which can be "cleared" per ms, for each neuron
    ArrayXd  tau_fast; 
    ArrayXd  tau_slow; 
    ArrayXd  tau_Vs;                 // vector giving STD recovery time constant, in ms/spike, for each neuron
    ArrayXd  dCdr; 
    ArrayXd  dVdr; 
    ArrayXd  spike_velocity;         // vector giving the transmission velocity (microns/ms) for each neuron
    ArrayXd  dendrite_velocity;      // vector giving the speed of signals traveling along dendrites (microns/ms) for each neuron
    ArrayXd  Ta;                     // vector giving the strength of the supra-threshold, sub-additive effect on synaptic integration across dendrites
    ArrayXd  tA;                     // vector giving the strength of the sub-threshold, supra-additive effect on synaptic integration across dendrites
    ArrayXXd v_eq;                   // array giving the equilibrium potential (mV) for each post-synaptic cell (rows), given each pre-synaptic cell's type (columns)
    ArrayXXd tau_syn;                // array giving the PSC decay time constant (ms) for each post-synaptic cell (rows), given each pre-synaptic cell's type (columns)
    ArrayXXd pre_syn_travel;         // array giving the distance (microns) along each pre-synaptic cell's axons (rows) between the post-synaptic cell's synapse (columns) and the post-synaptic soma
    ArrayXXd post_syn_travel;        // array giving the distance (microns) along each post-synaptic cell's dendrites (rows) between the pre-synaptic cell's synapse (columns) and the post-synaptic soma
  };

// Network edges (connections), per motif
struct ntw_edges {
    std::vector<ArrayXXd> g_syn;     // vector of square arrays, each giving the g_syn (nS) between each neuron in the network, one array per motif
    std::vector<MatrixXi> type;      // vector of integer matrices giving all g_syn matrix coordinates for each edge type
    CharacterVector       motif_name = {"local connections"};
  }; 

// Network node and cell coordinates
struct ntw_coords {
    MatrixXd node_spatial;  // Mx3 matrix giving the (z,y,x) spatial coordinates of each node in the network
    MatrixXd spatial;       // n_neurons x 3 matrix giving the (z,y,x) spatial coordinates of each neuron in the network
    MatrixXi node;          // n_neurons x 6 matrix giving the node coordinates of each neuron in the network
    /*
     * Node indexes (for 0-4) match ntw_struct: 
     *   (0) hemisphere                         (must be 1 or 2)
     *   (1) subcortical (e.g., thalamic) layer (can be zero)
     *   (2) cortical layer                     (must be > 0) ... 3 -> 2
     *   (3) laminar and subcortical columns    (must be > 0) ... 4 -> 3
     *   (4) laminar and subcortical patches    ... 2 -> 4
     *   (5) apical layer 
     */
   
  }; 

// Cortical projection motif
class motif {
  
  /*
   * Motifs are recipes for building internode projections within a neural network. They are 
   *   "columnar", in the sense that they are defined within and repeated across cortical columns. 
   */
  
  public:
    
    // Variables *********************************
    
    std::string      motif_name = "not_provided";  // name of motif
    std::vector<Prj> projections;                  // list of projection descriptions, for projections defining the motif
    Vint             max_col_shift_up;             // maximum number of columns to shift up when applying motif
    Vint             max_col_shift_down;           // maximum number of columns to shift down when applying motif
    Vint             max_pch_shift_up;             // maximum number of patches to shift up when applying motif
    Vint             max_pch_shift_down;           // maximum number of patches to shift down when applying motif
    Vdbl             projection_fraction;          // fraction of eligible pre-neurons that send axons for each projection [0.0, 1.0]; clipped to [1/n_pre, 1.0]
    int              n_projections = 0;            // number of projections in motif
    int              hemi = -1;                    // hemisphere to which motif is to be applied; -1 = all, 0 = the first (i.e., left), 1 = the second (i.e., right).
    
    // Functions *********************************
    
    motif(const std::string motif_name = "not_provided", int hemi = -1);
    virtual ~motif() {};
    motif(const motif& other) = default;
    
    void load_projection(
      const Prj& proj,
      int        max_col_up,
      int        max_col_down,
      int        max_pch_up,
      int        max_pch_down,
      double     pre_neuron_fraction
    );
    
  };

// Cortical network model
class network {
  
  // units of ms (time), mV (potential), pA (current), nS (conductance), micron (distance)
  
  public:
    
    // Variables *********************************
    
    // Set up random number generator and standard distributions
    std::mt19937                           cpp_rng;
    std::uniform_real_distribution<double> unif{0.0, 1.0}; 
    std::normal_distribution<double>       norm{0.0, 1.0};
    
    // Network structure
    std::vector<cell_type>        neuron_types;      // types of neurons in network, e.g., "pyramidal", "PV", "SST", "VIP"
    Vstr                          neuron_type_names; // names of neuron types, indexed by local type index
    ntw_struct                    ntw;               // network structure
    
    // Network components 
    int                           n_neurons = 0;     // total number of neurons in the network
    std::vector<cell_arbors>      arbors;            // vector of length n_neurons
    ntw_edges                     edges;             // structure holding network edges (connections), per motif
    ntw_coords                    coords;            // structure holding coordinates for cells and nodes
    per_nrn_params                per_nrn;           // structure giving cell type GT simulation values per neuron
    Vint                          node_range_ends;   // vector giving the ending neuron index for each node in the network
    std::unordered_map<int, Vint> apical_node_cells; // maps node index -> cell indices whose apical dendrites target that node
    
    // Data fields 
    double                        sim_dt;            // Size of simulation timestep, in ms
    ArrayXXd                      v_traces;          // traces from BGT simulation
    ArrayXd                       spike_counts;      // Vector of length n_neurons giving spike counts during a BGT simulation
    
    // Functions *********************************
    
    // Constructor and destructor
    network();
    virtual ~network() {};
    network(const network& other) = default;
    
    // Network structure
    int node_idx_lookup(
      int base,     // distinguishes cortical layers (base 0) and subcortical layers (base = n_cortical_layers)
      int p,        // patch index
      int l,        // layer index 
      int c,        // column index
      int h,        // hemisphere index
      int n_layers  // Number of layers within this region 
    );
    void set_neuron_params();
    void set_network_structure(
      CharacterVector nrn_types,
      List            hsl_names,
      IntegerVector   n, 
      NumericVector   sep_factor, 
      double          lyr_height,
      double          cls_diameter,
      double          seg_length, 
      double          synaptic_neighborhood,
      IntegerMatrix   nrn_per_node
    );
    
    // Build network
    void make_local_nodes(); 
    void make_arbor_branch(
          double          max_bias, 
          int             cell_idx, 
          bool            is_axon, 
          bool            is_apical, 
          int             parent_branch_idx = -1, 
          const Vector3d& attractor_point  = {0.0, 0.0, 0.0},                
          bool            hit_attractor = false
        );
    void make_arbor(
          int             n_branches,         
          int             cell_idx, 
          bool            is_axon, 
          bool            is_apical, 
          int             parent_branch_idx = -1, 
          const Pnt3&     attractor_points = {{0.0, 0.0, 0.0}}, 
          bool            hit_attractor = false
        );
    void apply_circuit_motif(
          const motif&    cmot
        );
    double find_synapse(
          int             idx_pre, 
          int             idx_post, 
          bool            via_apical
        );
    
    // BGT simulations 
    double integrate_along_arbor_to_soma(int node_idx, int arbor_idx, int cell_idx);
    void   BGT(const NumericMatrix& I_stim_R, double dt, double v_initial);
    
    // Fetch for R
    List   fetch_network_components(bool include_arbors = false) const;
    List   fetch_sim_results() const; 
    
  };

/*
 * ***********************************************************************************
 * Helper functions
 */

// Matrix conversions and overloads
ArrayXXd to_eMat(
    const NumericMatrix& X
  ) {
    int Xnrow = X.nrow();
    int Xncol = X.ncol();
    ArrayXXd M(Xnrow, Xncol);
    for (int j = 0; j < Xncol; ++j) {
      for (int i = 0; i < Xnrow; ++i) {
        M(i, j) = X(i, j);
      }
    }
    return M;
  }
MatrixXi to_eiMat(
    const IntegerMatrix& X
  ) {
    int Xnrow = X.nrow();
    int Xncol = X.ncol();
    MatrixXi M = MatrixXi(Xnrow, Xncol);
    for (int j = 0; j < Xncol; ++j) {
      for (int i = 0; i < Xnrow; ++i) {
        M(i, j) = X(i, j);
      }
    }
    return M;
  }
NumericMatrix to_NumMat(
    const MatrixXd& M
  ) {
    int M_nrow = M.rows();
    int M_ncol = M.cols();
    NumericMatrix X(M_nrow, M_ncol);
    for (int j = 0; j < M_ncol; ++j) {
      for (int i = 0; i < M_nrow; ++i) {
        X(i, j) = M(i, j);
      }
    }
    return X;
  }
NumericMatrix to_NumMat(
    const ArrayXXd& M
  ) {
    int M_nrow = M.rows();
    int M_ncol = M.cols();
    NumericMatrix X(M_nrow, M_ncol);
    for (int j = 0; j < M_ncol; ++j) {
      for (int i = 0; i < M_nrow; ++i) {
        X(i, j) = M(i, j);
      }
    }
    return X;
  }
NumericMatrix to_NumMat(
    const MatrixXi& M
  ) {
    int M_nrow = M.rows();
    int M_ncol = M.cols();
    NumericMatrix X(M_nrow, M_ncol);
    for (int j = 0; j < M_ncol; ++j) {
      for (int i = 0; i < M_nrow; ++i) {
        X(i, j) = M(i, j);
      }
    }
    return X;
  }

/*
 * ***********************************************************************************
 * Growth-transform helper functions
 */

// Membrane potential barrier function
ArrayXd v_barrier(
    const ArrayXd& v_input,           // Column vector of membrane potentials for a network of neurons at one time step
    const ArrayXd& v_threshold,       // Spike v_threshold in mV, for each neuron in network
    const ArrayXd& I_out              // Spike current in pA, for each neuron in network
  ) {
    ArrayXd output(v_input.size());
    for (int i = 0; i < v_input.size(); ++i) {
      output[i] = (v_input[i] < v_threshold[i]) ? 0.0 : I_out[i];
    }
    return output;
  } 

// Create lagged input trace matrix to simulate transmission delays
ArrayXXd lagged_traces(
    int   n,                          // Current step index
    const ArrayXXi& lag,              // Pairwise lags, in time steps, for signal to get from neuron (row) i to j
    const ArrayXXd& v                 // Input traces
  ) {
    const int n_neuron = v.rows();
    ArrayXXd v_lagged(n_neuron, n_neuron);
    
    for (int j = 0; j < n_neuron; ++j) {
      for (int i = 0; i < n_neuron; ++i) {
        int time_index = n - lag(i, j);
        if (time_index < 0) time_index = 0; 
        v_lagged(i, j) = v(i, time_index); // Neuron i's input as seen by neuron j
      }
    }
    return v_lagged;
  }

// Look up each pre-synaptic neuron's last_spike value from a circular history buffer
// ... ls_lagged(i,j) = last_spike of pre-syn j, as seen by post-syn i, accounting for conduction lag
ArrayXXi lagged_last_spike(
    int              t,           // Current step index
    const ArrayXXi&  lag,         // pre_syn_lags: rows = pre-syn, cols = post-syn
    const ArrayXXi&  ls_history,  // Circular buffer of last_spike; cols indexed by t % buffer_size
    int              buffer_size
  ) {
    const int n_neuron = ls_history.rows();
    ArrayXXi ls_lagged(n_neuron, n_neuron);
    for (int i = 0; i < n_neuron; ++i) {       // post-syn i
      for (int j = 0; j < n_neuron; ++j) {     // pre-syn j
        int raw_t = t - lag(j, i);
        int col   = raw_t < 0 ? 0 : raw_t % buffer_size;
        ls_lagged(i, j) = ls_history(j, col);
      }
    }
    return ls_lagged;
  }

// Derivative of intracellular slow-current molecule concentrations
ArrayXd dCdt(
    const ArrayXd& Ca,                  // Vector of intracellular slow-current molecule (e.g., calcium) concentrations, per cell
    const ArrayXd& recent_spike_count,  // Vector of counts of recent spikes, per cell; proxy for spike rate (spikes/ms)
    const ArrayXd& dCdr,                // vector giving the slow-current molecule (e.g., Ca2+) influx as concentration per spike (concentration/spike)
    const ArrayXd& tau_slow             // vector giving time constant for clearing calcium, per cell (ms)
  ) {
    return dCdr * recent_spike_count - Ca / tau_slow;
    // Returns concentration/ms
  }

// Derivative of synaptic vesicle concentrations (synaptic depression), from Schiff & Reyes 2012 (https://doi.org/10.1152/jn.00208.2011) 
ArrayXd dVdt(
    const ArrayXd& Vs,                  // Vector of synaptic vesicle concentrations, per cell (ratio, [0,1])
    const ArrayXd& recent_spike_count,  // Vector of counts of recent spikes, per cell; proxy for spike rate (spikes/ms)
    const ArrayXd& dVdr,                // vector giving the utilization ratio (concentration/spike) of vesicles per spike
    const ArrayXd& tau_Vs               // vector giving time constant for recovery from depression, per cell (ms)
  ) {
    return (1.0 - Vs) / tau_Vs - Vs * recent_spike_count * dVdr;
    // Returns concentration/ms
  }

/*
 * ***********************************************************************************
 * Cell types and related functions
 */

// Cell type registry: Meyers singleton pattern
Vstr& get_all_type_names() {
    static Vstr all_type_names = {
      // Excitatory
      "pyramidal", "callosal_pyramidal", "pyramidal_L6", "spiny_stellate", "thalmacortical",
      // Inhibitory
      "neurogliaform_cell", "PV", "callosal_PV", "SST", "VIP",
      // Generic template
      "neuron"
    };
    return all_type_names;
    
    /*
     * Canonical, ordered list of all known global cell type names.
     *   Single source of truth: used both to build the default registry and to
     *   parse user-supplied named g_syn lists. Order must stay
     *   consistent with the hardcoded conductance matrix g below and with any code
     *   that relies on positional indexing into this list.
     *   
     * This list is *not* fixed: new cell types (added via build_cell_type_from_list(),
     *   through R's modify.cell.type()) are appended to it as they are encountered, via 
     *   register_type_name() below. This allows user-defined cell types to be referenced by 
     *   name (e.g. in g_syn, v_eq, or tau_syn lists), including forward references to cell 
     *   types that will be created later in the same session.
     *   
     * register_type_name() is defined just below get_cell_types(), since it needs to extend 
     *   the per-type vectors of every already-registered cell type (see below).
     *
     */
    
  }

// Function to construct default cell types
static std::unordered_map<std::string, cell_type> make_default_cell_types() {
    std::unordered_map<std::string, cell_type> ct_map;
   
    // Pre-define the names of all known cell types (order must be consistent)
    const Vstr& all_type_names = get_all_type_names();
    const int   n_all_types    = all_type_names.size();
    
    // Set set-type dependent vectors: forms matrix [post_type][pre_type], pre_type Indexed by position in all_type_names vector
    // ... synaptic conductance
    Vdbl g_syn(n_all_types, 0.1);                                   // Default 0.1 nS for all connections
    // ... cell type valence 
    Vdbl v_eq(n_all_types, 0.0);                                    // Default to excitatory, 0 mV
    for (int i = 5; i < n_all_types - 1; ++i) { v_eq[i] = - 70; }   // Set pre-synaptic inhibitory cells to -70mV
    // ... synaptic (post-synaptic current) decay time constants per pre-synaptic type
    Vdbl tau_syn(n_all_types, 2.0);                                 // Default excitatory (AMPA-like), 2 ms
    for (int i = 5; i < n_all_types - 1; ++i) { tau_syn[i] = 6.0; } // Set pre-synaptic inhibitory cells (GABA_A-like) to 6 ms
   
    // Default shared values
    double      tau_fast                 = 5.0;   // ms
    double      tau_slow                 = 60.0;  // ms
    double      tau_Vs                   = 100.0; // ms
    double      dCdr                     = 0.01;  // concentration/spike; Default is no bursting; increase to induce bursting (or lower tau_slow)
    double      dVdr                     = 0.05;  // concentration/spike
    double      max_spike_rate           = 0.1;   // spikes/ms
    double      g_leak                   = 10.0;  // nS
    double      spike_velocity           = 1e3;   // microns/ms, 1e3 = 1 m/s
    double      dendrite_velocity        = 1e4;   // microns/ms, 1e4 = 10 m/s
    double      Ta                       = 0.0;
    double      tA                       = 0.0; 
    double      spine_density            = 0.0;
    std::string axon_target              = "dendrite_shaft";
    double      I_spike                  = 1e3;   // pA
    double      dHdv_bound               = 1.05; 
    double      v_spike                  = 35.0;  // mV
    double      tau_spike                = 1.0;   // ms
    double      v_threshold              = -55.0; // mV
    double      v_rest                   = -70.0; // mV
    double      v_bound                  = 1.15;  // multiplier on abs(v_rest)
    int         axon_branch_count        = 10;
    int         dendrite_branch_count    = 10;
    double      branch_independence      = 0.5;
    double      branch_spread            = 0.5;
    std::string apical_target_layer      = "none";
   
    // Excitatory cells
    ct_map["pyramidal"] = cell_type{ // Slow responders, 10-50 ms, No bursting 
      "pyramidal",
      tau_fast, tau_slow, tau_Vs, dCdr, dVdr, max_spike_rate, g_leak,
      spike_velocity, 0.5, "spine",
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      axon_branch_count, dendrite_branch_count,
      branch_independence * 0.5, branch_spread * 0.5, // Reduced branching
      "L1", // Harris2013a, for cells in L2, L3, and L5
      dendrite_velocity, Ta, tA
    };
    ct_map["callosal_pyramidal"] = cell_type{ // Slow responders, 10-50 ms, No bursting 
      "callosal_pyramidal", 
      tau_fast, tau_slow, tau_Vs, dCdr, dVdr, max_spike_rate, g_leak,
      spike_velocity, 0.5, "spine",
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn, 
      axon_branch_count * 2, dendrite_branch_count,
      branch_independence * 0.5, branch_spread * 0.5, // Reduced branching
      "L1", // Harris2013a, for cells in L2, L3, and L5
      dendrite_velocity, Ta, tA
    };
    ct_map["pyramidal_L6"] = cell_type{ // No bursting 
      "pyramidal_L6", 
      tau_fast, tau_slow, tau_Vs, dCdr, dVdr, max_spike_rate, g_leak,
      spike_velocity, 0.5, "spine",
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn, 
      axon_branch_count, dendrite_branch_count,
      branch_independence * 0.5, branch_spread * 0.5, // Reduced branching
      "L4", // Harris2013a
      dendrite_velocity, Ta, tA
    };
    ct_map["spiny_stellate"] = cell_type{ // No bursting 
      "spiny_stellate",
      tau_fast, tau_slow, tau_Vs, dCdr, dVdr, max_spike_rate, g_leak,
      spike_velocity, 0.5, "spine",
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      axon_branch_count, dendrite_branch_count,
      branch_independence * 1.5, branch_spread * 1.5, // Increased branching
      apical_target_layer,
      dendrite_velocity, Ta, tA
    };
    ct_map["thalmacortical"] = cell_type{ // No bursting, strong STD
      "thalmacortical",
      tau_fast, tau_slow, tau_Vs * 1.5, dCdr, dVdr * 1.5, max_spike_rate, g_leak,
      spike_velocity, 0.5, "spine",
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      static_cast<int>(std::round(axon_branch_count * 0.5)), dendrite_branch_count,
      0.1, 0.9,
      apical_target_layer,
      dendrite_velocity, Ta, tA
    };
    
    // Inhibitory cells
    ct_map["neurogliaform_cell"] = cell_type{ // bursting, Slower transmission, slow decay (i.e., large tau_slow, see Huang2024a p. 190)
      "neurogliaform_cell",
      tau_fast, tau_slow * 2.0, tau_Vs, dCdr * 3.5, dVdr, max_spike_rate, g_leak,
      spike_velocity * 0.5, spine_density, axon_target, 
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      axon_branch_count, dendrite_branch_count,
      branch_independence * 1.5, branch_spread * 1.5, // Increased branching
      apical_target_layer,
      dendrite_velocity, Ta, tA
    };
    ct_map["PV"] = cell_type{ // Faster responders, ~5 ms; No bursting
      "PV", 
      tau_fast * 0.5, tau_slow, tau_Vs, dCdr, dVdr, max_spike_rate * 3.0, g_leak * 2.0,
      spike_velocity, spine_density, "soma",
      I_spike, dHdv_bound, v_spike, 0.3, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      axon_branch_count, dendrite_branch_count,
      branch_independence * 1.25, branch_spread * 1.25, // Increased branching
      apical_target_layer,
      dendrite_velocity, Ta, tA
    };
    ct_map["callosal_PV"] = cell_type{ // Faster responders, ~5 ms; No bursting
      "callosal_PV",  
      tau_fast * 0.5, tau_slow, tau_Vs, dCdr, dVdr, max_spike_rate * 3.0, g_leak * 2.0,
      spike_velocity, spine_density, "soma",
      I_spike, dHdv_bound, v_spike, 0.3, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      axon_branch_count * 2, dendrite_branch_count,
      branch_independence * 0.5, branch_spread * 0.5, // Reduced branching
      apical_target_layer,
      dendrite_velocity, Ta, tA
    };
    ct_map["SST"] = cell_type{ // Slower responders, 10-30 ms
      "SST",  
      tau_fast, tau_slow, tau_Vs, dCdr * 3.5, dVdr, max_spike_rate, g_leak,
      spike_velocity, spine_density, axon_target,
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      axon_branch_count, dendrite_branch_count,
      branch_independence * 1.5, branch_spread * 1.5, // Increased branching
      apical_target_layer,
      dendrite_velocity, Ta, tA
    };
    ct_map["VIP"] = cell_type{ // Slow responders, 15-40 ms
      "VIP", 
      tau_fast, tau_slow, tau_Vs, dCdr * 3.5, dVdr, max_spike_rate, g_leak,
      spike_velocity, spine_density, axon_target,
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      axon_branch_count, dendrite_branch_count,
      branch_independence * 1.25, branch_spread * 1.25, // Increased branching
      apical_target_layer,
      dendrite_velocity, Ta, tA
    };
    
    // Generic 
    ct_map["neuron"] = cell_type{ 
      "neuron", 
      tau_fast, tau_slow, tau_Vs, dCdr, dVdr, max_spike_rate, g_leak,
      spike_velocity, spine_density, axon_target,
      I_spike, dHdv_bound, v_spike, tau_spike, v_threshold, v_eq,
      v_rest, v_bound, g_syn, tau_syn,
      axon_branch_count, dendrite_branch_count,
      branch_independence, branch_spread,
      apical_target_layer,
      dendrite_velocity, Ta, tA
    };
   
    return ct_map;
  }

// Meyers singleton: initialized once on first call, never reset by .onLoad
std::unordered_map<std::string, cell_type>& get_cell_types() {
    static std::unordered_map<std::string, cell_type> cell_types = make_default_cell_types();
    return cell_types;
    
    /*
     * Access cell type registry via get_cell_types() everywhere; do NOT declare a bare global.
     *   Defaults are constructed once in make_default_cell_types() on first call.
     *   
     * To use or modify cell types:
     *   const auto& ct = get_cell_types().at("PV");
     *   double cutoff = ct.tau_slow;
     *   get_cell_types()["PV"].dCdr = 0.03;
     */
    
  }

// Register cell type names
void register_type_name(
    const std::string& name
  ) {
    // Get all registered names 
    Vstr& all_type_names = get_all_type_names();
    // If this name is already registered, return
    if (std::find(all_type_names.begin(), all_type_names.end(), name) != all_type_names.end()) {
      return;
    }
    // else, add it
    all_type_names.push_back(name);
    // Update type-specific matrices
    for (auto& pair : get_cell_types()) {
      pair.second.g_syn.push_back(0.0);   // default: no synaptic connection (0) 
      pair.second.v_eq.push_back(0.0);    // default: excitatory cell (0) 
      pair.second.tau_syn.push_back(0.0); // default: Instantaneous (boxcar) post-synaptic current with no decay
    }
    
    /*
     * Register a (possibly new) cell type name in the canonical registry. If the 
     *   name is not yet known, it is appended to get_all_type_names(), and 
     *   every already-registered cell type's g_syn, v_eq, and tau_syn vectors 
     *   are extended by one slot (defaulting to 0.0) so that positional 
     *   indexing into those vectors stays consistent with the (now longer) name 
     *   list. A no-op if the name is already registered.
     */
    
  }

// Parse R object intended for type-dependent list
Vdbl parse_pre(
    SEXP X
  ) {
    // Get number of registered cell types 
    int n_all_types = get_all_type_names().size();
    // Initialize vector of doubles to pass out
    Vdbl out(n_all_types, 0.0);
    // Handle single numeric value: replicate across all known cell types
    if (Rf_isNumeric(X) && Rf_length(X) == 1) {
      double scalar_value = as<double>(X);
      out.assign(n_all_types, scalar_value);
    }
    // Handle vector: use as-is
    else if (Rf_isNumeric(X)) {
      out = as<Vdbl>(X);
    }
    // Handle named list for specific post-pre pairs: 
    else if (Rf_isNewList(X)) {
      // Get names from list
      List sc_list = as<List>(X);
      CharacterVector sc_names = sc_list.names();
      
      // For each named entry, update the corresponding pre-type conductance
      for (int i = 0; i < sc_list.size(); ++i) {
        std::string key = as<std::string>(sc_names[i]);
        double value = as<double>(sc_list[i]);
        
        // Register the name if not yet known (supports forward references
        // to cell types that will be created later in the same session),
        // then grow `out` to match if the registry grew.
        register_type_name(key);
        Vstr& all_type_names = get_all_type_names();
        if (static_cast<int>(all_type_names.size()) > static_cast<int>(out.size())) {
          out.resize(all_type_names.size(), 0.0);
        }
        
        // Find matching pre-type index (guaranteed to exist after registration)
        auto it  = std::find(all_type_names.begin(), all_type_names.end(), key);
        int  idx = std::distance(all_type_names.begin(), it);
        out[idx] = value;
      }
    }
    return out; 
  }

// Print known cell types 
// [[Rcpp::export]]
void print_known_celltypes() {
    Rcpp::Rcout << "Known cell types:" << std::endl;
    for (const auto& pair : get_cell_types()) {
      const cell_type& ct = pair.second;
      Rcpp::Rcout << "\nType: "                                            << ct.type_name << std::endl
                  << "  Time constant, fast current (ms): "                << ct.tau_fast << std::endl
                  << "  Time constant, slow current (ms): "                << ct.tau_slow << std::endl
                  << "  STD recovery time constant (spikes/ms): "          << ct.tau_Vs << std::endl
                  << "  Slow current influx (concentration/spike): "       << ct.dCdr << std::endl
                  << "  Vesicle utilization ratio (concentration/spike): " << ct.dVdr << std::endl
                  << "  Spike recovery rate (spikes/ms): "                 << ct.max_spike_rate << std::endl
                  << "  Leak conductance (nS): "                           << ct.g_leak << std::endl
                  << "  Axon transmission velocity (micron/ms): "          << ct.spike_velocity << std::endl
                  << "  Dendrite transmission velocity (micron/ms): "      << ct.dendrite_velocity << std::endl
                  << "  Supra-threshold, sub-additive integration: "       << ct.Ta << std::endl
                  << "  Sub-threshold, supra-additive integration: "       << ct.tA << std::endl
                  << "  Spine density: "                                   << ct.spine_density << std::endl
                  << "  Axon target: "                                     << ct.axon_target << std::endl
                  << "  Spike current (pA): "                              << ct.I_spike << std::endl
                  << "  dHdv bound (* I_spike): "                          << ct.dHdv_bound << std::endl
                  << "  Spike potential (mV): "                            << ct.v_spike << std::endl
                  << "  Spike width (ms): "                                << ct.tau_spike << std::endl
                  << "  Resting potential (mV): "                          << ct.v_rest << std::endl
                  << "  v_bound (* |v_rest|): "                            << ct.v_bound << std::endl
                  << "  Spike threshold (mV): "                            << ct.v_threshold << std::endl
                  << "  Axon branch count: "                               << ct.axon_branch_count << std::endl
                  << "  Dendrite branch count: "                           << ct.dendrite_branch_count << std::endl
                  << "  Branch independence: "                             << ct.branch_independence << std::endl
                  << "  Branch spread: "                                   << ct.branch_spread << std::endl
                  << "  Apical target layer: "                             << ct.apical_target_layer << std::endl;
    }
  }

// Fetch cell type parameters 
// [[Rcpp::export]]
List fetch_cell_type_params(
    std::string type_name
  ) {
    auto it = get_cell_types().find(type_name);
    if (it == get_cell_types().end()) {
      Rcpp::stop("Cell type not found in known cell types");
    }
    const cell_type& ct = it->second;
    List return_list = List::create(
      Named("type_name")             = ct.type_name,
      Named("tau_fast")              = ct.tau_fast,
      Named("tau_slow")              = ct.tau_slow,
      Named("tau_Vs")                = ct.tau_Vs,
      Named("dCdr")                  = ct.dCdr,
      Named("dVdr")                  = ct.dVdr, 
      Named("max_spike_rate")        = ct.max_spike_rate,
      Named("g_leak")                = ct.g_leak,
      Named("spike_velocity")        = ct.spike_velocity,
      Named("spine_density")         = ct.spine_density,
      Named("axon_target")           = ct.axon_target,
      Named("I_spike")               = ct.I_spike,
      Named("dHdv_bound")            = ct.dHdv_bound,
      Named("v_spike")               = ct.v_spike,
      Named("tau_spike")             = ct.tau_spike, 
      Named("v_threshold")           = ct.v_threshold,
      Named("v_rest")                = ct.v_rest,
      Named("axon_branch_count")     = ct.axon_branch_count
    );
    return_list["dendrite_branch_count"] = ct.dendrite_branch_count;
    return_list["branch_independence"]   = ct.branch_independence;
    return_list["branch_spread"]         = ct.branch_spread;
    return_list["apical_target_layer"]   = ct.apical_target_layer;
    return_list["v_bound"]               = ct.v_bound;
    return_list["dendrite_velocity"]     = ct.dendrite_velocity;
    return_list["Ta"]                    = ct.Ta; 
    return_list["tA"]                    = ct.tA; 
    // Extract and convert named list elements
    List ct_tau_syn;
    List ct_g_syn; 
    List ct_v_eq;
    CharacterVector cell_type_names = Rcpp::wrap(get_all_type_names()); 
    int n_cell_types = static_cast<int>(cell_type_names.size()); 
    if (ct.tau_syn.size() != n_cell_types) { Rcpp::stop("Mismatch between length of cell type names and length of tau_syn"); }
    if (ct.g_syn.size()   != n_cell_types) { Rcpp::stop("Mismatch between length of cell type names and length of g_syn"); }
    if (ct.v_eq.size()    != n_cell_types) { Rcpp::stop("Mismatch between length of cell type names and length of v_eq"); }
    for (int i = 0; i < n_cell_types; ++i) {
      String ctn      = cell_type_names[i]; 
      ct_tau_syn[ctn] = ct.tau_syn[i];
      ct_g_syn[ctn]   = ct.g_syn[i]; 
      ct_v_eq[ctn]    = ct.v_eq[i]; 
    }
    return_list["tau_syn"]               = ct_tau_syn;
    return_list["g_syn"]                 = ct_g_syn;
    return_list["v_eq"]                  = ct_v_eq;
    return return_list;
  }

// Modify cell type (or add if new)
// ... unpack a fully-specified named R List into a cell_type struct
// ... NULL-substitution for unspecified fields is handled on the R side
// [[Rcpp::export]]
void build_cell_type_from_list(
    const std::string& type_name, 
    const List& params
  ) {
    cell_type ct;
    ct.type_name             = as<std::string>(params["type_name"]);
    // Register this type's name up front so that self-referencing entries
    // in its own g_syn / v_eq / tau_syn
    // lists (e.g. a type listing its conductance onto itself) resolve
    // correctly below.
    register_type_name(ct.type_name);
    ct.tau_fast              = as<double>(     params["tau_fast"]);
    ct.tau_slow              = as<double>(     params["tau_slow"]);
    ct.tau_Vs                = as<double>(     params["tau_Vs"]);
    ct.dCdr                  = as<double>(     params["dCdr"]);
    ct.dVdr                  = as<double>(     params["dVdr"]); 
    ct.max_spike_rate        = as<double>(     params["max_spike_rate"]);
    ct.spike_velocity        = as<double>(     params["spike_velocity"]);
    ct.spine_density         = as<double>(     params["spine_density"]);
    ct.axon_target           = as<std::string>(params["axon_target"]);
    ct.I_spike               = as<double>(     params["I_spike"]);
    ct.dHdv_bound            = as<double>(     params["dHdv_bound"]);
    ct.v_spike               = as<double>(     params["v_spike"]);
    ct.tau_spike             = as<double>(     params["tau_spike"]);
    ct.v_rest                = as<double>(     params["v_rest"]);
    ct.v_bound               = as<double>(     params["v_bound"]);
    ct.v_threshold           = as<double>(     params["v_threshold"]);
    ct.g_leak                = as<double>(     params["g_leak"]);
    ct.axon_branch_count     = as<int>(        params["axon_branch_count"]);
    ct.dendrite_branch_count = as<int>(        params["dendrite_branch_count"]);
    ct.branch_independence   = as<double>(     params["branch_independence"]);
    ct.branch_spread         = as<double>(     params["branch_spread"]);
    ct.apical_target_layer   = as<std::string>(params["apical_target_layer"]);
    ct.dendrite_velocity     = as<double>(     params["dendrite_velocity"]);
    ct.Ta                    = as<double>(     params["Ta"]);
    ct.tA                    = as<double>(     params["tA"]); 
    
    // Synaptic conductance: if provided, use it; otherwise will be initialized as zero
    if (params.containsElementNamed("g_syn")) {
      SEXP sc_param = params["g_syn"];
      ct.g_syn      = parse_pre(sc_param); 
    }
    
    // Synaptic decay time constant: if provided, use it; otherwise left empty and defaults apply
    if (params.containsElementNamed("tau_syn")) {
      SEXP ts_param = params["tau_syn"];
      ct.tau_syn    = parse_pre(ts_param); 
    }
    
    // Equilibrium potential: if provided, use it; otherwise will be initialized with defaults
    if (params.containsElementNamed("v_eq")) {
      SEXP sc_param = params["v_eq"];
      ct.v_eq       = parse_pre(sc_param); 
    }
    
    // Final checks and return
    if (ct.spine_density < 0.0 || ct.spine_density > 1.0)
      Rcpp::stop("spine_density must be between 0 and 1");
    if (ct.branch_independence < 0.0 || ct.branch_independence > 1.0)
      Rcpp::stop("branch_independence must be between 0 and 1");
    if (ct.branch_spread < 0.0 || ct.branch_spread > 1.0)
      Rcpp::stop("branch_spread must be between 0 and 1");
    
    // Add new cell type 
    get_cell_types()[type_name] = ct;
  }

/*
 * ***********************************************************************************
 * Network (and related) member function implementations
 */

// Find first neighbor 
Vint find_first_neighbor(
    const Pnt3& b_active,            // Branch searching for neighbor
    const Pnt3& b_all,               // Branch being searched
    double      neighborhood_radius,
    bool        skip_origin
  ) {
    double neighborhood_radius_squared = neighborhood_radius * neighborhood_radius;
    int    i_initial                   = 0;
    if (skip_origin) { i_initial = 1; }
    // Run loops backwards for speed, as higher-indexed nodes are more likely to be nearby
    for (int i = static_cast<int>(b_active.size()) - 1; i >= i_initial; --i) {
      for (int j = static_cast<int>(b_all.size()) - 1; j >= 0; --j) {
        double distance = (b_active[i] - b_all[j]).squaredNorm();
        if (distance <= neighborhood_radius_squared) {
          return {i, j}; // Return index of first neighbor found
        }
      }
    }
    return {-1, -1}; // Return -1 if no neighbor is found within the radius
  }

// Constructor, motif
motif::motif(
  const std::string motif_name,
  int               hemi
  ) : motif_name(motif_name), hemi(hemi)
  { 
    // No initialization operations
  }

// Load projection into motif
void motif::load_projection(
    const Prj& proj,
    int        max_col_up,
    int        max_col_down,
    int        max_pch_up,
    int        max_pch_down, 
    double     pre_neuron_fraction
  ) {
    projections.push_back(proj);
    max_col_shift_up.push_back(max_col_up);
    max_col_shift_down.push_back(max_col_down);
    max_pch_shift_up.push_back(max_pch_up);
    max_pch_shift_down.push_back(max_pch_down);
    projection_fraction.push_back(pre_neuron_fraction);
    n_projections++;
  }

// Constructor, network
network::network() { 
    cpp_rng.seed(1234); // Fixed seed for reproducibility of internal C++ sampling
  }

// Helper: reconstruct node index
int network::node_idx_lookup(
    int base,     // distinguishes cortical layers (base 0) and subcortical layers (base = n_cortical_layers)
    int p,        // patch index
    int l,        // layer index 
    int c,        // column index
    int h,        // hemisphere index
    int n_layers  // Number of layers within this region 
  ) {
    /*
     * Network dimensions and indexes: 
     * n_hem = ntw.n[0];
     * n_sub = ntw.n[1];
     * n_lyr = ntw.n[2];
     * n_cls = ntw.n[3];
     * n_pch = ntw.n[4];
     * 5 = apical layer
     */
    int idx =
      base                                    +
      p    * (ntw.n[0] * ntw.n[3] * n_layers) +
      l    * (ntw.n[0] * ntw.n[3])            +
      c    *  ntw.n[0]                        +
      h;
    return idx;
  }

// Expand all cell type scalar parameters into per-neuron Eigen arrays
void network::set_neuron_params() {
    // Initialize
    per_nrn.v_bound               = ArrayXd(n_neurons);
    per_nrn.dHdv_bound            = ArrayXd(n_neurons);
    per_nrn.I_spike               = ArrayXd(n_neurons);
    per_nrn.v_spike               = ArrayXd(n_neurons);
    per_nrn.tau_spike             = ArrayXd(n_neurons); 
    per_nrn.v_rest                = ArrayXd(n_neurons);
    per_nrn.v_threshold           = ArrayXd(n_neurons);
    per_nrn.g_leak                = ArrayXd(n_neurons);
    per_nrn.max_spike_rate        = ArrayXd(n_neurons);
    per_nrn.tau_fast              = ArrayXd(n_neurons); 
    per_nrn.tau_slow              = ArrayXd(n_neurons); 
    per_nrn.tau_Vs                = ArrayXd(n_neurons);
    per_nrn.dCdr                  = ArrayXd(n_neurons); 
    per_nrn.dVdr                  = ArrayXd(n_neurons); 
    per_nrn.spike_velocity        = ArrayXd(n_neurons);
    per_nrn.dendrite_velocity     = ArrayXd(n_neurons); 
    per_nrn.Ta                    = ArrayXd(n_neurons); 
    per_nrn.tA                    = ArrayXd(n_neurons); 
    per_nrn.v_eq                  = ArrayXXd(n_neurons, n_neurons); 
    per_nrn.tau_syn               = ArrayXXd(n_neurons, n_neurons); 
    
    // Create temporary equilibrium potential and tau_syn decay time constant matrices
    int n_types = static_cast<int>(neuron_types.size());
    ArrayXXd temp_ep = ArrayXXd(n_types, n_neurons); 
    ArrayXXd temp_ts = ArrayXXd(n_types, n_neurons); 
    for (int i = 0; i < n_neurons; ++i) {
      for (int j = 0; j < n_types; ++j) {
        temp_ep(j, i) = neuron_types[j].v_eq[per_nrn.neuron_type_num[i]];
        temp_ts(j, i) = neuron_types[j].tau_syn[per_nrn.neuron_type_num[i]];
      }
    }
    
    // Fill values per neuron
    for (int i = 0; i < n_neurons; ++i) {
        const cell_type& ct          = neuron_types[per_nrn.neuron_type_num[i]];
        per_nrn.v_bound(i)           = std::abs(ct.v_rest) * ct.v_bound;
        per_nrn.dHdv_bound(i)        = ct.dHdv_bound * ct.I_spike;
        per_nrn.I_spike(i)           = ct.I_spike;
        per_nrn.v_spike(i)           = ct.v_spike;
        per_nrn.tau_spike(i)         = ct.tau_spike;
        per_nrn.v_rest(i)            = ct.v_rest;
        per_nrn.v_threshold(i)       = ct.v_threshold;
        per_nrn.g_leak(i)            = ct.g_leak;
        per_nrn.max_spike_rate(i)    = ct.max_spike_rate;
        per_nrn.tau_fast(i)          = ct.tau_fast; 
        per_nrn.tau_slow(i)          = ct.tau_slow; 
        per_nrn.tau_Vs(i)            = ct.tau_Vs;
        per_nrn.dCdr(i)              = ct.dCdr; 
        per_nrn.dVdr(i)              = ct.dVdr; 
        per_nrn.spike_velocity(i)    = ct.spike_velocity;
        per_nrn.dendrite_velocity(i) = ct.dendrite_velocity; 
        per_nrn.Ta(i)                = ct.Ta; 
        per_nrn.tA(i)                = ct.tA; 
        per_nrn.v_eq.row(i)          = temp_ep.row(per_nrn.neuron_type_num[i]);
        per_nrn.tau_syn.row(i)       = temp_ts.row(per_nrn.neuron_type_num[i]);
    }
  }

// Set up network structure
void network::set_network_structure(
    CharacterVector nrn_types,
    List            hsl_names,  // list containing three elements, all CharacterVector
    IntegerVector   n, 
    NumericVector   sep_factor, 
    double          lyr_height,
    double          cls_diameter,
    double          seg_length, 
    double          synaptic_neighborhood,
    IntegerMatrix   nrn_per_node
  ) {
    
    // Check and unpack n
    if (n.size() != 5) {
      Rcpp::Rcout << "n size: " << n.size() << std::endl;
      Rcpp::stop("Length of n must be 5");
    }
    int n_hem = n[0];
    int n_sub = n[1];
    int n_lyr = n[2];
    int n_cls = n[3];
    int n_pch = n[4];
    if (!(n_hem == 1 || n_hem == 2)) {
      Rcpp::Rcout << "n_hem: " << n_hem << std::endl;
      Rcpp::stop("n_hem must equal 1 or 2");
    }
    if (n_sub < 0) {
      Rcpp::Rcout << "n_sub: " << n_sub << std::endl;
      Rcpp::stop("n_sub must be >= 0");
    }
    if (n_lyr < 1 || n_cls < 1 || n_pch < 1) {
      Rcpp::Rcout << "n_lyr: " << n_lyr << ", n_cls: " << n_cls << ", n_pch: " << n_pch << std::endl;
      Rcpp::stop("n_lyr, n_cls, and n_pch must all be >= 1");
    }
    
    // Check and unpack sep_factor
    if (sep_factor.size() != 5) {
      Rcpp::Rcout << "sep_factor size: " << sep_factor.size() << std::endl;
      Rcpp::stop("Length of sep_factor must be 5");
    }
    double hem_separation_factor = sep_factor[0];
    double sub_separation_factor = sep_factor[1];
    double lyr_separation_factor = sep_factor[2];
    double cls_separation_factor = sep_factor[3];
    double pch_separation_factor = sep_factor[4];
    
    // Check and unpack layer names (needed for motifs)
    if (hsl_names.size() != 3) {
      Rcpp::Rcout << "hsl_names size: " << hsl_names.size() << std::endl;
      Rcpp::stop("Length of hsl_names must be 3");
    }
    CharacterVector hem_names = hsl_names[0];
    CharacterVector sub_names = hsl_names[1]; 
    CharacterVector lyr_names = hsl_names[2];
    if (hem_names.size() != n_hem) {
      Rcpp::Rcout << "hem_names size: " << hem_names.size() << ", n_hem: " << n_hem << std::endl;
      Rcpp::stop("Length of hem_names must equal n_hem");
    }
    if (sub_names.size() != n_sub) {
      Rcpp::Rcout << "sub_names size: " << sub_names.size() << ", n_sub: " << n_sub << std::endl;
      Rcpp::stop("Length of sub_names must equal n_sub");
    }
    if (lyr_names.size() != n_lyr) {
      Rcpp::Rcout << "lyr_names size: " << lyr_names.size() << ", n_lyr: " << n_lyr << std::endl;
      Rcpp::stop("Length of lyr_names must equal n_lyr");
    }
    
    // Validate that cortical and subcortical layer names are globally unique
    for (int i = 0; i < n_sub; ++i) {
      if (find_first(lyr_names, std::string(sub_names[i])) >= 0) {
        Rcpp::Rcout << "Duplicate layer name: " << sub_names[i] << std::endl;
        Rcpp::stop("Subcortical and cortical layer names must all be distinct");
      }
    }
    
    // Load cell types 
    for (String nt : nrn_types) {
      std::string nts = nt;
      auto it = get_cell_types().find(nts);
      if (it == get_cell_types().end()) Rcpp::stop("Unknown neuron type: %s", nts);
      neuron_types.push_back((*it).second);
      neuron_type_names.push_back(nts);
    }
    
    // Prune each local cell type's g_syn, v_eq, and tau_syn vectors 
    /*
     * They are currently indexed by position in the global type registry) down to only the types present in
     * this network, reordered to match local indexing. This lets find_synapse() and any other per-network
     * lookup use the local type index directly, with no need to re-derive a global index or duplicate the 
     * global type-name list.
     */
    const Vstr& global_names = get_all_type_names();
    int         n_local      = neuron_type_names.size();
    for (int i = 0; i < n_local; ++i) {
      const Vdbl fullsc = neuron_types[i].g_syn;  // copy before overwrite
      const Vdbl fullep = neuron_types[i].v_eq;
      const Vdbl fullts = neuron_types[i].tau_syn; 
      Vdbl prunedsc(n_local, 0.0);
      Vdbl prunedep(n_local, 0.0); 
      Vdbl prunedts(n_local, 0.0); 
      for (int j = 0; j < n_local; ++j) {
        int g = find_first(global_names, neuron_type_names[j]);
        if (g >= 0 && g < static_cast<int>(fullsc.size())) {
          prunedsc[j] = fullsc[g];
        } else {
          Rcpp::warning(
            "Cell type '%s' has no defined synaptic conductance for '%s' inputs; defaulting to 0",
            neuron_type_names[i].c_str(), neuron_type_names[j].c_str()
          );
        }
        if (g >= 0 && g < static_cast<int>(fullep.size())) {
          prunedep[j] = fullep[g]; 
        } else {
          Rcpp::warning(
            "Cell type '%s' has no defined equilibrium potential for '%s' inputs; defaulting to 0",
            neuron_type_names[i].c_str(), neuron_type_names[j].c_str()
          );
        }
        if (g >= 0 && g < static_cast<int>(fullts.size())) {
          prunedts[j] = fullts[g];
        } else {
          Rcpp::warning(
            "Cell type '%s' has no defined tau_syn for '%s' inputs; defaulting to 0",
            neuron_type_names[i].c_str(), neuron_type_names[j].c_str()
          );
        }
      }
      neuron_types[i].g_syn   = prunedsc;
      neuron_types[i].v_eq    = prunedep;
      neuron_types[i].tau_syn = prunedts;
    }
    
    // Check dimensions of nrn_per_node
    bool reuse_nrn_per_node = true; 
    if (nrn_per_node.nrow() != n_lyr + n_sub) {
      if (n_hem == 1 || nrn_per_node.nrow() != 2 * n_lyr + n_sub) {
        Rcpp::Rcout << "nrn_per_node nrow: " << nrn_per_node.nrow() << ", n_lyr: " << n_lyr << ", n_sub: " << n_sub << std::endl;
        Rcpp::stop("Nrows of nrn_per_node must equal n_lyr + n_sub, or (if n_hem = 2), 2 * this sum");
      } else {
        reuse_nrn_per_node = false; 
      }
    }
    if (nrn_per_node.ncol() != neuron_types.size()) {
      Rcpp::Rcout << "nrn_per_node ncol: " << nrn_per_node.ncol() << ", length of neuron_types: " << neuron_types.size() << std::endl;
      Rcpp::stop("Ncols of nrn_per_node must equal length of neuron_types");
    }
    
    // Are we making a single-neuron (per type) network? 
    bool single_cell = 
      nrn_per_node.nrow()     == 1       && 
      nrn_per_node.ncol()     == n_local && 
      Rcpp::sum(nrn_per_node) == n_local ? true : false;
    
    // Set/save other network parameters
    ntw.hsl_names             = {hem_names, sub_names, lyr_names};
    ntw.n                     = {n_hem, n_sub, n_lyr, n_cls, n_pch};
    ntw.sep_factors           = {
      hem_separation_factor, 
      sub_separation_factor, 
      lyr_separation_factor, 
      cls_separation_factor, 
      pch_separation_factor
    };
    ntw.lyr_height            = lyr_height;
    ntw.cls_diameter          = cls_diameter;
    ntw.seg_length            = seg_length;
    ntw.synaptic_neighborhood = synaptic_neighborhood;
    ntw.nrn_per_node          = to_eiMat(nrn_per_node);
    
    // Set expected node radius
    double lh                 = lyr_height   * lyr_separation_factor / 2.0;
    double cd                 = cls_diameter * cls_separation_factor / 2.0;
    double pd                 = cls_diameter * pch_separation_factor / 2.0;
    ntw.expected_node_radius  = std::sqrt(lh*lh + cd*cd + pd*pd);
    
    // Set network components
    n_neurons            = 0;                                                 // Compute total number of neurons as we go
    int n_nodes_cortical = n_hem * n_lyr * n_cls * n_pch;                     // Compute cortical nodes
    int n_nodes          = n_nodes_cortical + n_hem * n_sub * n_cls * n_pch;  // ... add subcortical nodes
    node_range_ends.assign(n_nodes, 0);
    coords.node_spatial.resize(n_nodes, 3);
    
    // Build layer groups: cortical first, then subcortical (if any)
    struct LayerGroup { bool is_sub; int n_layers; int node_base; int row_base; };
    std::vector<LayerGroup> groups = {{false, n_lyr, 0, 0}};
    if (n_sub > 0) groups.push_back({true, n_sub, n_nodes_cortical, n_lyr});
    
    // Set node locations for all layer groups
    for (const auto& g : groups) {
      for (int p = 0; p < n_pch; ++p) {
        for (int l = 0; l < g.n_layers; ++l) {
          for (int c = 0; c < n_cls; ++c) {
            for (int h = 0; h < n_hem; ++h) {
              // Find node index
              int node_idx = node_idx_lookup(g.node_base, p, l, c, h, g.n_layers); 
              // Set global spatial coordinates for this node
              coords.node_spatial(node_idx, 0) = static_cast<double>(p) * cls_diameter/2.0 * pch_separation_factor;  // z, patch
              coords.node_spatial(node_idx, 1) = static_cast<double>(l) * lyr_height  /2.0 * lyr_separation_factor;  // y, layer
              coords.node_spatial(node_idx, 2) = static_cast<double>(c) * cls_diameter/2.0 * cls_separation_factor;  // x, column
              // If subcortical, shift y below the cortical sheet
              if (g.is_sub) { coords.node_spatial(node_idx, 1) -= lyr_height  /2.0 * sub_separation_factor; }
              // If second hemisphere, adjust z
              if (h > 0)    { coords.node_spatial(node_idx, 0) += cls_diameter/2.0 * hem_separation_factor; }
              for (int t = 0; t < neuron_types.size(); t++) {
                // Randomly select neuron numbers for each node
                int row_idx = g.row_base + l; 
                if (!reuse_nrn_per_node) { row_idx *= h + 1; }
                int n = single_cell ? 1 : std::poisson_distribution<int>(ntw.nrn_per_node(row_idx, t))(cpp_rng);
                // Keep track of the number of cells assigned so far
                n_neurons += n;
                // Record type identity for each cell
                for (int i = 0; i < n; ++i) { per_nrn.neuron_type_num.push_back(t); }
              }
              // Save end-point index for this node
              node_range_ends[node_idx] = n_neurons - 1;
            }
          }
        }
      }
    }
    
    // Expand all cell type parameters into per-neuron vectors (now that n_neurons is known)
    set_neuron_params();
    
    // Set length of the vectors holding cell processes
    arbors.resize(n_neurons);
    
    // Resize arbor path distance matrices
    per_nrn.pre_syn_travel  = ArrayXXd::Constant(n_neurons, n_neurons, -1.0);
    per_nrn.post_syn_travel = ArrayXXd::Constant(n_neurons, n_neurons, 0.0);
    
    // Resize network coordinate components 
    coords.spatial = MatrixXd::Zero(n_neurons, 3); 
    coords.node    = MatrixXi::Zero(n_neurons, 6); 
    // ... have: hemisphere (z), subcortical layer (y), patch (z), layer (y), column (x), apical layer (y)
    
  }

// Function to make axon and dendrite branches
void network::make_arbor_branch(
    double          max_bias,          // Maximum weight of bias when following attractor
    int             cell_idx,          // Number of neuron for which to make processes
    bool            is_axon,           // Whether to make axon (true) or dendrite (false)
    bool            is_apical,         // Whether to mark as apical dendrite
    int             parent_branch_idx, // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
    const Vector3d& attractor_point,   // z (hemisphere, patch), y (subcortical or cortical layer), x (column); if all zeros, no attractor bias; otherwise, bias branch growth toward this point
    bool            hit_attractor      // Ensure arbor reaches attractor? For apical dendrites and long-range connections
  ) {
    
    // Check attractor point
    bool use_attractor = false;
    if (attractor_point(0) != 0.0 ||
        attractor_point(1) != 0.0 ||
        attractor_point(2) != 0.0) {
      use_attractor = true;
    } else {
      hit_attractor = false; 
    }
    
    // Check segment divisor
    if (ntw.seg_length <= 0.0) { Rcpp::stop("segment length less than or equal to zero"); }
    // Compute expected number of segments 
    int n_segments_radius = static_cast<int>(std::round(ntw.expected_node_radius / ntw.seg_length));
    int n_segments        = n_segments_radius;
    if (n_segments < 2) { n_segments = 2; }
    
    // Set parent flag 
    bool has_parent = parent_branch_idx >= 0;
    
    // Find initial point 
    Vector3d last_node;
    Vector3d soma_coordinates = coords.spatial.row(cell_idx);
    int parent_idx;
    if (has_parent) {
      // If child of parent branch, make sure parent exists and check axon flag
      if (parent_branch_idx >= arbors[cell_idx].axon.size())   { Rcpp::stop("Parent branch index exceeds number of branches in arbor"); }
      if (is_axon != arbors[cell_idx].axon[parent_branch_idx]) { Rcpp::stop("Parent branch type (axon vs dendrite) does not match specified branch type for new branch"); }
      // Randomly select branch point 
      int parent_branch_length = arbors[cell_idx].coordinates[parent_branch_idx].size();
      if (parent_branch_length == 0) { Rcpp::stop("Parent branch has no segments to branch from"); }
      Vdbl probs(parent_branch_length);
      // ... set probabilities for higher weight near 1 (will be normalized by std::discrete_distribution)
      for (int i = 1; i <= parent_branch_length; ++i) { probs[i - 1] = 1.0 / (i*i); }
      std::discrete_distribution<int> branch_point_dist(probs.begin(), probs.end());
      int branch_point = branch_point_dist(cpp_rng);
      // ... and set as initial point
      last_node = arbors[cell_idx].coordinates[parent_branch_idx][branch_point];
      // ... ensure this point not marked as a leaf 
      arbors[cell_idx].leafs[parent_branch_idx][branch_point] = 0;
      // ... and set initial parent node idx
      parent_idx = branch_point;
    } else {
      // Set axon flag for new process arbor
      if (is_axon) {
        arbors[cell_idx].axon_idx.push_back(static_cast<int>(arbors[cell_idx].axon.size()));
        arbors[cell_idx].axon.push_back(true);
        arbors[cell_idx].apical.push_back(false);
      } else {
        arbors[cell_idx].axon.push_back(false);
        arbors[cell_idx].apical.push_back(is_apical);       // always push to keep apical in sync with axon
      }
      last_node = soma_coordinates;                         // Set initial point as the soma location
      arbors[cell_idx].coordinates.push_back({last_node});  // Initialize new coordinates vector (of Vector3d) with first row as spatial coordinates of the cell body 
      arbors[cell_idx].node_type.push_back({"soma"});       // ... and initialize node_type vector and mark that this first point is "soma"
      arbors[cell_idx].parents.push_back({-1});             // ... and initialize new vector to track node parents
      arbors[cell_idx].leafs.push_back({0});                // ... and initialize leafs vector and mark that this first point is not a leaf 
      arbors[cell_idx].synapses.push_back({0});             // ... and initialize synapses vector and mark that this first point is not a synapse
      parent_branch_idx = arbors[cell_idx].axon.size() - 1; // ... set as parent branch 
      parent_idx = 0;                                       // ... and set initial parent node idx 
    }
    
    // Grab spine density and branch spread
    double spine_density = neuron_types[per_nrn.neuron_type_num[cell_idx]].spine_density;
    double branch_spread = neuron_types[per_nrn.neuron_type_num[cell_idx]].branch_spread;
    
    // If using attractor, adjust expected number of segments to ensure the attractor can be reached
    double bias_component_magnitude_init = std::numeric_limits<double>::infinity();
    if (use_attractor) {
      // Get initial distance to the attractor point
      Vector3d bias_attractor_point = attractor_point - arbors[cell_idx].coordinates[parent_branch_idx].back();
      bias_component_magnitude_init = bias_attractor_point.norm();
      // Ensure not zero (division below)
      if (bias_component_magnitude_init == 0.0) { bias_component_magnitude_init = 1.0; }
      // Find straight-line ratio of distance to attractor to node radius (the initial expected distance)
      double n_segment_scalar = bias_component_magnitude_init / ntw.expected_node_radius;
      // Apply scalar
      n_segments *= static_cast<int>(std::round(n_segment_scalar));
    }
    double bias_component_magnitude = bias_component_magnitude_init;
    
    // Apply exponential decay to n_segments, ensuring at least 1
    if (!hit_attractor) {
      n_segments = static_cast<int>(std::round(
        std::exponential_distribution<double>(
          2.995732/(static_cast<double>(n_segments) - 1.0)
        )(cpp_rng))) + 1;
      /* 
       * CDF of exponential distribution is 1 - exp(-lambda * x)
       * So, lambda = log(0.05)/-x is the exponential rate such that 
       *  95% of points are within x
       * log(0.05) = -2.995732
       */
    } else {
      // Add enough leeway to reach attractor
      n_segments *= 3.0; 
    }
    
    // Make branch
    double attractor_boundary_distance = ntw.expected_node_radius / 8.0; 
    for (int s = 0; s < n_segments; s++) {
      
      // Make random component of the step
      Vector3d step = {
        norm(cpp_rng) * ntw.seg_length,  // z
        norm(cpp_rng) * ntw.seg_length,  // y
        norm(cpp_rng) * ntw.seg_length   // x
      };
      double random_component_magnitude = step.norm();
      // ... bias step away from soma in proportion to branch spread
      Vector3d expand = last_node - soma_coordinates;
      // ... normalize expansion component so that it's the same magnitude as the random component and set weight with branch_spread, in proportion to distance from soma
      double weight_expand              = branch_spread;
      double expand_component_magnitude = expand.norm();
      if (expand_component_magnitude > 0) {
        expand        *= random_component_magnitude / expand_component_magnitude;
        weight_expand *= ntw.seg_length             / expand_component_magnitude; 
        if (weight_expand >  0.9) { weight_expand = 0.9; } // Cap weighted branch spread at 0.9
        if (weight_expand <= 0.1) { weight_expand = 0.1; } // Ensure weighted branch spread is positive
      } else {
        expand = Vector3d::Zero(); // If last_node is exactly at the soma, there is no expansion component
      }
      // ... make weighted combination of the step and directed component
      step = (1 - weight_expand) * step + weight_expand * expand;
      
      // Make directed component of the step (z, y, x)
      if (use_attractor) {
        Vector3d bias = attractor_point - last_node;
        // ... normalize directed component so that it's the same magnitude as the random component
        bias_component_magnitude = bias.norm();
        if (bias_component_magnitude > 0) {
          bias *= random_component_magnitude / bias_component_magnitude;
        } else {
          bias = Vector3d::Zero(); // If last_node is exactly at the attractor point, there is no bias component
        }
        // ... randomly select weight with expected value in proportion to distance to attractor point 
        double weight_bias = bias_component_magnitude / bias_component_magnitude_init;
        if (weight_bias >  max_bias) { weight_bias = max_bias; } // Cap weight
        if (weight_bias <= 0.1)      { weight_bias = 0.1; }      // Ensure weight is positive
        // ... make weighted combination of the step and bias 
        step = (1 - weight_bias) * step + weight_bias * bias;
      }
      
      // Add the step to the previous segment's coordinates to get the new segment's coordinates, and add to arbor coordinates
      Vector3d new_node = last_node + step;
      if (new_node.array().isNaN().any()) { Rcpp::stop("NaN detected in Vector3d"); }
      arbors[cell_idx].coordinates[parent_branch_idx].push_back(new_node);
      // ... and update last_node 
      last_node = new_node;
      
      // Update and add parent index
      arbors[cell_idx].parents[parent_branch_idx].push_back(parent_idx);
      parent_idx = arbors[cell_idx].coordinates[parent_branch_idx].size() - 1;
      
      // Mark node type 
      if (is_axon) {
        arbors[cell_idx].node_type[parent_branch_idx].push_back("axon_shaft");
      } else {
        // Randomly mark some nodes as spines based on spine density
        if (unif(cpp_rng) < spine_density) {
          arbors[cell_idx].node_type[parent_branch_idx].push_back("spine");
        } else {
          arbors[cell_idx].node_type[parent_branch_idx].push_back("dendrite_shaft");
        }
      }
      
      // Mark whether this node is a leaf
      if (s < n_segments - 1) {
        arbors[cell_idx].leafs[parent_branch_idx].push_back(0);
      } else {
        arbors[cell_idx].leafs[parent_branch_idx].push_back(1);
      }
      
      // Mark that this node is not a synapse 
      arbors[cell_idx].synapses[parent_branch_idx].push_back(0);
      
      // Check distance to attractor point 
      if (use_attractor && bias_component_magnitude < attractor_boundary_distance) { break; }
      
    }
    
  }

// Function to make axon and dendrite arbors
void network::make_arbor(
    int         n_branches,            // Expected number of branches, including the main process 
    int         cell_idx,              // Number of neuron for which to make processes
    bool        is_axon,               // Whether to make axon (true) or dendrite (false)
    bool        is_apical,             // Whether to mark as apical dendrite
    int         parent_branch_idx,     // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
    const Pnt3& attractor_points,      // z (patch), y (layer), x (column); if all zeros, no attractor bias; otherwise, bias branch growth toward this point
    bool        hit_attractor          // Ensure arbor reaches attractor? For apical dendrites and long-range connections
  ) {
    
    // Find number of existing branches 
    int n_existing_arbors = arbors[cell_idx].axon.size();
    
    // Randomly set number of branches 
    if (n_branches < 2) { n_branches = 2; }
    n_branches = std::poisson_distribution<int>(n_branches - 1)(cpp_rng) + 1; // Ensure at least 1 branch
    
    // Make branch structure
    Vint parent_branch_idx_list; 
    if (parent_branch_idx == -1) {
      
      // Starting fresh, no matter what other branches already exist
      parent_branch_idx_list.push_back(-1);
      // Set arbor ID number 
      arbors[cell_idx].arbor_id.push_back(n_existing_arbors);
      // Grab branch independence 
      double branch_independence = neuron_types[per_nrn.neuron_type_num[cell_idx]].branch_independence;
      // Track number of additional new branches
      int n_new_branches = 0;
      for (int b = 1; b < n_branches; ++b) {
        if (unif(cpp_rng) < branch_independence) {
          parent_branch_idx_list.push_back(-1);
          arbors[cell_idx].arbor_id.push_back(++n_new_branches + n_existing_arbors);
        } else {
          int parent_idx = n_existing_arbors + std::uniform_int_distribution<int>(0, n_new_branches)(cpp_rng);
          parent_branch_idx_list.push_back(parent_idx);
        }
      }
      
    } else {
      
      // Starting from existing branch, so make sure to build off of that branch and its children
      if (parent_branch_idx >= n_existing_arbors) {
        Rcpp::Rcout << "Parent branch index: " << parent_branch_idx << ", number of existing branches: " << n_existing_arbors << std::endl;
        Rcpp::stop("Parent branch index exceeds number of branches in arbor");
      }
      parent_branch_idx_list.assign(n_branches, parent_branch_idx);
      
    }
    
    // Ensure apical dendrites hit their attractor
    hit_attractor   = is_apical || hit_attractor;
    // Set branch to hit by
    int hit_by      = std::uniform_int_distribution<int>(0, n_branches)(cpp_rng);
    // Set max bias 
    double max_bias = 0.5;
    
    // For each branch to-be-made:
    int n_attractor_points = attractor_points.size();
    for (int b = 0; b < n_branches; ++b) {
      
      // Make random linear combination of attractor points 
      Vector3d pnt = Vector3d::Zero();
      double   rw  = 1.0;
      double   w   = 0.0;
      for (int i = 0; i < n_attractor_points; ++i) {
        if (i + 1 < n_attractor_points) {
          w   = unif(cpp_rng) * rw;
          rw -= w;
        } else {
          w   = rw; // Ensure all weights sum to 1
        }
        pnt += attractor_points[i] * w;
      }
      
      // Make arbor branch
      make_arbor_branch(
        max_bias,
        cell_idx,
        is_axon,
        is_apical,
        parent_branch_idx_list[b],
        pnt,
        hit_attractor && b >= hit_by
      );
      
    }
    
  }

// Function to set synaptic conductances and spatial coordinates for all local nodes 
void network::make_local_nodes() {
    
    if (edges.type.size() != 0) {
      Rcpp::Rcout << "Edge types have already been set; cannot run make_local_nodes twice; returning." << std::endl;
      return;
    }
    
    // Initialize vectors to track local edge coordinates
    Vint local_edges_pre; 
    Vint local_edges_post;
    
    // Initialize local synaptic conductance array
    ArrayXXd local_g_syn = ArrayXXd::Zero(n_neurons, n_neurons);
    
    // Build layer groups: cortical first, then subcortical (if any)
    struct                  LayerGroup { bool is_sub; int n_layers; int node_base; };
    std::vector<LayerGroup> groups           = {{false, ntw.n[2], 0}};
    int                     n_nodes_cortical = ntw.n[0] * ntw.n[2] * ntw.n[3] * ntw.n[4];
    if (ntw.n[1] > 0) groups.push_back({true, ntw.n[1], n_nodes_cortical});
    
    // Layer type group
    for (const auto& g : groups) {
      // Patch (n_pch) index of the local node
      for (int p = 0; p < ntw.n[4]; ++p) {
        // Layer index of the local node
        for (int l = 0; l < g.n_layers; ++l) {
          // Column (n_cls) index of the local node
          for (int c = 0; c < ntw.n[3]; ++c) {
            // Hemisphere (n_hem) index of the local node
            for (int h = 0; h < ntw.n[0]; ++h) {
              
              // Get node ID number
              int    node_idx         = node_idx_lookup(g.node_base, p, l, c, h, g.n_layers); 
              // Get spatial position of this node
              double node_z           = coords.node_spatial(node_idx, 0); // z = hemisphere, patch
              double node_y           = coords.node_spatial(node_idx, 1); // y = cortical or subcortical layer
              double node_x           = coords.node_spatial(node_idx, 2); // x = column
              // Get the range of neuron ID numbers for this node
              int    node_range_start = (node_idx == 0) ? 0 : node_range_ends[node_idx - 1] + 1;
              int    node_range_end   = node_range_ends[node_idx];
              
              // Make local process arbors for all cells in this node
              for (int cell_idx = node_range_start; cell_idx <= node_range_end; ++cell_idx) {
                
                // Set spatial coordinates
                coords.spatial(cell_idx, 0) = node_z + norm(cpp_rng) * ntw.cls_diameter / 2.0;
                coords.spatial(cell_idx, 1) = node_y + norm(cpp_rng) * ntw.lyr_height   / 2.0;
                coords.spatial(cell_idx, 2) = node_x + norm(cpp_rng) * ntw.cls_diameter / 2.0;
                
                // Set node coordinates 
                /*
                 * For 0-4, must match ntw_struct: 
                 *   (0) hemisphere
                 *   (1) subcortical (e.g., thalamic) layer
                 *   (2) cortical layer 
                 *   (3) laminar and subcortical columns 
                 *   (4) laminar and subcortical patches    
                 *   (5) apical target layer 
                 */
                coords.node(cell_idx, 0) = h;
                coords.node(cell_idx, 1) = g.is_sub ? l  : -1;
                coords.node(cell_idx, 2) = g.is_sub ? -1 : l;
                coords.node(cell_idx, 3) = c;
                coords.node(cell_idx, 4) = p;
                coords.node(cell_idx, 5) = -1;
                
                // Get neurite information
                int         axon_branch_count     = neuron_types[per_nrn.neuron_type_num[cell_idx]].axon_branch_count;
                int         dendrite_branch_count = neuron_types[per_nrn.neuron_type_num[cell_idx]].dendrite_branch_count;
                std::string apical_target_layer   = neuron_types[per_nrn.neuron_type_num[cell_idx]].apical_target_layer;
                
                // Create local axon and dendrite arbors
                make_arbor(axon_branch_count,     cell_idx, true,  false);
                make_arbor(dendrite_branch_count, cell_idx, false, false);
                
                // Create apical dendrite, if any (cortical cells only)
                if (!g.is_sub && apical_target_layer != "none") {
                  // Find apical target layer idx (a_idx)
                  int a_idx = find_first(ntw.hsl_names[2], apical_target_layer);
                  coords.node(cell_idx, 5) = a_idx; 
                  if (a_idx >= 0) {
                    // MB: Much of this code duplicated from apply_circuit_motif; create function??
                    
                    // Get coordinates of the target node
                    // ... reconstruct target node ID number
                    int target_node_idx       = node_idx_lookup(g.node_base, p, a_idx, c, h, ntw.n[2]); 
                    // ... get spatial position of this node, z (patch), y (layer), x (column)
                    Vector3d attractor_point  = coords.node_spatial.row(target_node_idx); 
                    // ... use attractor y and cell z, x as attractor point (i.e., go straight up)
                    Pnt3     attractor_points = { {coords.spatial(cell_idx, 0), attractor_point(1), coords.spatial(cell_idx, 2)} };
                    
                    // Make apical dendrite arbor
                    make_arbor(
                      dendrite_branch_count, 
                      cell_idx, 
                      false,
                      true, // will also trigger hit_attractor
                      -1,   // should be a new dendrite
                      attractor_points 
                    );
                    
                  } else {
                    Rcpp::Rcout << "Apical target layer: " << apical_target_layer << ", layer names: " << ntw.hsl_names[2] << std::endl;
                    Rcpp::stop("Apical target layer not found in layer names");
                  }
                }
                
              }
              
              // For all combinations of pre- and post-synaptic neurons in this node
              for (int idx_pre = node_range_start; idx_pre <= node_range_end; idx_pre++) {
                
                // Set synaptic conductances into post-synaptic cells
                for (int idx_post = node_range_start; idx_post <= node_range_end; idx_post++) {
                  
                  // Search for synapse and compute its conductance
                  double g_syn = find_synapse(
                    idx_pre, idx_post, false
                  );
                  
                  if (g_syn > 0) {
                    // Set conductance 
                    local_g_syn(idx_post, idx_pre) = g_syn;
                    // Save edge coordinate
                    local_edges_pre.push_back(idx_pre);
                    local_edges_post.push_back(idx_post);
                  }
                  
                }
                
              }
              
            }
          }
        }
      }
    }
   
    // Save to g_syn matrix
    edges.g_syn.push_back(local_g_syn);
    
    // Collect local edge coordinates in matrix
    int n_local_edges  = local_edges_pre.size();
    MatrixXi local_edges(n_local_edges, 2); 
    local_edges.col(0) = Eigen::Map<VectorXi>(local_edges_pre.data(),  n_local_edges);
    local_edges.col(1) = Eigen::Map<VectorXi>(local_edges_post.data(), n_local_edges);
    
    // Save to edge types
    edges.type.push_back(local_edges);
   
    // Build apical-node -> cell index lookup table.
    // ... For each cell with an apical dendrite (coords.node col 5 >= 0), reconstruct
    //     its apical target node index and register the cell under that node.
    apical_node_cells.clear();
    for (int i = 0; i < n_neurons; ++i) {
      if (coords.node(i, 5) < 0) { continue; }
      // ... Apical dendrites are cortical-only
      int apical_node_idx = node_idx_lookup(
        0,                 // node_base
        coords.node(i, 4), // p
        coords.node(i, 5), // apical target layer index; -1 if none
        coords.node(i, 3), // c
        coords.node(i, 0), // h
        ntw.n[2]           // n_lyr 
      );
      apical_node_cells[apical_node_idx].push_back(i);
    }
    
  }

// Function to find synapses and return conductance into post-synaptic cell 
double network::find_synapse(
    int  idx_pre,
    int  idx_post,
    bool via_apical
  ) {
   
    // Check if this pre-post pair already has a synapse
    if (per_nrn.pre_syn_travel(idx_pre, idx_post) >= 0.0) { return(0.0); } 
   
    // Get post-synaptic dendrite branch indices
    int n_arbors = arbors[idx_post].axon.size();
    Vint  dendrite_idx;
    for (int i = 0; i < n_arbors; ++i) { 
      if (!arbors[idx_post].axon[i] && arbors[idx_post].apical[i] == via_apical) { dendrite_idx.push_back(i); }
    }
    if (dendrite_idx.empty()) { return 0.0; }
    
    // Check all axons 
    for (int ax : arbors[idx_pre].axon_idx) {
      // Check all dendrites 
      for (int dd : dendrite_idx) {
        
        // Check for synapses
        // ... first element of neighbor_idx is the axon node idx, second is the dendrite node idx
        Vint neighbor_idx = find_first_neighbor(
          arbors[idx_pre].coordinates[ax], 
          arbors[idx_post].coordinates[dd],
          ntw.synaptic_neighborhood,
          true // skip origin
        );
        
        // If one is found, create it
        if (neighbor_idx[0] >= 0) {
          
          // Extend the axon
          // ... add coordinates
          arbors[idx_pre].coordinates[ax].push_back(
              arbors[idx_post].coordinates[dd][neighbor_idx[1]]
            );
          // ... add parent 
          if (neighbor_idx[0] >= arbors[idx_pre].coordinates[ax].size()) { 
            Rcpp::stop("Neighbor index out of bounds for axon coordinates"); 
          }
          arbors[idx_pre].parents[ax].push_back(neighbor_idx[0]);
          // ... add node type 
          arbors[idx_pre].node_type[ax].push_back("axon_shaft");
          // ... ensure old node not marked as leaf 
          arbors[idx_pre].leafs[ax][neighbor_idx[0]] = 0;
          // ... and mark new node as leaf
          arbors[idx_pre].leafs[ax].push_back(1);
          // ... and mark new node as synapse
          arbors[idx_pre].synapses[ax].push_back(1);
          
          // Find signal travel distances to/from this synapse 
          per_nrn.pre_syn_travel(idx_post, idx_pre)  = integrate_along_arbor_to_soma(arbors[idx_pre].coordinates[ax].size() - 1, ax, idx_pre); 
          per_nrn.post_syn_travel(idx_post, idx_pre) = integrate_along_arbor_to_soma(neighbor_idx[1], dd, idx_post); 
         
          // Find and return default synaptic conductance
          return neuron_types[
            per_nrn.neuron_type_num[idx_post] // conductance across membrane of post-synaptic cell
          ].g_syn[
            per_nrn.neuron_type_num[idx_pre]  // as determined by the neurotransmitter type of the pre-synaptic cell
          ];
          
        }
        
      }
    }
    
    return 0.0;
    
  }

// Function to apply circuit motif
void network::apply_circuit_motif(
    const motif& cmot
  ) {
    
    // Check for local edges
    if (edges.type.size() < 1) { Rcpp::stop("Must set local edges before applying any circuit motifs."); }
    
    // Initialize vectors to track motif edge coordinates
    Vint motif_edges_pre; 
    Vint motif_edges_post;
    
    // Initialize motif synaptic conductance array
    ArrayXXd motif_g_syn = ArrayXXd::Zero(n_neurons, n_neurons);
    
    // Unpack frequently used network dimensions
    int n_hem = ntw.n[0];
    int n_sub = ntw.n[1];
    int n_lyr = ntw.n[2];
    int n_cls = ntw.n[3];
    int n_pch = ntw.n[4];
    int n_nodes_cortical = n_hem * n_lyr * n_cls * n_pch;
    
    // Layer-group descriptor (mirrors the struct used in set_network_structure / make_local_nodes)
    struct LayerGroup { bool is_sub; int n_layers; int node_base; };
    
    // Helper: resolve a layer name to its group and local layer index.
    // ... searches cortical names (hsl_names[2]) first, then subcortical (hsl_names[1]).
    // ... returns {group, local_idx}; local_idx == -1 if not found.
    auto resolve_layer = [&](const std::string& name) -> std::pair<LayerGroup, int> {
      int idx = find_first(ntw.hsl_names[2], name);
      if (idx >= 0) return {{false, n_lyr, 0}, idx};
      idx = find_first(ntw.hsl_names[1], name);
      if (idx >= 0) return {{true, n_sub, n_nodes_cortical}, idx};
      return {{false, 0, 0}, -1};
    };
    
    // Update arbors with this motif index
    for (int i = 0; i < n_neurons; ++i) { arbors[i].motifs.push_back(0); }
    int m_idx = edges.type.size();
    
    // For each projection in the motif
    const int n_projections = cmot.n_projections;
    for (int pj = 0; pj < n_projections; ++pj) {
      
      // Grab projection
      Prj proj = cmot.projections[pj];
      if (proj.hem_shift == n_hem) { Rcpp::stop("Can't apply contralateral projection if only one hemisphere"); }
      
      // Get indices for neuron_types in this network
      int t_pre  = find_first_by(
          static_cast<int>(neuron_types.size()),
          [&](int i){ return neuron_types[i].type_name; }, 
          proj.pre_type
        );
      int t_post = find_first_by(
          static_cast<int>(neuron_types.size()),
          [&](int i){ return neuron_types[i].type_name; }, 
          proj.post_type
        );
      if (t_pre < 0 || t_post < 0) { continue; }
      
      // Check dendrite type 
      if (proj.via_apical) {
        if (neuron_types[t_post].apical_target_layer == "none") { 
          Rcpp::stop("Projection requests via apical, but target type has no apical dendrite"); 
        }
      }
      
      // Resolve pre and post layers (search cortical then subcortical name lists)
      auto [g_pre,  layer_pre]  = resolve_layer(proj.pre_layer);
      auto [g_post, layer_post] = resolve_layer(proj.post_layer);
      if (layer_pre < 0 || layer_post < 0) { continue; }
      
      // Build layer masks using the correct coords.node column for each group:
      //   col 2 = cortical layer, col 1 = subcortical layer
      int pre_layer_col  = g_pre.is_sub  ? 1 : 2;
      int post_layer_col = g_post.is_sub ? 1 : 2;
      Vboo pre__type_layer_mask(n_neurons);
      Vboo post_type_layer_mask(n_neurons); 
      for (int i = 0; i < n_neurons; ++i) {
        pre__type_layer_mask[i] = per_nrn.neuron_type_num[i] == t_pre && coords.node(i, pre_layer_col) == layer_pre; 
        post_type_layer_mask[i] = proj.via_apical ? 
          per_nrn.neuron_type_num[i] == t_post : 
          per_nrn.neuron_type_num[i] == t_post && coords.node(i, post_layer_col) == layer_post;
      }
     
      // Grab max column/patch shifts
      VectorXi col_range =  VectorXi::Constant(2, 0);
      col_range(0)       = -cmot.max_col_shift_down[pj];
      col_range(1)       =  cmot.max_col_shift_up[pj];
      VectorXi pch_range =  VectorXi::Constant(2, 0);
      pch_range(0)       = -cmot.max_pch_shift_down[pj];
      pch_range(1)       =  cmot.max_pch_shift_up[pj];
      
      // Is this a intralaminar projection? 
      bool intralam = layer_pre == layer_post && !proj.via_apical;
      
      // Apply projection to each patch, hemisphere, and column
      for (int p = 0; p < n_pch; p++) {
        for (int h = 0; h < n_hem; h++) {
          
          // Determine if this motif applies to this hemisphere 
          if (cmot.hemi > 0 && cmot.hemi != h) { continue; }
          // Determine post-synaptic hemisphere (hem_shift: 0 = same, 1 = contralateral)
          int th = (h + proj.hem_shift) % n_hem;
          
          // Is this a long-range projectino? 
          bool long_range = proj.hem_shift || g_pre.is_sub != g_post.is_sub;
          
          // Apply projection to each column 
          for (int c = 0; c < n_cls; c++) {
            
            // Find indexes of pre-synaptic cells (layer + patch + hemisphere + column)
            // Sample based on pre_neuron_fraction parameter
            Vint eligible_pre_indices;
            int pre_node_idx         = node_idx_lookup(g_pre.node_base, p, layer_pre, c, h, g_pre.n_layers); 
            int pre_node_range_start = (pre_node_idx == 0) ? 0 : node_range_ends[pre_node_idx - 1] + 1;
            int pre_node_range_end   = node_range_ends[pre_node_idx];
            for (int i = pre_node_range_start; i <= pre_node_range_end; ++i) {
              if (pre__type_layer_mask[i]) { 
                eligible_pre_indices.push_back(i);
              }
            }
            if (eligible_pre_indices.empty()) { continue; }
            
            // Sample subset based on pre_neuron_fraction
            double pre_fraction = cmot.projection_fraction[pj];
            // Clip to valid range: [1/n_eligible, 1.0], then round
            int n_eligible  = eligible_pre_indices.size();
            int n_to_select = std::max(
              1, 
              std::min(
                static_cast<int>(std::round(n_eligible * pre_fraction)), 
                n_eligible
              )
            );
            
            // Randomly select subset
            std::shuffle(eligible_pre_indices.begin(), eligible_pre_indices.end(), cpp_rng);
            Vint pre_indices;
            for (int i = 0; i < n_to_select; ++i) {
              pre_indices.push_back(eligible_pre_indices[i]);
              arbors[eligible_pre_indices[i]].motifs[m_idx] = 1;
            }
            
            // Compute target column and patch ranges
            VectorXi col_range_shifted = col_range.array() + c;
            VectorXi pch_range_shifted = pch_range.array() + p;
            if (col_range_shifted[0] < 0)      { col_range_shifted[0] = 0; }
            if (col_range_shifted[1] >= n_cls) { col_range_shifted[1] = n_cls - 1; }
            if (pch_range_shifted[0] < 0)      { pch_range_shifted[0] = 0; }
            if (pch_range_shifted[1] >= n_pch) { pch_range_shifted[1] = n_pch - 1; } 
            
            // Construct target cell indices
            Vint post_indices;
            for (int tp = pch_range_shifted(0); tp <= pch_range_shifted(1); ++tp) {
              for (int tc = col_range_shifted(0); tc <= col_range_shifted(1); ++tc) {
                // Skip same-node self-connections
                if (intralam && h == th && c == tc && p == tp) { continue; }
                // Get coordinates of target node (post-synaptic hemisphere = th)
                int post_node_idx = node_idx_lookup(g_post.node_base, tp, layer_post, tc, th, g_post.n_layers);
                // Check whether each cell is a potential target.
                // ... When via_apical, post_node_idx is the apical target node, so use the
                //     apical-node lookup to find cells whose apical dendrites reach it.
                //     Otherwise use the body-node range as usual.
                if (proj.via_apical) {
                  auto it = apical_node_cells.find(post_node_idx);
                  if (it != apical_node_cells.end()) {
                    for (int i : it->second) {
                      if (post_type_layer_mask[i]) { post_indices.push_back(i); }
                    }
                  }
                } else {
                  int post_node_range_start = (post_node_idx == 0) ? 0 : node_range_ends[post_node_idx - 1] + 1;
                  int post_node_range_end   = node_range_ends[post_node_idx];
                  for (int i = post_node_range_start; i <= post_node_range_end; ++i) {
                    if (post_type_layer_mask[i]) { post_indices.push_back(i); }
                  }
                }
              }
            }
            if (post_indices.empty()) { continue; }
            
            // Make vectors of cardinal direction pairs
            Pnt3 tuu, tdd, tud, tdu;
            // ... check patch up-shift
            int tpu = pch_range_shifted(1);
            if (!(intralam && h == th && p == tpu)) {
              Vector3d tp_up = coords.node_spatial.row(
                node_idx_lookup(g_post.node_base, tpu, layer_post, c, th, g_post.n_layers)
              );
              tuu.push_back(tp_up); 
              tud.push_back(tp_up); 
            }
            // ... check patch down-shift
            int tpd = pch_range_shifted(0);
            if (!(intralam && h == th && p == tpd)) {
              Vector3d tp_down = coords.node_spatial.row(
                node_idx_lookup(g_post.node_base, tpd, layer_post, c, th, g_post.n_layers)
              );
              tdd.push_back(tp_down); 
              tdu.push_back(tp_down); 
            }
            // ... check column up-shift
            int tcu = col_range_shifted(1);
            if (!(intralam && h == th && c == tcu)) {
              Vector3d tc_up = coords.node_spatial.row(
                node_idx_lookup(g_post.node_base, p, layer_post, tcu, th, g_post.n_layers)
              );
              tuu.push_back(tc_up); 
              tdu.push_back(tc_up); 
            }
            // ... check column down-shift
            int tcd = col_range_shifted(0);
            if (!(intralam && h == th && c == tcd)) {
              Vector3d tc_down = coords.node_spatial.row(
                node_idx_lookup(g_post.node_base, p, layer_post, tcd, th, g_post.n_layers)
              );
              tdd.push_back(tc_down);
              tud.push_back(tc_down);
            }
            // Construct admissible attractor points 
            std::vector<Pnt3> attractor_points;
            if (!tuu.empty()) { attractor_points.push_back(tuu); }
            if (!tdd.empty()) { attractor_points.push_back(tdd); }
            if (!tud.empty()) { attractor_points.push_back(tud); }
            if (!tdu.empty()) { attractor_points.push_back(tdu); }
            if (attractor_points.empty()) { continue; }
            
            // Make meso-scale and long-range axon arbors for pre-synaptic cells
            std::uniform_int_distribution<int> attractor_dist(0, static_cast<int>(attractor_points.size() - 1));
            for (int cell_idx : pre_indices) {
              
              // Select random existing axon branch to extend from, or start from soma
              int axon_parent;
              if (!arbors[cell_idx].axon_idx.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(arbors[cell_idx].axon_idx.size() - 1));
                axon_parent = arbors[cell_idx].axon_idx[dist(cpp_rng)];
              } else {
                axon_parent = -1;
              }
              
              make_arbor(
                neuron_types[t_pre].axon_branch_count, 
                cell_idx, 
                true,                                        // is axon
                false,                                       // is not apical dendrite
                axon_parent, 
                attractor_points[attractor_dist(cpp_rng)],   // randomly chosen std::vector<Vector3d> (i.e., Pnt3)
                long_range || !intralam                      // if long-range, ensure attractor is hit
              );
              
            }
            
            // For all combinations of pre- and post-synaptic neurons in this projection
            for (int idx_pre : pre_indices) {
              for (int idx_post : post_indices) {
                
                double g_syn = find_synapse(
                  idx_pre, idx_post, proj.via_apical
                );
                
                if (g_syn > 0) {
                  motif_g_syn(idx_post, idx_pre) = g_syn;
                  motif_edges_pre.push_back(idx_pre);
                  motif_edges_post.push_back(idx_post);
                }
                
              }
            }
            
          } // end column loop
        } // end hemisphere loop
      } // end patch loop
    } // end projection loop
    
    // Save to synaptic conductance matrix vector 
    edges.g_syn.push_back(motif_g_syn);
    
    // Collect local edge coordinates in matrix
    int n_motif_edges = motif_edges_pre.size();
    MatrixXi motif_edges(n_motif_edges, 2); 
    motif_edges.col(0) = Eigen::Map<VectorXi>(motif_edges_pre.data(), n_motif_edges);
    motif_edges.col(1) = Eigen::Map<VectorXi>(motif_edges_post.data(), n_motif_edges);
    
    // Save to edge types
    edges.type.push_back(motif_edges);
    
    // Add motif name
    edges.motif_name.push_back(cmot.motif_name);
    
  }

// Method to fetch network components 
List network::fetch_network_components(
    bool include_arbors
  ) const {
   
    // Convert synaptic conductances into list of NumericMatrix
    int n_g_syn_matrices = edges.g_syn.size();
    List g_syn_matrices(n_g_syn_matrices);
    if (n_g_syn_matrices > 0) {
      for (int tci = 0; tci < n_g_syn_matrices; ++tci) {
        ArrayXXd tc = edges.g_syn[tci];
        NumericMatrix tc_r = to_NumMat(tc);
        g_syn_matrices[tci] = tc_r;
      } 
      g_syn_matrices.names() = edges.motif_name; 
    }
    
    // Convert edges.type into list of NumericMatrix
    List edge_type_matrices(edges.type.size());
    CharacterVector emn = CharacterVector::create("pre_neuron_idx", "post_neuron_idx");
    for (int eti = 0; eti < edges.type.size(); ++eti) {
      MatrixXi et = edges.type[eti];
      NumericMatrix et_r = to_NumMat(et);
      for (double& v : et_r) v++; // put into 1-indexed form for R
      colnames(et_r) = emn;
      edge_type_matrices[eti] = et_r;
    }
    
    // Convert arbors into numeric matrix; accumulate per-neuron synapse counts
    NumericMatrix arbor_motifs;
    NumericMatrix arbor_matrix;
    IntegerVector synapse_counts;
    if (include_arbors) {
      // Count total number of segments across all arbors
      int n_segments = 0;
      int n_roots    = 0;
      for (int n = 0; n < n_neurons; n++) {
        int n_arbors = arbors[n].axon.size();
        n_roots     += n_arbors;
        for (int a = 0; a < n_arbors; a++) {
          n_segments += arbors[n].coordinates[a].size();
        }
      }
      
      // Create matrix to hold arbor data, and a per-neuron synapse count vector
      arbor_matrix           = NumericMatrix(n_segments - n_roots, 13);
      colnames(arbor_matrix) = CharacterVector::create(
        "neuron_idx", "arbor_id", "is_axon", "node_type", "parent_idx", "is_leaf", "is_synapse", 
        "z_start", "y_start", "x_start", "z_end", "y_end", "x_end"
      );
      synapse_counts         = IntegerVector(n_neurons, 0);
      
      // Fill matrix with arbor data; accumulate synapse counts in the same pass
      int seg_idx = 0;
      int parent_skip_counter = 0;
      for (int n = 0; n < n_neurons; n++) {
        int n_arbors = arbors[n].axon.size();
        for (int a = 0; a < n_arbors; a++) {
          int n_segs = arbors[n].coordinates[a].size();
          double arbor_type = arbors[n].axon[a] ? 1.0 : 0.0;               // 1 for axon, 0 for dendrite
          // ... for each segment endpoint i
          for (int i = 0; i < n_segs; ++i) { 
            double parent_idx = arbors[n].parents[a][i];
            double node_type_idx;
            std::string node_type_str = arbors[n].node_type[a][i];
            if (node_type_str == "soma") {
              node_type_idx = 0.0;
            } else if (node_type_str == "dendrite_shaft") {
              node_type_idx = 1.0;
            } else if (node_type_str == "axon_shaft") {
              node_type_idx = 2.0;
            } else if (node_type_str == "spine") {
              node_type_idx = 3.0;
            } else {
              Rcpp::Rcout << "Unknown node type string: " << node_type_str << std::endl;
              Rcpp::stop("Unknown node type string in arbor structure");
            }
            if (parent_idx < 0) {                                                // Skip root node since it has no parent
              parent_skip_counter++;
              continue;
            }
            if (arbors[n].synapses[a][i]) synapse_counts[n]++;                   // Accumulate synapse count for this neuron
            arbor_matrix(seg_idx, 0)  = (double)n + 1;                           // neuron index, put into 1-indexed form for R
            arbor_matrix(seg_idx, 1)  = (double)arbors[n].arbor_id[a];           // arbor ID number
            arbor_matrix(seg_idx, 2)  = arbor_type;
            arbor_matrix(seg_idx, 3)  = node_type_idx;                           // 0 = "soma", 1 = "dendrite_shaft", 2 = "axon_shaft", or 3 = "spine"
            arbor_matrix(seg_idx, 4)  = parent_idx - parent_skip_counter + 1;    // put into 1-indexed form for R
            arbor_matrix(seg_idx, 5)  = (double)arbors[n].leafs[a][i];           // 1 if node is a leaf, zero if not
            arbor_matrix(seg_idx, 6)  = (double)arbors[n].synapses[a][i];        // 1 if node is a synapse, zero if not
            arbor_matrix(seg_idx, 7)  = arbors[n].coordinates[a][parent_idx][0]; // z_start
            arbor_matrix(seg_idx, 8)  = arbors[n].coordinates[a][parent_idx][1]; // y_start
            arbor_matrix(seg_idx, 9)  = arbors[n].coordinates[a][parent_idx][2]; // x_start
            arbor_matrix(seg_idx, 10) = arbors[n].coordinates[a][i][0];          // z_end
            arbor_matrix(seg_idx, 11) = arbors[n].coordinates[a][i][1];          // y_end
            arbor_matrix(seg_idx, 12) = arbors[n].coordinates[a][i][2];          // x_end
            seg_idx++;
          }
        }
      }
      
    }
    
    // Gather which cells have arbors involving which motifs
    int n_motifs = edges.motif_name.size();
    arbor_motifs = NumericMatrix(n_neurons, n_motifs);
    for (int i = 0; i < n_neurons; ++i) {
      for (int j = 0; j < n_motifs; ++j) {
        arbor_motifs(i, j) = arbors[i].motifs[j];
      }
    }
    colnames(arbor_motifs) = edges.motif_name;
    
    // Add labels 
    NumericMatrix coordinates_node_R         = to_NumMat(coords.node);
    // coords.node columns: 0=hemisphere, 1=subcortical layer, 2=cortical layer, 3=laminar and subcortical columns, 4=laminar and subcortical patches, 5=apical target layer
    if (coordinates_node_R.size()         > 0) { colnames(coordinates_node_R)         = CharacterVector::create("hem_idx", "sub_lyr_idx", "lyr_idx", "col_idx", "patch_idx", "apical_lyr"); }
    NumericMatrix coordinates_spatial_R      = to_NumMat(coords.spatial); 
    if (coordinates_spatial_R.size()      > 0) { colnames(coordinates_spatial_R)      = CharacterVector::create("z", "y", "x"); }
    NumericMatrix node_coordinates_spatial_R = to_NumMat(coords.node_spatial);
    if (node_coordinates_spatial_R.size() > 0) { colnames(node_coordinates_spatial_R) = CharacterVector::create("z", "y", "x"); }
    
    // Put non-sentinel values into 1-indexed form (-1 encodes "not in this group" and must be preserved)
    for (double& v : coordinates_node_R) { if (v >= 0) v++; }
    
    // Create neuron type (per neuron) list 
    CharacterVector neuron_type_name(n_neurons);
    for (int i = 0; i < n_neurons; ++i) neuron_type_name[i] = neuron_types[per_nrn.neuron_type_num[i]].type_name;
    
    // Guard hsl_names access: may be empty if set_network_structure was never called
    CharacterVector hem_names_out = (ntw.hsl_names.size() > 0) ? ntw.hsl_names[0] : CharacterVector(0);
    CharacterVector sub_names_out = (ntw.hsl_names.size() > 1) ? ntw.hsl_names[1] : CharacterVector(0);
    CharacterVector lyr_names_out = (ntw.hsl_names.size() > 2) ? ntw.hsl_names[2] : CharacterVector(0);
    
    // Build return list element-by-element to avoid the 20-argument limit on List::create()
    List result = List::create(
      _["n_neurons"]                = n_neurons,
      _["n_nodes"]                  = (int)node_range_ends.size(),
      _["n_hem"]                    = ntw.n[0],
      _["n_sub"]                    = ntw.n[1],
      _["n_layers"]                 = ntw.n[2],
      _["n_columns"]                = ntw.n[3],
      _["n_patches"]                = ntw.n[4],
      _["hem_names"]                = hem_names_out,
      _["sub_names"]                = sub_names_out,
      _["layer_names"]              = lyr_names_out,
      _["g_syn"]                    = g_syn_matrices,
      _["node_coordinates_spatial"] = node_coordinates_spatial_R,
      _["coordinates_spatial"]      = coordinates_spatial_R,
      _["coordinates_node"]         = coordinates_node_R,
      _["neuron_type_name"]         = neuron_type_name,
      _["neuron_type_num"]          = per_nrn.neuron_type_num,
      _["node_range_ends"]          = node_range_ends,
      _["edge_idx_by_type"]         = edge_type_matrices,
      _["edge_type_names"]          = edges.motif_name,
      _["sim_dt"]                   = sim_dt
    );
    result["arbor_motifs"]   = arbor_motifs;
    result["arbors"]         = arbor_matrix;
    result["synapse_counts"] = synapse_counts;  // length n_neurons when include_arbors = TRUE, empty otherwise
    return result;
   
  }

// Method to fetch BGT simulation results 
List network::fetch_sim_results() const {
    return  List::create(
      _["v_traces"]     = v_traces,
      _["spike_counts"] = spike_counts
    );
  }

// Function to compute pairwise lags based on axon path lengths and membrane velocity
double network::integrate_along_arbor_to_soma(
    int node_idx,   // Integrate from this node 
    int arbor_idx,  // ... on this arbor
    int cell_idx    // ... of this cell, to the cell's soma
  ) {
    double dist = std::numeric_limits<double>::epsilon();
    int    parent_node_idx = arbors[cell_idx].parents[arbor_idx][node_idx];
    while (parent_node_idx >= 0) {
      Vector3d node   = arbors[cell_idx].coordinates[arbor_idx][node_idx];
      dist           += (node - arbors[cell_idx].coordinates[arbor_idx][parent_node_idx]).norm();
      node_idx        = parent_node_idx; 
      parent_node_idx = arbors[cell_idx].parents[arbor_idx][node_idx];
    }
    return dist;
  }

// Simulate network responses to input current using Growth Transform model
void network::BGT(
    const NumericMatrix& I_stim_R, // matrix of stimulus currents in pA, n_neurons x n_steps
    double               dt,       // time step length in ms; units: ms/step
    double               v_initial // start all neurons with this membrane potential
  ) {
    
    // Save dt
    sim_dt = dt;
    
    // Convert stimulus current to Eigen array
    ArrayXXd I_stim = to_eMat(I_stim_R);
   
    // Check size of stimulus current matrix 
    if (I_stim.rows() != n_neurons) { Rcpp::stop("I_stim must have n_neurons rows"); }
    
    // Find number of time steps to simulate
    const int n_steps = I_stim.cols();
    
    // Collapse the synaptic conductances into a single array
    // ... rows as post-synaptic, cols as pre-synaptic
    ArrayXXd g_syn = ArrayXXd::Zero(n_neurons, n_neurons);
    for (const auto& m : edges.g_syn) { g_syn += m; }
    
    // Convert travel time matrices into time-step lags 
    ArrayXXi pre_syn_lags  = (per_nrn.pre_syn_travel.colwise()  / per_nrn.spike_velocity    / dt).round().cast<int>();   // rows as pre-synaptic, cols as post-synaptic, i.e., pre_syn_lags.col(j) = steps back needed for signal from i
    ArrayXXi post_syn_lags = (per_nrn.post_syn_travel.colwise() / per_nrn.dendrite_velocity / dt).round().cast<int>();   // rows as post-synaptic, cols as pre-synaptic, i.e., post_syn_lags.row(i) = steps back needed for signal from j
   
    // Resize matrix to hold simulated spike traces (membrane potential plus spike)
    v_traces.resize(n_neurons, n_steps);
    v_traces.setZero();
    v_traces.col(0).setConstant(v_initial);
    
    // Initialize array to hold simulated sub-threshold membrane potential traces (without spike)
    ArrayXXd v_sub    = ArrayXXd::Zero(n_neurons, n_steps);
    v_sub.col(0).setConstant(v_initial);
    // ... initialize matrix to keep track of spikes each at time step
    ArrayXd  spikes   = (v_sub.col(0) >= per_nrn.v_threshold).cast<double>();
    // ... initialize circular buffer to hold recent last_spike history for lagged lookups
    int      ls_buffer_size     = pre_syn_lags.maxCoeff() + 1;
    ArrayXXi last_spike_history = ArrayXXi::Zero(n_neurons, ls_buffer_size);
    
    // Resize spike_counts vector
    spike_counts.resize(n_neurons);
    spike_counts.setZero();
    // ... and make a local copy for tracking recent spikes (proxy for spikes/ms)
    ArrayXd spike_counts_recent = ArrayXd::Zero(n_neurons);
    // ... make vector to track time since last spike 
    ArrayXi last_spike          = ArrayXi::Zero(n_neurons); 
    
    // Compute spike height
    ArrayXd spike_height        = per_nrn.v_spike - per_nrn.v_threshold;
    
    // Convert max_spike_rate to simulation time steps 
    ArrayXd max_spike_rate_dt   = per_nrn.max_spike_rate * dt;
    
    // Set threshold for membrane response to slow currents 
    double  theta_low           = 0.1;
    double  theta_high          = 0.9; 
    ArrayXd theta_n             = ArrayXd::Constant(n_neurons, std::pow(theta_low, 4.0));
    
    // Initialize vector to hold Schmitt trigger for whether Ca levels are rising or falling
    // ... = 1 if + flowing in, = 0 if + being pushed out
    ArrayXd slow_current(n_neurons);
    
    // Set initial slow current
    slow_current.setOnes(); 
    
    // Initialize vectors to hold synaptic vesicle and intracellular calcium concentrations
    ArrayXd Vs(n_neurons); 
    ArrayXd Ca(n_neurons); 
    Vs.setOnes(); 
    Ca.setZero(); 
    
    // Set spike widths in simulation steps 
    ArrayXi tau_spike(n_neurons); 
    for (int i = 0; i < n_neurons; ++i) {
      tau_spike(i) = static_cast<int>(std::round(per_nrn.tau_spike(i) / dt)) + 1; 
    }
    
    // Precompute onset threshold: tau_onset(i,j) = tau_spike(j) - 1
    // ... last_spike_lagged == tau_onset identifies the first step of a spike arriving at synapse (i,j)
    ArrayXXi tau_onset = (tau_spike - 1).transpose().replicate(n_neurons, 1);
    
    // Initialize synaptic (post-synaptic current) gating matrix and its per-step decay factor
    // ... S(i, j) = fraction of open post-synaptic receptors on neuron i due to pre-synaptic neuron j
    ArrayXXd S                    = ArrayXXd::Zero(n_neurons, n_neurons); 
    // Normalize post_syn_travel
    // ... note: will be pathologically degenerate if only a few synapses very close to soma
    ArrayXd  travel_row_max       = per_nrn.post_syn_travel.rowwise().maxCoeff();
             travel_row_max       = (travel_row_max == 0.0).select(ArrayXd::Ones(n_neurons), travel_row_max);
    ArrayXXd post_syn_travel_norm = per_nrn.post_syn_travel.colwise() / travel_row_max;
    // Set syn_decay with tau_syn adjusted for normalized distance from soma 
    // ... tau_syn -> 0 gives a per-step decay of 0, recovering an instantaneous (boxcar) post-synaptic current
    ArrayXXd syn_decay            = (-dt / (per_nrn.tau_syn * post_syn_travel_norm)).exp(); 
    
    // Initialize vector of arrays to hold each cell's current dendrite state 
    std::vector<ArrayXXd> dendrite_states(n_neurons);
    ArrayXi max_post_lags = post_syn_lags.rowwise().maxCoeff().max(1);
    for (int i = 0; i < n_neurons; ++ i) {
      dendrite_states[i] = ArrayXXd::Zero(max_post_lags(i), n_neurons);
    }
    
    // Initialize array to track index of current time in each dendrite state matrix
    ArrayXi ds_now = ArrayXi::Zero(n_neurons);
    
    // Simulate each time step after the initial
    for (int t = 1; t < n_steps; ++t) {
      
      // Look up each pre-synaptic neuron's lagged last_spike value as seen by each post-synaptic neuron
      // ... ls_lagged(i,j) = last_spike of pre-syn j, as seen by post-syn i, accounting for conduction lag
      ArrayXXi ls_lagged = lagged_last_spike(t, pre_syn_lags, last_spike_history, ls_buffer_size);
     
      // Advance S: add 1 on spike arrival (onset only); hold S during spike width; decay after spike ends
      // ... onset:  ls_lagged(i,j) == tau_onset(i,j)  [== tau_spike(j) - 1, first step of the arriving spike]
      // ... active: ls_lagged(i,j) > 0                [spike still ongoing as seen by this synapse]
      // ... Accumulation above 1 is possible (supra-additive subthreshold effects from multiple spikes)
      auto active = (ls_lagged > 0).eval();
      S = active.select(
          S + (ls_lagged == tau_onset).cast<double>(),
          syn_decay * S
        );
      
      // Scale supra-additive portion of S by tA (per-neuron):
      //   tA = 0 → S capped at 1 (no supra-additive effect)
      //   tA = 1 → S accumulates freely (same behavior as before)
      //   0 < tA < 1 → excess above 1 is linearly attenuated
      ArrayXXd S_excess = (S - 1.0).max(0.0);
      S = (S - S_excess) + S_excess.colwise() * per_nrn.tA;
      
      // Compute synaptic and leak currents
      ArrayXXd v_drive = -(per_nrn.v_eq.colwise() - v_sub.col(t - 1));
      ArrayXXd I_syn   = v_drive * g_syn * S; 
      ArrayXd  I_leak  = per_nrn.g_leak * (v_sub.col(t - 1) - per_nrn.v_rest);
      
      /*
       * Dendritic computing model: 
       *  1. Same-site, over-time supra-additive effect handled above via S update (lagged_last_spike + tau_onset):
       *      S is incremented by 1 at each spike arrival (onset only), held during the spike width, then decays with syn_decay.
       *      Dependence on distance from soma is built in via adjustment of tau_syn, so that the supra-additive
       *      effect is strongest furthest from soma and gone near soma. 
       *  2. To handle different site, same-time supra-additive effect: after updating 
       *      dendrite_states[i].row(ds_now(i)) = I_syn.row(i), multiply row by a scalar determined by 
       *      the number of active synapses and distance from soma. per_nrn.tA is applied. 
       *  3. sublinear, supra-threshold effect: distance-dependent decay term based on recent spike count, applied 
       *      per pre-synaptic neuron as dendritice currents are accumulated. per_nrn.Ta is applied. 
       */
      
      // Apply dendritic computing to I_syn to find input "felt" at the soma
      ArrayXd I_syn_effective = ArrayXd::Zero(n_neurons); 
      // For each post-synaptic neuron i ...
      for (int i = 0; i < n_neurons; ++i) {
        
        // Get number of active synapses
        double n_syn_on = static_cast<double>((I_syn.row(i) != 0).count());
        // Compute super-additive effect 
        double tAe      = per_nrn.tA(i) * n_syn_on > 1.0 ? (n_syn_on - 1.0) / static_cast<double>(n_neurons) : 0.0;
        // Adjust for distance from soma and add 1
        auto   sae_adj  = (post_syn_travel_norm.row(i) * tAe + 1.0).eval();
        // Update dendrite state, with distance-adjusted supra-additive effect 
        dendrite_states[i].row(ds_now(i)) = I_syn.row(i) * sae_adj; 
        
        // Scale calcium concentration by distance to soma to estimate calcium at synapse
        auto Ca_adj = (Ca(i) * (1.0 - post_syn_travel_norm.row(i))).eval();
        // Find distant-dependent somatic-calcium supra-threshold sub-additive effect 
        auto Tae    = (1.0 - per_nrn.Ta(i) * Ca_adj).eval();
        
        // Initialize index vector
        ArrayXi ds_felt = ds_now(i) - post_syn_lags.row(i); 
        // For each pre-synaptic neuron ...
        for (int j = 0; j < n_neurons; ++j) {
          // Find index by subtracting lag from current time step, modulo max_post_lags
          // Double-modulo ensures a non-negative result even when ds_felt(j) < 0
          // (i.e., during the first max_post_lags steps before the buffer has full history)
          ds_felt(j) = ((ds_felt(j) % max_post_lags(i)) + max_post_lags(i)) % max_post_lags(i);
          // Add felt synaptic current from this synapse
          I_syn_effective(i) += dendrite_states[i](ds_felt(j), j) * Tae(j);
        }
      }
      // Advance ds_now, modulo max_post_lags 
      for (int i = 0; i < n_neurons; ++i) {
        ds_now(i)++;
        ds_now(i) = ds_now(i) % max_post_lags(i);
      }
      
      /*
       * Have: 
       * ... g_syn(i, j)                 = conductance from neuron j to neuron i
       * ... ls_lagged(i, j) = last_spike of pre-syn j as seen by post-syn i at this time step
       * ... so, row-wise sum of I_syn gives power dissipation from input into i
       */
      
      // Compute rate of change for total metabolic power dissipation in the network, w.r.t. each neuron
      // ... Units of dHdv are power/voltage: femto-Watts/mV = pA.
      ArrayXd dHdv = (
        I_syn_effective   +      // synaptic current (outward-positive); an excitatory (inward, negative) current lowers dHdv and depolarizes cell
        I_stim.col(t - 1) +      // injected stimulus current (negative = depolarizing)
        I_leak            +      // leak current (outward-positive): g * (v - v_rest)
        spikes * per_nrn.I_spike // spike: large outward (repolarizing) current at v_threshold, driving the reset
      ).min(per_nrn.dHdv_bound - std::numeric_limits<double>::epsilon());
      
      // For each neuron in network, at this time step, 
      // ... compute power to repolarize after a spike:
      ArrayXd spike_repolarization_power           = per_nrn.dHdv_bound * v_sub.col(t - 1);
      ArrayXd rest_maintenance_power               = dHdv * per_nrn.v_bound;
      ArrayXd spike_cost                           = spike_repolarization_power - rest_maintenance_power;  
      // ... compute max power to initiate a spike
      ArrayXd spike_repolarization_power_from_rest = per_nrn.dHdv_bound * per_nrn.v_bound;
      ArrayXd maintenance_power                    = dHdv * v_sub.col(t - 1);
      ArrayXd max_spike_cost                       = spike_repolarization_power_from_rest - maintenance_power; 
      // ... normalize spike cost
      ArrayXd normalized_spike_cost                = spike_cost / max_spike_cost; 
      
      // Multiple potential bound by normalized spike cost ... example units: mV * W/W = mV
      ArrayXd v_bound_fraction                     = per_nrn.v_bound * normalized_spike_cost; 
      
      // Set dvdt based on v_bound fraction
      ArrayXd dvdt                                 = v_bound_fraction - v_sub.col(t - 1);
      
      // Compute temporal modulation term T
      ArrayXd Ca_n            = slow_current - Ca;
      for (int i = 0; i < 2; ++i) Ca_n = Ca_n * Ca_n; 
      ArrayXd tau_slow_effect = slow_current * Ca_n / (Ca_n + theta_n);
      ArrayXd T               = (Vs * tau_slow_effect) / per_nrn.tau_fast;
      /*
       * T               = temporal modulation term, units of 1/ms
       * 
       * Vs              = synaptic vesicle concentration, models STD. Unitless, ratio [0,1].
       * tau_slow_effect = membrane response to slow currents, e.g., Ca2+ (calcium), for modeling bursting. Unitless, ratio [0,1].
       * tau_fast        = membrane response to fast currents, e.g., Na+ (sodium). Units of ms.
       * 
       * Ca              = Intracellular calcium concentrations (or, whatever molecule controls the slow current). Unitless, ratio [0,1]. 
       * theta           = threshold level (of Ca concentration) for 50% effect of slow current. Unitless, ratio [0,1]. 
       * 
       * If slow_current = 1, then Ca_n represents the n-th power of the remaining capacity (i.e., (1.0 - Ca)^n). 
       *  If it's instead zero, then Ca_n represents the n-th power of the remaining intracellular calcium, assuming 
       *  n is even.
       */
      
      // Apply T to dvdt 
      dvdt = (spikes == 1.0).select(
        dvdt,   // ... reset if immediately after a spike
        (last_spike > 0).select(
          ArrayXd::Zero(n_neurons),          // ... hold at rest if spike is on-going 
          dvdt * T * dt)
        );
      
      // Find new sub-threshold membrane potential by adding dvdt
      v_sub.col(t)         = ((v_sub.col(t - 1) + dvdt).max(-per_nrn.v_bound)).min(per_nrn.v_threshold);
      
      // Find spikes
      spikes               = (v_sub.col(t) >= per_nrn.v_threshold).cast<double>();
      // ... update spike counts
      spike_counts        += spikes;
      spike_counts_recent += spikes; 
      spike_counts_recent  = (spike_counts_recent - max_spike_rate_dt).max(0.0);
      last_spike          += spikes.cast<int>() * tau_spike;
      last_spike           = (last_spike - 1).max(0); 
      last_spike_history.col(t % ls_buffer_size) = last_spike;
      // ... update Vs and Ca
      Vs                  += dVdt(Vs, spike_counts_recent, per_nrn.dVdr, per_nrn.tau_Vs)   * dt; 
      Ca                  += dCdt(Ca, spike_counts_recent, per_nrn.dCdr, per_nrn.tau_slow) * dt;
      // ... and slow-current trigger 
      for (int i = 0; i < n_neurons; ++i) {
        if (slow_current(i)) {
          if (Ca(i) > theta_high) {
            slow_current(i) = 0.0; 
          }
        } else if (Ca(i) < theta_low) {
          slow_current(i) = 1.0;
        }
      }
      
      // Add spike to raw membrane potential and save to spike traces 
      v_traces.col(t) = v_sub.col(t) + spike_height * spikes;
      
    }
    
  }

/*
 * RCPP_MODULE to expose class to R
 */

RCPP_EXPOSED_CLASS(motif)
RCPP_MODULE(motif) {
  class_<motif>("motif")
  .constructor<std::string, int>()
  .method("load_projection", &motif::load_projection);
}

RCPP_EXPOSED_CLASS(network)
RCPP_MODULE(network) {
  class_<network>("network")
  .constructor()
  .method("set_network_structure", &network::set_network_structure)
  .method("make_local_nodes", &network::make_local_nodes)
  .method("apply_circuit_motif", &network::apply_circuit_motif)
  .method("fetch_network_components", &network::fetch_network_components)
  .method("fetch_sim_results", &network::fetch_sim_results)
  .method("BGT", &network::BGT);
}

RCPP_EXPOSED_CLASS(Prj)
RCPP_MODULE(Prj) {
  class_<Prj>("Prj")
  .constructor()
  .field("pre_type",      &Prj::pre_type)
  .field("pre_layer",     &Prj::pre_layer)
  .field("post_type",     &Prj::post_type)
  .field("post_layer",    &Prj::post_layer)
  .field("hem_shift",     &Prj::hem_shift)
  .field("via_apical",    &Prj::via_apical);
}
