
// DACx.cpp
#include "DACx.h"

/*
 * Sections: 
 * - Helper functions
 * - Growth-transform helper functions
 * - Matrix and vector operations
 * - Cell types and related functions
 * - Network and related classes
 * - Network (and related) member function implementations
 */

/*
 * ***********************************************************************************
 * Helper functions
 */

// Define a pseudo-infinity value
const double Inf = 1e20;

// Return logical vector giving elements of left which match right
LogicalVector Rmask(
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
LogicalVector Rmask(
    const CharacterVector& left,
    const std::string& right
  ) {
    int n = left.size();
    LogicalVector out(n);
    for (int i = 0; i < n; i++) {
      out[i] = left[i] == right;
    }
    return out;
  }
// ... overload
LogicalVector Rmask(
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
LogicalVector Rmask(
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
// ... overload
IntegerVector Rwhich(
    const std::vector<bool>& x
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
bool any_true(
    const std::vector<bool>& x
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
bool all_true(
    const std::vector<bool>& x
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
      lagged_power_dissipation -                // power dissipation (electrical current) from coupling between neurons
      stimulus_current +                        // power injected into the system (electrical current) from external stimulation
      v_barrier(v_traces, threshold, I_spike);  // power dissipated (electrical current) from neural responses (namely, spikes)
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

// Find first neighbor 
std::vector<int> find_first_neighbor(
    const std::vector<Vector3d>& b_active, // Branch searching for neighbor
    const std::vector<Vector3d>& b_all,    // Branch being searched
    const double& neighborhood_radius,
    const bool& skip_origin
  ) {
    double neighborhood_radius_squared = neighborhood_radius * neighborhood_radius;
    int i_initial = 0;
    if (skip_origin) {i_initial = 1;}
    for (int i = i_initial; i < b_active.size(); ++i) {
      for (int j = 0; j < b_all.size(); ++j) {
        double distance = (b_active[i] - b_all[j]).squaredNorm();
        if (distance <= neighborhood_radius_squared) {
          return {i, j}; // Return index of first neighbor found
        }
      }
    }
    return {-1, -1}; // Return -1 if no neighbor is found within the radius
  }

/*
 * ***********************************************************************************
 * Cell types and related functions
 */


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
  // Defaults ...
  // Membrane kinetics (burst control)
  double temporal_modulation_bias = 10;        // temporal modulation time (in unit_time) bias for each neuron type
  double temporal_modulation_timeconstant = 10;       // temporal modulation time (in unit_time) step for each neuron type
  double temporal_modulation_amplitude = 10;          // temporal modulation time (in unit_time) cutoff for each neuron type
  // Intercell transmission
  double transmission_velocity = 30e3;         // microns/ms ... 30 m/s = 30e6 micron/s = 30e6 micron/ 1e3 ms = 30e3 micron/ms
  double spine_density = 0.0;                  // Scale between 0 and 1: 0 = no nodes have spines, 1 = all nodes have spines
  std::string axon_target = "dendrite_shaft";  // "spine", "dendrite_shaft", "soma", and "axon_shaft"
  // Membrane characteristics
  double v_bound = 75.0;                       // potential bound, in unit_potential
  double dHdv_bound = 1.05e-3;                 // bound the derivative of metabolic energy wrt potential, in unit_current
  double I_spike = 1e-3;                       // spike current, in unit_current (by default, unit_current is a mA, so this is 1 micro amp)
  double spike_potential = 35.0;               // Magnitude of each spike, in unit_potential
  double resting_potential = -70.0;            // resting potential, in unit_potential
  double threshold = -55.0;                    // spike threshold, in unit_potential
  // Process size and structure parameters
  int axon_branch_count = 10;                  // Sets n_branches in make_arbor, in terms of expected number of branches per process length
  int dendrite_branch_count = 10;              // Sets n_branches in make_arbor, in terms of expected number of branches per process length
  double branch_independence = 0.5;            // Scale between 0 and 1; 0 = all branches connect to soma from single segment, 1 = all branches connect directly to soma
  double branch_spread = 0.5;                  // Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 = straight line away from soma
  // Apical dendrite parameters 
  std::string apical_target_layer = "none";

  // Define excitatory cells
  double tmb_fix = 0.0001;
  cell_types["pyramidal"] = cell_type{
    "pyramidal", 1,
    35.0*tmb_fix, temporal_modulation_timeconstant*tmb_fix, // Slow responders, 10-50 ms
    0.0, // No bursting
    transmission_velocity, 0.5, "spine",
    v_bound, dHdv_bound, I_spike,
    spike_potential, resting_potential, threshold,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 0.5, branch_spread * 0.5, // Reduced branching
    "L1" // Harris2013a, for cells in L2, L3, and L5
  };
  cell_types["pyramidal_L6"] = cell_type{
    "pyramidal_L6", 1,
    35.0*tmb_fix, temporal_modulation_timeconstant*tmb_fix,
    0.0, // No bursting
    transmission_velocity, 0.5, "spine",
    v_bound, dHdv_bound, I_spike,
    spike_potential, resting_potential, threshold,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 0.5, branch_spread * 0.5, // Reduced branching
    "L4" // Harris2013a: 
  };
  cell_types["spiny_stellate"] = cell_type{
    "spiny_stellate", 1,
    15.0*tmb_fix, temporal_modulation_timeconstant*tmb_fix,
    0.0, // No bursting
    transmission_velocity, 0.5, "spine",
    v_bound, dHdv_bound, I_spike,
    spike_potential, resting_potential, threshold,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.5, branch_spread * 1.5, // Increased branching
    apical_target_layer
  };
  // Define inhibitory cells
  cell_types["Neurogliaform_cell"] = cell_type{
    "Neurogliaform_cell", -1,
    temporal_modulation_bias*tmb_fix, temporal_modulation_timeconstant*tmb_fix,
    temporal_modulation_amplitude*tmb_fix,
    transmission_velocity * 0.5, spine_density, axon_target, // Slower transmission for neurogliaform cells
    v_bound, dHdv_bound, I_spike,
    spike_potential, resting_potential, threshold,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.5, branch_spread * 1.5, // Increased branching
    apical_target_layer
  };
  cell_types["PV"] = cell_type{
    "PV", -1,
    5.0*tmb_fix*2, temporal_modulation_timeconstant*tmb_fix, // Faster responders, 5 ms
    0.0, // No bursting
    transmission_velocity, spine_density, "soma",
    v_bound, dHdv_bound, I_spike,
    spike_potential, resting_potential, threshold,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.25, branch_spread * 1.25, // Increased branching
    apical_target_layer
  };
  cell_types["SST"] = cell_type{
    "SST", -1,
    10.0*tmb_fix, temporal_modulation_timeconstant*tmb_fix, // Slower responders, 10-30 ms
    30*tmb_fix,
    transmission_velocity, spine_density, axon_target,
    v_bound, dHdv_bound, I_spike,
    spike_potential, resting_potential, threshold,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.5, branch_spread * 1.5, // Increased branching
    apical_target_layer
  };
  cell_types["VIP"] = cell_type{
    "VIP", -1,
    15.0*tmb_fix, temporal_modulation_timeconstant*tmb_fix, // Slow responders, 15-40 ms
    25*tmb_fix,
    transmission_velocity, spine_density, axon_target,
    v_bound, dHdv_bound, I_spike,
    spike_potential, resting_potential, threshold,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.25, branch_spread * 1.25, // Increased branching
    apical_target_layer
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
                << "  Temporal modulation bias (ms): " << ct.temporal_modulation_bias << std::endl
                << "  Temporal modulation time constant (ms): " << ct.temporal_modulation_timeconstant << std::endl
                << "  Temporal modulation amplitude (ms): " << ct.temporal_modulation_amplitude << std::endl
                << "  Transmission velocity: " << ct.transmission_velocity << std::endl
                << "  Spine density: " << ct.spine_density << std::endl
                << "  Axon target: " << ct.axon_target << std::endl
                << "  Potential bound (mV): " << ct.v_bound << std::endl
                << "  Metabolic energy derivative dHdv bound (mA): " << ct.dHdv_bound << std::endl
                << "  Spike current (mA): " << ct.I_spike << std::endl
                << "  Spike potential (mV): " << ct.spike_potential << std::endl
                << "  Resting potential (mV): " << ct.resting_potential << std::endl
                << "  Threshold (mV): " << ct.threshold << std::endl
                << "  Axon branch count: " << ct.axon_branch_count << std::endl
                << "  Dendrite branch count: " << ct.dendrite_branch_count << std::endl
                << "  Branch independence: " << ct.branch_independence << std::endl
                << "  Branch spread: " << ct.branch_spread << std::endl
                << "  Apical target layer: " << ct.apical_target_layer << std::endl;
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
      Named("spine_density") = ct.spine_density,
      Named("axon_target") = ct.axon_target,
      Named("v_bound") = ct.v_bound,
      Named("dHdv_bound") = ct.dHdv_bound,
      Named("I_spike") = ct.I_spike,
      Named("spike_potential") = ct.spike_potential,
      Named("resting_potential") = ct.resting_potential,
      Named("threshold") = ct.threshold,
      Named("axon_branch_count") = ct.axon_branch_count,
      Named("dendrite_branch_count") = ct.dendrite_branch_count,
      Named("branch_independence") = ct.branch_independence,
      Named("branch_spread") = ct.branch_spread,
      Named("apical_target_layer") = ct.apical_target_layer
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
    const double& spine_density,                // Scale between 0 and 1: 0 = no nodes have spines, 1 = all nodes have spines
    const std::string& axon_target,             // "spine", "dendrite_shaft", "soma", and "axon_shaft"
    const double& v_bound,                      // potential bound, in unit_potential
    const double& dHdv_bound,                   // bound on dHdv, in unit_current
    const double& I_spike,                      // spike current, in unit_current
    const double& spike_potential,              // Magnitude of each spike, in unit_potential
    const double& resting_potential,            // resting potential, in unit_potential
    const double& threshold,                    // spike threshold, in unit_potential
    const int& axon_branch_count,               // Expected number of axon branches 
    const int& dendrite_branch_count,           // Expected number of dendrite branches 
    const double& branch_independence,          // Scale between 0 and 1; 0 = all branches connect to soma from single segment, 1 = all branches connect directly to soma
    const double& branch_spread,                // Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 = straight line away from soma
    const std::string& apical_target_layer      // For pyramidal cells, which layer their apical dendrites target
  ) {
    if (cell_types.find(type_name) != cell_types.end()) {
      Rcpp::stop("Cell type already exists in known cell types");
    } else {
      if (spine_density < 0.0 || spine_density > 1.0) {Rcpp::stop("spine_density must be between 0 and 1");}
      if (branch_independence < 0.0 || branch_independence > 1.0) {Rcpp::stop("branch_independence must be between 0 and 1");} 
      if (branch_spread < 0.0 || branch_spread > 1.0) {Rcpp::stop("branch_spread must be between 0 and 1");}
      cell_types[type_name] = cell_type{
        type_name, valence,
        temporal_modulation_bias, temporal_modulation_timeconstant,
        temporal_modulation_amplitude,
        transmission_velocity, spine_density, axon_target,
        v_bound, dHdv_bound, I_spike,
        spike_potential, resting_potential, threshold,
        axon_branch_count, dendrite_branch_count,
        branch_independence, branch_spread,
        apical_target_layer
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
    const double& spine_density,                // Scale between 0 and 1: 0 = no nodes have spines, 1 = all nodes have spines
    const std::string& axon_target,             // "spine", "dendrite_shaft", "soma", and "axon_shaft"
    const double& v_bound,                      // potential bound, in unit_potential
    const double& dHdv_bound,                   // bound on dHdv, in unit_current
    const double& I_spike,                      // spike current, in unit_current
    const double& spike_potential,              // Magnitude of each spike, in unit_potential
    const double& resting_potential,            // resting potential, in unit_potential
    const double& threshold,                    // spike threshold, in unit_potential
    const int& axon_branch_count,               // Expected number of axon branches 
    const int& dendrite_branch_count,           // Expected number of dendrite branches 
    const double& branch_independence,          // Scale between 0 and 1; 0 = all branches connect to soma from single segment, 1 = all branches connect directly to soma
    const double& branch_spread,                // Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 = straight line away from soma
    const std::string& apical_target_layer      // For pyramidal cells, which layer their apical dendrites target
  ) {
    if (cell_types.find(type_name) != cell_types.end()) {
      if (spine_density < 0.0 || spine_density > 1.0) {Rcpp::stop("spine_density must be between 0 and 1");}
      if (branch_independence < 0.0 || branch_independence > 1.0) {Rcpp::stop("branch_independence must be between 0 and 1");} 
      if (branch_spread < 0.0 || branch_spread > 1.0) {Rcpp::stop("branch_spread must be between 0 and 1");}
      cell_types[type_name].valence = valence;
      cell_types[type_name].temporal_modulation_bias = temporal_modulation_bias;
      cell_types[type_name].temporal_modulation_timeconstant = temporal_modulation_timeconstant;
      cell_types[type_name].temporal_modulation_amplitude = temporal_modulation_amplitude;
      cell_types[type_name].transmission_velocity = transmission_velocity;
      cell_types[type_name].spine_density = spine_density;
      cell_types[type_name].axon_target = axon_target;
      cell_types[type_name].v_bound = v_bound;
      cell_types[type_name].dHdv_bound = dHdv_bound;
      cell_types[type_name].I_spike = I_spike;
      cell_types[type_name].spike_potential = spike_potential;
      cell_types[type_name].resting_potential = resting_potential;
      cell_types[type_name].threshold = threshold;
      cell_types[type_name].axon_branch_count = axon_branch_count;
      cell_types[type_name].dendrite_branch_count = dendrite_branch_count;
      cell_types[type_name].branch_independence = branch_independence;
      cell_types[type_name].branch_spread = branch_spread;
      cell_types[type_name].apical_target_layer = apical_target_layer;
    } else {
      Rcpp::stop("Cell type not found in known cell types");
    }
  }

/*
 * ***********************************************************************************
 * Network and related classes
 */

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

/*
 * ***********************************************************************************
 * Network (and related) member function implementations
 */

void motif::load_projection(
    const Projection& proj,
    const int& max_up,
    const int& max_down,
    const double& proj_conductance
  ) {
    projections.push_back(proj);
    max_col_shift_up.push_back(max_up);
    max_col_shift_down.push_back(max_down);
    projection_conductance.push_back(proj_conductance);
    n_projections++;
  }

void network::set_network_structure(
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
    List            local_con
  ) {
   
    // Check layer names (needed for motifs)
    if (lyr_names.size() != n_lyr) {
      Rcpp::Rcout << "lyr_names size: " << lyr_names.size() << ", n_layers: " << n_lyr << std::endl;
      Rcpp::stop("Length of lyr_names must equal n_layers");
    }
    
    // Convert local synaptic conductance values from R List to std::vector<MatrixXd>
    if (local_con.size() != n_lyr) {
      Rcpp::Rcout << "local_con size: " << local_con.size() << ", n_layers: " << n_lyr << std::endl;
      Rcpp::stop("Length of local_con must equal n_layers");
    }
    for (int i = 0; i < local_con.size(); ++i) {
      NumericMatrix con_mat_r = local_con[i];
      local_conductance.push_back(to_eMat(con_mat_r));
    }
    
    // Load cell types 
    for (String nt : nrn_types) {
      std::string nts = nt;
      auto it = cell_types.find(nts);
      if (it == cell_types.end()) Rcpp::stop("Unknown neuron type: %s", nts);
      neuron_types.push_back((*it).second);
    }
    
    // Set other network parameters
    layer_names              = lyr_names;
    n_layers                 = n_lyr;
    n_columns                = n_cls;
    n_patches                = n_pch;
    layer_height             = lyr_height;
    column_diameter          = cls_diameter;
    segment_length           = seg_length;
    layer_separation_factor  = lyr_separation_factor;
    column_separation_factor = cls_separation_factor;
    patch_separation_factor  = pch_separation_factor;
    neurons_per_node         = to_eiMat(nrn_per_node);
    synaptic_neighborhood    = synaptic_neighborhood_radius;
    
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
    
    // Resize synapse index matrices and set all values to -1
    synapse_arbor_idx = MatrixXi::Constant(n_neurons, n_neurons, -1);
    synapse_node_idx  = MatrixXi::Constant(n_neurons, n_neurons, -1);
    
    // Convert neuron temporal modulation to Eigen matrix
    neuron_temporal_modulation        = MatrixXd::Zero(n_neurons, 3);
    neuron_temporal_modulation.col(0) = Map<VectorXd>(neuron_temporal_modulation_bias.data(),         neuron_temporal_modulation_bias.size());
    neuron_temporal_modulation.col(1) = Map<VectorXd>(neuron_temporal_modulation_timeconstant.data(), neuron_temporal_modulation_timeconstant.size());
    neuron_temporal_modulation.col(2) = Map<VectorXd>(neuron_temporal_modulation_amplitude.data(),    neuron_temporal_modulation_amplitude.size());
    
    // Convert neuron transmission delay to Eigen vector
    neuron_transmission_velocity = Map<VectorXd>(neuron_transmission_velocity_tmp.data(), neuron_transmission_velocity_tmp.size());
    
    // Resize network coordinate components 
    coordinates_spatial = MatrixXd::Zero(n_neurons, 3); 
    coordinates_node    = MatrixXi::Zero(n_neurons, 3); // patch (z), layer (y), column (x)
    
  };

// Function to make axon and dendrite branches
void network::make_arbor_branch(
    int             cell_idx,                           // Number of neuron for which to make processes
    bool            is_axon,                            // Whether to make axon (true) or dendrite (false)
    int             parent_branch_idx,                  // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
    const Vector3d& attractor_point                     // z (patch), y (layer), x (column); if all zeros, no attractor bias; otherwise, bias branch growth toward this point
  ) {
    
    // Check attractor point
    bool use_attractor = false;
    if (attractor_point(0) != 0.0 ||
        attractor_point(1) != 0.0 ||
        attractor_point(2) != 0.0) {
      use_attractor = true;
    }
    
    // Compute expected node radius 
    double expected_node_radious = compute_expected_node_radius();
    // Check segment divisor
    if (segment_length <= 0.0) {Rcpp::stop("segment length less than or equal to zero");}
    // Compute expected number of segments 
    int n_segments = (int)std::round(expected_node_radious / segment_length);
    if (n_segments < 2) {n_segments = 2;}
    // Randomly select the number of segments, ensuring at least 1
    n_segments = R::rpois(n_segments - 1) + 1;
   
    // Set parent flag 
    bool has_parent = parent_branch_idx >= 0;
    
    // Find initial point 
    Vector3d last_node;
    Vector3d soma_coordinates = coordinates_spatial.row(cell_idx);
    int parent_idx;
    if (has_parent) {
      // If child of parent branch, make sure parent exists and check axon flag
      if (parent_branch_idx >= arbors[cell_idx].axon.size()) {Rcpp::stop("Parent branch index exceeds number of branches in arbor");}
      if (is_axon != arbors[cell_idx].axon[parent_branch_idx]) {Rcpp::stop("Parent branch type (axon vs dendrite) does not match specified branch type for new branch");}
      // Randomly select branch point 
      int parent_branch_length = arbors[cell_idx].coordinates[parent_branch_idx].size();
      if (parent_branch_length == 0) {Rcpp::stop("Parent branch has no segments to branch from");}
      NumericVector probs(parent_branch_length);
      for (int i = 1; i <= parent_branch_length; ++i) {
        probs[i - 1] = 1.0 / (i*i);   // higher weight near 1
      }
      probs = probs / Rcpp::sum(probs); // Normalize to sum to 1
      int branch_point = Rcpp::sample(parent_branch_length, 1, true, probs)[0] - 1; // Rcpp::sample samples between 1 and its first argument, so subtract 1 for 0-indexing
      // ... and set as initial point
      last_node = arbors[cell_idx].coordinates[parent_branch_idx][branch_point];
      // ... ensure this point not marked as a leaf 
      arbors[cell_idx].leafs[parent_branch_idx][branch_point] = 0;
      // ... and set initial parent node idx
      parent_idx = branch_point;
    } else {
      // Set axon flag for new process arbor
      if (is_axon) {
        arbors[cell_idx].axon.push_back(true);
      } else {
        arbors[cell_idx].axon.push_back(false);
      }
      last_node = soma_coordinates;              // Set initial point as the soma location
      arbors[cell_idx].coordinates.push_back({last_node});  // Initialize new coordinates vector (of Vector3d) with first row as spatial coordinates of the cell body 
      arbors[cell_idx].node_type.push_back({"soma"});       // ... and initialize node_type vector and mark that this first point is "soma"
      arbors[cell_idx].parents.push_back({-1});             // ... and initialize new vector to track node parents
      arbors[cell_idx].leafs.push_back({0});                // ... and initialize leafs vector and mark that this first point is not a leaf 
      arbors[cell_idx].synapses.push_back({0});             // ... and initialize synapses vector and mark that this first point is not a synapse
      parent_branch_idx = arbors[cell_idx].axon.size() - 1; // ... set as parent branch 
      parent_idx = 0;                            // ... and set initial parent node idx 
    }
    
    // If using attractor, adjust expected number of segments
    double bias_component_magnitude_init; 
    if (use_attractor) {
      Vector3d bias_attractor_point = attractor_point - arbors[cell_idx].coordinates[parent_branch_idx].back();
      bias_component_magnitude_init = bias_attractor_point.norm();
      if (bias_component_magnitude_init == 0.0) {bias_component_magnitude_init = 1.0;}
      int n_segment_scalar = (int)std::round(bias_component_magnitude_init / expected_node_radious);
      if (n_segment_scalar < 1) {n_segment_scalar = 1;}
      n_segments *= n_segment_scalar;
      n_segments = Rcpp::sample(n_segments, 1)[0];
    }
    
    // Grab spine density and branch spread
    int t_num = neuron_type_num[cell_idx];
    double spine_density = neuron_types[t_num].spine_density;
    double branch_spread = neuron_types[t_num].branch_spread;
    
    // Make branch
    for (int s = 0; s < n_segments; s++) {
      
      // Make random component of the step
      Vector3d step = {
         R::rnorm(0.0, segment_length),  // z
         R::rnorm(0.0, segment_length),  // y
         R::rnorm(0.0, segment_length)   // x
      };
      double random_component_magnitude = step.norm();
      // ... bias step away from soma in proportion to branch spread
      Vector3d expand = last_node - soma_coordinates;
      // ... normalize expansion component so that it's the same magnitude as the random component and set weight with branch_spread, in proportion to distance from soma
      double weight_expand = branch_spread;
      double expand_component_magnitude = expand.norm();
      if (expand_component_magnitude > 0) {
        expand *= random_component_magnitude / expand_component_magnitude;
        weight_expand *= segment_length / expand_component_magnitude; 
        if (weight_expand > 0.9) {weight_expand = 0.9;} // Cap weighted branch spread at 0.9
        if (weight_expand <= 0.1) {weight_expand = 0.1;} // Ensure weighted branch spread is positive
      } else {
        expand = Vector3d::Zero(); // If last_node is exactly at the soma, there is no expansion component
      }
      // ... make weighted combination of the step and directed component
      step = (1 - weight_expand) * step + weight_expand * expand;
      
      // Make directed component of the step (z, y, x)
      if (use_attractor) {
        Vector3d bias = attractor_point - last_node;
        // ... normalize directed component so that it's the same magnitude as the random component
        double bias_component_magnitude = bias.norm();
        if (bias_component_magnitude > 0) {
          bias *= random_component_magnitude / bias_component_magnitude;
        } else {
          bias = Vector3d::Zero(); // If last_node is exactly at the attractor point, there is no bias component
        }
        // ... randomly select weight with expected value in proportion to distance to attractor point 
        double weight_bias = bias_component_magnitude / bias_component_magnitude_init;
        if (weight_bias > 0.9) {weight_bias = 0.9;} // Cap weight at 0.9
        if (weight_bias <= 0.1) {weight_bias = 0.1;} // Ensure weight is positive
        // ... make weighted combination of the step and bias 
        step = (1 - weight_bias) * step + weight_bias * bias;
      }
      
      // Add the step to the previous segment's coordinates to get the new segment's coordinates, and add to arbor coordinates
      Vector3d new_node = last_node + step;
      if (new_node.array().isNaN().any()) {Rcpp::stop("NaN detected in Vector3d");}
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
        if (R::runif(0.0, 1.0) < spine_density) {
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
      
    }
    
  }

// Function to make axon and dendrite arbors
void network::make_arbor(
    int                          cell_idx,              // Number of neuron for which to make processes
    int                          n_branches,            // Expected number of branches, including the main process 
    bool                         is_axon,               // Whether to make axon (true) or dendrite (false)
    int                          parent_branch_idx,     // Index of parent branch, if this is a branch off of a main process; otherwise, -1 for new process arbor
    const std::vector<Vector3d>& attractor_points       // z (patch), y (layer), x (column); if all zeros, no attractor bias; otherwise, bias branch growth toward this point
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
      // Set arbor ID number 
      arbors[cell_idx].arbor_id.push_back(n_existing_arbors);
      // Grab branch independence 
      int t_num = neuron_type_num[cell_idx];
      double branch_independence = neuron_types[t_num].branch_independence;
      // Track number of additional new branches
      int n_new_branches = 0;
      for (int b = 1; b < n_branches; ++b) {
        if (R::runif(0.0, 1.0) < branch_independence) {
          n_new_branches++;
          parent_branch_idx_list.push_back(-1);
          arbors[cell_idx].arbor_id.push_back(n_existing_arbors + n_new_branches);
        } else {
          int parent_idx = n_existing_arbors;
          parent_idx += (int)(Rcpp::sample(n_new_branches + 1, 1)[0] - 1); 
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
    
    // For each branch to-be-made: 
    int n_attractor_points = attractor_points.size();
    for (int b = 0; b < n_branches; ++b) {
      
      // Make random linear combination of attractor points 
      Vector3d attractor_point = Vector3d::Zero();
      double remaining_weight = 1.0;
      double w = 0.0;
      for (int i = 0; i < n_attractor_points; i++) {
        if (i + 1 < n_attractor_points) {
          w = R::runif(0.0, remaining_weight);
          remaining_weight -= w;
        } else {
           w = remaining_weight; // Ensure all weights sum to 1
        }
        attractor_point += attractor_points[i] * w;
      }
      
      // Grab parent branch index for this branch
      int parent_branch_idx_b = parent_branch_idx_list[b];
      
      // Make arbor branch
      make_arbor_branch(
        cell_idx,
        is_axon,
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
    
    // Compute expected node radius 
    double expected_node_radious = compute_expected_node_radius();
    
    // Initialize local transconductance matrix
    MatrixXd local_transconductances = MatrixXd::Zero(n_neurons, n_neurons);
    
    // Patch index of the local node 
    for (int p = 0; p < n_patches; ++p) {
      
      // Layer index of the local node
      for (int l = 0; l < n_layers; ++l) {
        
        // Get recurrence factor matrix for this layer
        MatrixXd local_conductance_matrix = local_conductance[l];
        
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
          
          // Make local process arbors for all cells in this node
          for (int cell_idx = node_range_start; cell_idx <= node_range_end; cell_idx++) {
            
            // Set spatial coordinates
            coordinates_spatial(cell_idx, 0) = node_z + R::rnorm(0.0, column_diameter/2.0);
            coordinates_spatial(cell_idx, 1) = node_y + R::rnorm(0.0, layer_height/2.0);
            coordinates_spatial(cell_idx, 2) = node_x + R::rnorm(0.0, column_diameter/2.0);
            
            // Set node coordinates
            coordinates_node(cell_idx, 0) = p;
            coordinates_node(cell_idx, 1) = l;
            coordinates_node(cell_idx, 2) = c;
            
            // Get neuron types 
            int t_num = neuron_type_num[cell_idx];
            int axon_branch_count = neuron_types[t_num].axon_branch_count;
            int dendrite_branch_count = neuron_types[t_num].dendrite_branch_count;
            std::string apical_target_layer = neuron_types[t_num].apical_target_layer;
           
            // Create local axon arbor
            make_arbor(cell_idx, axon_branch_count, true);
            
            // Create local dendrite arbor
            make_arbor(cell_idx, dendrite_branch_count, false);
            
            // Create apical dendrite, if any 
            if (apical_target_layer != "none") {
              LogicalVector layer_mask = Rmask(layer_names, apical_target_layer);
              if (any_true(layer_mask)) {
                // MB: Much of this code duplicated from apply_motif; create function??
                
                // Get coordinates of the target node 
                int apical_target_layer_idx = Rwhich(layer_mask)[0];
                // ... reconstruct target node ID number
                int target_node_idx = p * (n_layers * n_columns) + apical_target_layer_idx * n_columns + c;
                // ... get spatial position of this node, z (patch), y (layer), x (column)
                Vector3d attractor_point = node_coordinates_spatial.row(target_node_idx); 
                std::vector<Vector3d> attractor_points = {attractor_point};
                
                // Make apical dendrite arbor
                make_arbor(
                  cell_idx, 
                  dendrite_branch_count, 
                  false,
                  -1, // should be a new dendrite
                  attractor_points 
                );
                
              } else {
                Rcpp::Rcout << "Apical target layer: " << apical_target_layer << ", layer names: " << layer_names << std::endl;
                Rcpp::stop("Apical target layer not found in layer names");
              }
              
            }
            
          }
          
          // For all combinations of pre- and post-synaptic neurons in this node
          for (int idx_pre = node_range_start; idx_pre <= node_range_end; idx_pre++) {
            
            // Get neuron types for pre-synaptic neurons
            int t_pre = neuron_type_num[idx_pre];
            
            // Get neuron valences for pre-synaptic neurons
            double val_pre = neuron_types[t_pre].valence;
            
            // Set transconductance into post-synaptic cells
            for (int idx_post = node_range_start; idx_post <= node_range_end; idx_post++) {
              
              // Get neuron types for post-synaptic neurons
              int t_post = neuron_type_num[idx_post];
              
              // Search for synapse and compute its transconductance
              double this_transconductance = find_synapse(
                idx_pre, 
                idx_post, 
                val_pre, 
                local_conductance_matrix(t_post, t_pre)
              );
              
              if (this_transconductance > 0) {
                // Set transductance 
                local_transconductances(idx_post, idx_pre) = this_transconductance;
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

// Function to find synapses and return transconductance into post-synaptic cell 
double network::find_synapse(
    int    idx_pre,
    int    idx_post,
    double val_pre,
    double transductance_bias
  ) {
   
    // Check if this pre-post pair already has a synapse
    if (synapse_arbor_idx(idx_pre, idx_post) >= 0) {return(0.0);} 
    
    // Get axon indices 
    IntegerVector axon_idx = Rwhich(arbors[idx_pre].axon);
    
    // Get post-synaptic dendrite branch indices
    std::vector<bool> dendrite_mask = arbors[idx_post].axon; 
    for (int i = 0; i < dendrite_mask.size(); i++) {dendrite_mask[i] = !dendrite_mask[i];}
    IntegerVector dendrite_idx = Rwhich(dendrite_mask);
    
    // Check all axons 
    for (int ax : axon_idx) {
      
      // Check all dendrites 
      for (int dd : dendrite_idx) {
        
        // Check for synapses 
        // ... first element of neighbor_idx is the axon node idx, second is the dendrite node idx
        std::vector<int> neighbor_idx = find_first_neighbor(
          arbors[idx_pre].coordinates[ax], 
          arbors[idx_post].coordinates[dd],
          synaptic_neighborhood
        );
        
        // If one is found, create it
        if (neighbor_idx[0] >= 0) {
          
          // Extend the axon
          // ... add coordinates
          arbors[idx_pre].coordinates[ax].push_back(arbors[idx_post].coordinates[dd][neighbor_idx[1]]);
          // ... add parent 
          if (neighbor_idx[0] >= arbors[idx_pre].coordinates[ax].size()) {Rcpp::stop("Neighbor index out of bounds for axon coordinates");}
          arbors[idx_pre].parents[ax].push_back(neighbor_idx[0]);
          // ... add node type 
          arbors[idx_pre].node_type[ax].push_back("axon_shaft");
          // ... ensure old node not marked as leaf 
          arbors[idx_pre].leafs[ax][neighbor_idx[0]] = 0;
          // ... and mark new node as leaf
          arbors[idx_pre].leafs[ax].push_back(1);
          // ... and mark new node as synapse
          arbors[idx_pre].synapses[ax].push_back(1);
          // ... and mark in the synapse idx matrices 
          synapse_arbor_idx(idx_pre, idx_post) = ax;
          synapse_node_idx(idx_pre, idx_post) = arbors[idx_pre].coordinates[ax].size() - 1;
         
          // Find and return transductance 
          double trans = R::runif(0.0, 2.0) * transductance_bias;
          return(trans);
          
        }
        
      }
      
    }
    
    return(0.0);
    
  }

// Compute expected node radius 
double network::compute_expected_node_radius() {
  double lh = layer_height * layer_separation_factor / 2.0;
  double cd = column_diameter * column_separation_factor / 2.0;
  double pd = column_diameter * patch_separation_factor / 2.0;
  return std::sqrt(lh*lh + cd*cd + pd*pd);
}

// Function to apply circuit motif
void network::apply_circuit_motif(
    const motif& cmot,
    bool         verbose
  ) {
    
    if (verbose) {
      Rcpp::Rcout << "Applying motif: " << cmot.motif_name << std::endl;
    }
   
    if (edge_types.size() < 1) {
      Rcpp::stop("Must set local edges before applying any circuit motifs.");
    }
    
    // Initialize vectors to track motif edge coordinates
    std::vector<int> motif_edges_pre; 
    std::vector<int> motif_edges_post;
    
    // Initialize motif transconductance matrix
    MatrixXd motif_transconductances = MatrixXd::Zero(n_neurons, n_neurons);
    
    // Pre-make all column masks 
    LogicalMatrix column_masks(n_neurons, n_columns);
    for (int c = 0; c < n_columns; c++) {
      column_masks(_, c) = Rmask(coordinates_node(Eigen::all,2), c); // column 2 is the column
    }
    
    // Pre-make all patch masks 
    LogicalMatrix patch_masks(n_neurons, n_patches);
    for (int p = 0; p < n_patches; p++) {
      patch_masks(_, p) = Rmask(coordinates_node(Eigen::all,0), p); // column 0 is the patch
    }
    
    // Compute expected node radius 
    double expected_node_radious = compute_expected_node_radius();
    
    // For each projection in the motif
    const int n_projections = cmot.n_projections;
    for (int pj = 0; pj < n_projections; pj++) {
      
      // Grab projection
      Projection proj = cmot.projections[pj];
      
      // Grab pre- and post-synaptic cell types for this projection
      std::string pre_type_name = proj.pre_type;
      std::string post_type_name = proj.post_type;
      cell_type pre_type = cell_types.at(pre_type_name);
      cell_type post_type = cell_types.at(post_type_name);
      // Get indices for neuron_types in this network
      CharacterVector type_names(neuron_types.size());
      for (int i = 0; i < neuron_types.size(); i++) {type_names[i] = neuron_types[i].type_name;}
      LogicalVector pre_type_exists = Rmask(type_names, pre_type_name);
      LogicalVector post_type_exists = Rmask(type_names, post_type_name);
      if (!(any_true(pre_type_exists) && any_true(post_type_exists))) {
        if (verbose) {
          Rcpp::Rcout << "Projection " << pj << " in motif " << cmot.motif_name << " has pre- or post-synaptic type that does not exist in this network." << std::endl
                      << "  Pre-synaptic type: " << pre_type_name << ", post-synaptic type: " << post_type_name << std::endl
                      << "  Available types: " << type_names << std::endl
                      << "  ...  skipping this projection." << std::endl;
        }
        continue;
      }
      int t_pre = Rwhich(pre_type_exists)[0];
      int t_post = Rwhich(post_type_exists)[0];
      // ... and make masks for neurons in this network
      LogicalVector pre_type_mask = Rmask(neuron_type_num, t_pre);
      LogicalVector post_type_mask = Rmask(neuron_type_num, t_post);
      
      // Grab pre-synaptic valence and axon characteristics
      int val_pre = neuron_types[t_pre].valence;
      int axon_branch_count = neuron_types[t_pre].axon_branch_count;
      
      // Grab dendrite characteristics for post-synaptic cells
      int dendrite_branch_count = neuron_types[t_post].dendrite_branch_count;
      
      // Grab pre and post layers
      LogicalVector pre_layer_exists = Rmask(layer_names, proj.pre_layer);
      LogicalVector post_layer_exists = Rmask(layer_names, proj.post_layer);
      if (!(any_true(pre_layer_exists) && any_true(post_layer_exists))) {
        if (verbose) {
          Rcpp::Rcout << "Projection " << 
          pj << " in motif " << 
          cmot.motif_name << " has layer not in network; skipping projection." << std::endl;
        }
        continue;
      }
      int layer_pre = Rwhich(pre_layer_exists)[0];
      int layer_post = Rwhich(post_layer_exists)[0];
      // ... and make masks 
      LogicalVector pre_layer_mask = pre_type_mask & Rmask(coordinates_node(Eigen::all,1), layer_pre); // column 1 is the layer
      LogicalVector post_layer_mask = post_type_mask & Rmask(coordinates_node(Eigen::all,1), layer_post);
      
      // Grab max shifts
      int max_up = cmot.max_col_shift_up[pj];
      int max_down = cmot.max_col_shift_down[pj];
      VectorXi col_range = VectorXi::Constant(2, 0);
      col_range(0) = -max_down;
      col_range(1) = max_up;
      
      // Apply projection to each patch
      for (int p = 0; p < n_patches; p++) {
        
        // Get pre-synaptic patch mask 
        LogicalVector pre_patch_mask = pre_layer_mask & patch_masks(_, p);
        
        // Apply projection to each column 
        for (int c = 0; c < n_columns; c++) {
          
          // Print progress
          if (verbose) {
            Rcpp::Rcout 
              << "Applying projection: " << pj << " / " << n_projections 
              << ", column: " << c << " / " << n_columns 
              << ", patch " << p << " / " << n_patches 
            << std::endl;
          }
          
          // Get pre-synaptic column mask
          LogicalVector pre_column_mask = column_masks(_, c);
          
          // Find indexes of pre-synaptic cells 
          LogicalVector pre_mask = pre_patch_mask & pre_column_mask;
          if (!any_true(pre_mask)) {continue;} // Skip if no pre-synaptic cells of the right type in this column and layer
          IntegerVector pre_indices = Rwhich(pre_mask);
          
          // Get coordinates of home node, z (patch), y (layer), x (column)
          int home_node_idx = p * (n_layers * n_columns) + layer_pre * n_columns + c;
          Vector3d home_node_coordinates = node_coordinates_spatial.row(home_node_idx);
          
          // Shift range to this column and patch
          VectorXi col_range_shifted;
          VectorXi patch_range_shifted;
          if (col_range[0] == col_range[1]) {
            col_range_shifted.resize(1);
            col_range_shifted(0) = c;
            patch_range_shifted.resize(1);
            patch_range_shifted(0) = p;
          } else {
            col_range_shifted = col_range.array() + c;
            patch_range_shifted = col_range.array() + p;
            // ... ensure target columns and patches are valid
            if (col_range_shifted[0] < 0) {col_range_shifted[0] = 0;}
            if (col_range_shifted[1] >= n_columns) {col_range_shifted[1] = n_columns - 1;}
            if (patch_range_shifted[0] < 0) {patch_range_shifted[0] = 0;}
            if (patch_range_shifted[1] >= n_patches) {patch_range_shifted[1] = n_patches - 1;}
          }
          
          // Reconstruct all attractor points and build masks 
          LogicalVector post_mask(n_neurons, false);
          std::vector<Vector3d> attractor_points; 
          for (int tp : patch_range_shifted) {
            for (int tc : col_range_shifted) {
              // Don't make local connections 
              if (layer_pre == layer_post && c == tc && p == tp) {continue;}
              // Get coordinates of the target node 
              // ... reconstruct target node ID number
              int target_node_idx = tp * (n_layers * n_columns) + layer_post * n_columns + tc;
              // ... get spatial position of this node, z (patch), y (layer), x (column)
              Vector3d attractor_point = node_coordinates_spatial.row(target_node_idx);
              attractor_points.push_back(attractor_point);
              // Add masks
              post_mask = post_mask | (post_layer_mask & patch_masks(_, tp) & column_masks(_, tc));
            }
          }
          
          if (!any_true(post_mask)) {continue;} // Skip if no post-synaptic cells of the right type in this column and layer
          IntegerVector post_indices = Rwhich(post_mask);
          
          // Make mesoscale axon arbors for these cells
          for (int cell_idx : pre_indices) {
            
            // Select random axon to extend
            int axon_parent; 
            std::vector<bool> axon_mask = arbors[cell_idx].axon; 
            IntegerVector axon_idx = Rwhich(axon_mask);
            if (axon_idx.size() > 0) {
              axon_parent = Rcpp::sample(axon_idx, 1)[0];
            } else {
              axon_parent = -1; // if no axon branches yet, start from the soma
            }
            
            // Make axon
            make_arbor(
              cell_idx, 
              axon_branch_count, 
              true, 
              axon_parent, 
              attractor_points
            );
            
          }
          
          // For all combinations of pre- and post-synaptic neurons in this projection
          for (int idx_pre : pre_indices) {
            // Set transconductance into post-synaptic cells
            for (int idx_post : post_indices) {
              
              // Search for synapse and compute its transconductance
              double this_transconductance = find_synapse(
                idx_pre, 
                idx_post, 
                val_pre, 
                cmot.projection_conductance[pj]
              );
              
              // If one is found, create it
              if (this_transconductance > 0) {
                // Set transductance 
                motif_transconductances(idx_post, idx_pre) = val_pre * this_transconductance;
                // Save edge coordinate
                motif_edges_pre.push_back(idx_pre);
                motif_edges_post.push_back(idx_post);
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

// Method to fetch network components 
List network::fetch_network_components(
    bool include_arbors
  ) const {
   
    // Convert transconductances into list of NumericMatrix
    int n_transconductance_matrices = transconductances.size();
    List transconductance_matrices(n_transconductance_matrices);
    if (n_transconductance_matrices > 0) {
      for (int tci = 0; tci < transconductances.size(); tci++) {
        MatrixXd tc = transconductances[tci];
        NumericMatrix tc_r = to_NumMat(tc);
        transconductance_matrices[tci] = tc_r;
      } 
      transconductance_matrices.names() = edge_type_names; 
    }
    
    // Convert edge_types into list of NumericMatrix
    List edge_type_matrices(edge_types.size());
    CharacterVector emn = CharacterVector::create("pre_neuron_idx", "post_neuron_idx");
    for (int eti = 0; eti < edge_types.size(); eti++) {
      MatrixXi et = edge_types[eti];
      NumericMatrix et_r = to_NumMat(et);
      for (double& v : et_r) v++; // put into 1-indexed form for R
      colnames(et_r) = emn;
      edge_type_matrices[eti] = et_r;
    }
    
    // Convert arbors into numeric matrix
    NumericMatrix arbor_matrix;
    if (include_arbors) {
      // Count total number of segments across all arbors
      int n_segments = 0;
      int n_roots = 0;
      for (int n = 0; n < n_neurons; n++) {
        int n_arbors = arbors[n].axon.size();
        n_roots += n_arbors;
        for (int a = 0; a < n_arbors; a++) {
          n_segments += arbors[n].coordinates[a].size();
        }
      }
      
      // Create matrix to hold arbor data
      arbor_matrix = NumericMatrix(n_segments - n_roots, 13);
      colnames(arbor_matrix) = CharacterVector::create(
        "neuron_idx", "arbor_id", "is_axon", "node_type", "parent_idx", "is_leaf", "is_synapse", 
        "z_start", "y_start", "x_start", "z_end", "y_end", "x_end"
      );
      
      // Fill matrix with arbor data
      int seg_idx = 0;
      int parent_skip_counter = 0;
      for (int n = 0; n < n_neurons; n++) {
        int n_arbors = arbors[n].axon.size();
        for (int a = 0; a < n_arbors; a++) {
          int n_segs = arbors[n].coordinates[a].size();
          double arbor_type = arbors[n].axon[a] ? 1.0 : 0.0;               // 1 for axon, 0 for dendrite
          // ... for each segment endpoint i
          for (int i = 0; i < n_segs; i++) { 
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
            if (parent_idx < 0) {                                               // Skip root node since it has no parent
              parent_skip_counter++;
              continue;
            }                                     
            arbor_matrix(seg_idx, 0) = (double)n + 1;                           // neuron index, put into 1-indexed form for R
            arbor_matrix(seg_idx, 1) = (double)arbors[n].arbor_id[a];           // arbor ID number
            arbor_matrix(seg_idx, 2) = arbor_type;
            arbor_matrix(seg_idx, 3) = node_type_idx;                           // 0 = "soma", 1 = "dendrite_shaft", 2 = "axon_shaft", or 3 = "spine"
            arbor_matrix(seg_idx, 4) = parent_idx - parent_skip_counter + 1;    // put into 1-indexed form for R
            arbor_matrix(seg_idx, 5) = (double)arbors[n].leafs[a][i];           // 1 if node is a leaf, zero if not
            arbor_matrix(seg_idx, 6) = (double)arbors[n].synapses[a][i];        // 1 if node is a synapse, zero if not
            arbor_matrix(seg_idx, 7) = arbors[n].coordinates[a][parent_idx][0]; // z_start
            arbor_matrix(seg_idx, 8) = arbors[n].coordinates[a][parent_idx][1]; // y_start
            arbor_matrix(seg_idx, 9) = arbors[n].coordinates[a][parent_idx][2]; // x_start
            arbor_matrix(seg_idx, 10) = arbors[n].coordinates[a][i][0];         // z_end
            arbor_matrix(seg_idx, 11) = arbors[n].coordinates[a][i][1];         // y_end
            arbor_matrix(seg_idx, 12) = arbors[n].coordinates[a][i][2];         // x_end
            seg_idx++;
          }
        }
      }
    } else {
      arbor_matrix = NumericMatrix(1, 1);
    }
    
    // Add labels 
    NumericMatrix coordinates_node_R = to_NumMat(coordinates_node);
    if (coordinates_node_R.size() > 0) {colnames(coordinates_node_R) = CharacterVector::create("patch_idx", "layer_idx", "column_idx");}
    NumericMatrix coordinates_spatial_R = to_NumMat(coordinates_spatial);
    if (coordinates_spatial.size() > 0) {colnames(coordinates_spatial_R) = CharacterVector::create("z", "y", "x");}
    NumericMatrix node_coordinates_spatial_R = to_NumMat(node_coordinates_spatial);
    if (node_coordinates_spatial.size() > 0) {colnames(node_coordinates_spatial_R) = CharacterVector::create("z", "y", "x");}
    
    // Put into 1-indexed form
    for (double& v : coordinates_node_R) v++;
    
    return List::create(
      _["network_name"] = network_name,
      _["n_neurons"] = n_neurons,
      _["n_neuron_types"] = n_neuron_types,
      _["n_nodes"] = node_range_ends.size(),
      _["n_layers"] = n_layers,
      _["n_columns"] = n_columns,
      _["n_patches"] = n_patches,
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
      _["arbors"] = arbor_matrix, 
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

// Function to compute pairwise lags based on axon path lengths and membrane velocity
MatrixXi network::find_pairwise_lags_by_axon(
    double dt // time step length, in unit_time
  ) {
    
    // Initialize matrix to hold pairwise lags, with default value of 0 
    MatrixXi pairwise_lags = MatrixXi::Constant(n_neurons, n_neurons, 0);
    
    // Precompute reciprocals
    const VectorXd inv_vel = neuron_transmission_velocity.cwiseInverse();
    const double inv_dt = 1.0 / dt;
    
    // For each neuron, find the lag to each other neuron based on axonal path length, synapse location, and transmission velocity
    for (int idx_pre = 0; idx_pre < n_neurons; ++idx_pre) {
     
      // Confirm there are axons for this cell
      if (!any_true(arbors[idx_pre].axon)) {continue;}
      
      // For each post-synaptic neuron, check for synapses and set lag if found
      for (int idx_post = 0; idx_post < n_neurons; ++idx_post) {
        
        // Get axon and node idx
        int axon_idx = synapse_arbor_idx(idx_pre, idx_post);
        if (axon_idx < 0) {continue;} // No synapse from this pre to this post
        int node_idx = synapse_node_idx(idx_pre, idx_post);
        
        // Grab all nodes along the axon and their parents
        std::vector<Vector3d> axon_coordinates = arbors[idx_pre].coordinates[axon_idx];
        std::vector<int> axon_node_parents = arbors[idx_pre].parents[axon_idx];
        double dist = 0;
        int parent_node_idx = axon_node_parents[node_idx];
        while (parent_node_idx >= 0) {
          Vector3d node = axon_coordinates[node_idx];
          Vector3d parent_node = axon_coordinates[parent_node_idx];
          dist += (node - parent_node).norm();
          node_idx = parent_node_idx; 
          parent_node_idx = axon_node_parents[node_idx];
        }
        
        // Convert distance into simulation time-step lag
        const double lag = dist * inv_vel[idx_pre] * inv_dt;
        pairwise_lags(idx_pre, idx_post) = static_cast<int>(std::round(lag));
        
      }
      
    }
    
    return pairwise_lags;
    
  }


// Simulate network responses to input current using Growth Transform model
void network::SGT(
    const NumericMatrix& stimulus_current_R, // matrix of stimulus currents, in unit_current, n_neurons x n_steps
    double               dt                  // time step length, in unit_time
  ) {
    
    /*
     * Dev note: The stimulus_current matrix is only used in dHdv, one time step at a time. Hence, when 
     *   the thalamus model is implemented, it can take the form of a function that takes auditory input 
     *   and outputs a vector of stimulus currents. The thalamus model can be called each time step to produce 
     *   the input currents to each cell. The thalamus model could also have an argument which takes feedback from 
     *   the cortex. 
     */
    
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
    MatrixXi pair_lags = find_pairwise_lags_by_axon(dt);
    
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
    
    // Initialize count to keep track of time since last spike, for bursting
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
      
      // Set dvdt based on v_bound fraction
      VectorXd dvdt_unmodulated = v_bound_fraction - v_traces.col(t - 1);
      
      // Find temporal modulation for this time step with vectorized operations
      VectorXd neuron_temporal_modulation = 
        neuron_temporal_modulation_bias.array() + 
        neuron_temporal_modulation_amplitude.array() - 
        neuron_temporal_modulation_amplitude.array() * (-burst_step_counter.array() / neuron_temporal_modulation_timeconstant.array()).exp();
      
      // Find dvdt by dividing by the temporal modulation
      VectorXd dvdt = dt * dvdt_unmodulated.array() / neuron_temporal_modulation.array();
      
      // Find new subthreshold membrane potential
      VectorXd v_subthreshold = v_traces.col(t - 1) + dvdt; 
      // ... save for next step
      v_traces.col(t) = v_subthreshold;
      
      // Divide spike_potential (minus threshold) by spike current to get transimpedance value necessary for that spike potential
      VectorXd transimpedance = (spike_potential - threshold).array()/I_spike.array();
      
      // Find spike value
      VectorXd barrier_values = v_barrier(v_subthreshold, threshold, I_spike);
      VectorXd spike = transimpedance.array() * barrier_values.array(); 
      // ... update burst counter and spike counts
      VectorXd spikes = (barrier_values.array() / I_spike.array()).matrix();
      burst_step_counter.array() *= (1.0 - spikes.array()); // reset burst counter to zero if spike, otherwise keep counting up
      burst_step_counter.array() += dt;
      spike_counts.array() += spikes.array();
      
      // Add spike to raw membrane potential and save to spike traces 
      sim_traces.col(t) = v_subthreshold + spike;
      
    }
    
  }

/*
 * RCPP_MODULE to expose class to R
 */

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
  .field("post_type",     &Projection::post_type)
  .field("post_layer",    &Projection::post_layer);
}