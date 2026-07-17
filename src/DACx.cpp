
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
    const VectorXd& threshold,      // Spike threshold in mV, for each neuron in network
    const VectorXd& I_out           // Spike current in pA, for each neuron in network
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
    const MatrixXd& v_lagged,         // n_neuron x n_neuron matrix giving membrane potentials in mV, with each column j giving the membrane potentials of all neurons as seen by neuron j at this time step
    const VectorXd& v,                // n_neuron x 1 matrix (column vector) of membrane potentials in mV, from which to calculate derivative
    const VectorXd& membrane_current, // n_neuron x 1 matrix (column vector) of membrane currents in pA, from which to calculate derivative
    const MatrixXd& transconductance, // n_neuron x n_neuron transconductance matrix, giving connections between neurons in nS
    const VectorXd& I_spike,          // spike current in pA
    const VectorXd& threshold         // spike threshold in mV
  ) {  
    // Change dH in total dissipated metabolic power in network (a current) from small change dv in membrane potential, 
    //  given the membrane potential at time step n, for each neuron in network
    //  ... Notice that this function implies that row indices represent post-synaptic neurons, column indices represent pre-synaptic neurons
    VectorXd lagged_synaptic_power_dissipation = (transconductance.array() * v_lagged.transpose().array()).rowwise().sum();
    // ... transconductance(i, j) = conductance from neuron j to neuron i
    // ... v_lagged(i, j)  = neuron i's membrane potential as seen by neuron j at this time step
    // ... v_lagged.transpose()(i, j) = neuron j's membrane potential as seen by neuron i at this time step
    // ... so, row-wise sum gives power dissipation from input into i
    VectorXd dHdv = 
      lagged_synaptic_power_dissipation -       // power dissipation (electrical current) from coupling between neurons
      membrane_current +                        // power dissipation (electrical current) from current across the membrane (negative because in-flowing positive current takes energy to pump out)
      v_barrier(v, threshold, I_spike);         // power dissipation (electrical current) from neural responses (namely, spikes)
    return dHdv;
    
    /*
     * transconductance * v_lagged >>>
     *      (rows are post-synaptic neuron, columns are pre-synaptic neuron) >>>
     *        transconductance row i * v_lagged col j = input into neuron i from all other neurons.
     * ... so, need v_lagged to be a matrix, with each column j giving the membrane potentials of all neurons as seen by neuron j at this time step.
     * ... then the relevant output is the diagonal of the output matrix. 
     *      so, compute only (transconductance.cwiseProduct(v_lagged.transpose())).rowwise().sum()
     * ... How do I make the v_lagged matrix? 
     * ... Need to know, for each neuron i, how many time steps it takes the soma potential of neuron j to reach neuron i (for all j). 
     * ... Time for i to reach j, lag(i, j) = distance(i, j)/conduction_velocity(i), rounded to nearest time step.
     * ... v_lagged(n).col(j)(i) = neuron i's membrane potential at time step n - lag(i, j)
     * ... v_lagged(n).col(j)(i) = v(i, n - lag(i, j));
     */
    
  }

// Derivative of intracellular slow-current molecule concentrations
VectorXd dCadt(
    const VectorXd& Ca,                 // Vector of intracellular slow-current molecule (e.g., calcium) concentrations, per cell
    const VectorXd& recent_spike_count, // Vector of counts of recent spikes, per cell; proxy for spike rate (spikes/ms)
    const VectorXd& I_slow,             // Vector giving the slow-current molecule (e.g., Ca2+) influx as concentration per spike (concentration/spike)
    const VectorXd& tau_slow            // Vector giving time constant for clear calcium, per cell (ms)
  ) {
    return I_slow.cwiseProduct(recent_spike_count) - Ca.cwiseQuotient(tau_slow);
    // Returns concentration/ms
  }

// Derivative of synaptic vesicle concentrations (synaptic depression), from Schiff & Reyes 2012 (https://doi.org/10.1152/jn.00208.2011) 
VectorXd dVsdt(
    const VectorXd& Vs,                 // Vector of synaptic vesicle concentrations, per cell (ratio, [0,1])
    const VectorXd& recent_spike_count, // Vector of counts of recent spikes, per cell; proxy for spike rate (spikes/ms)
    const VectorXd& U_Vs,               // Vector giving the utilization ratio (concentration/spike) of vesicles per spike
    const VectorXd& tau_Vs              // Vector giving time constant for recovery from depression, per cell (ms)
  ) {
    return (1.0 - Vs.array()).matrix().cwiseQuotient(tau_Vs) - Vs.cwiseProduct(recent_spike_count.cwiseProduct(U_Vs));
    // Returns concentration/ms
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


/*
 * Cell type registry: Meyers singleton pattern.
 * Access via get_cell_types() everywhere; do NOT declare a bare global.
 * Defaults are constructed once in make_default_cell_types() on first call.
 *
 * To use or modify cell types:
 *   const auto& ct = get_cell_types().at("PV");
 *   double cutoff = ct.tau_slow;
 *   get_cell_types()["PV"].I_slow = 0.03;
 */

static std::unordered_map<std::string, cell_type> make_default_cell_types() {
  std::unordered_map<std::string, cell_type> ct_map;

  // Default shared values
  double tau_fast                 = 1.0;
  double tau_slow                 = 60.0;
  double tau_Vs                   = 100.0;   // ms/spike
  double I_slow                   = 0.01; // Default is no bursting; increase to induce bursting (or lower tau_slow)
  double U_Vs                     = 0.05; 
  double max_spike_rate           = 0.1;   // spikes/ms
  double transmission_velocity    = 30e3;  // microns/ms
  double spine_density            = 0.0;
  std::string axon_target         = "dendrite_shaft";
  double I_spike                  = 1e3;   // pA
  double spike_potential          = 35.0;  // mV
  double resting_potential        = -70.0; // mV
  double threshold                = -55.0; // mV
  double leak_conductance         = 10.0;  // nS
  int    axon_branch_count        = 10;
  int    dendrite_branch_count    = 10;
  double branch_independence      = 0.5;
  double branch_spread            = 0.5;
  std::string apical_target_layer = "none";

  // Excitatory cells
  ct_map["pyramidal"] = cell_type{ // Slow responders, 10-50 ms, No bursting 
    "pyramidal", 1,
    tau_fast, tau_slow, tau_Vs, I_slow, U_Vs, max_spike_rate,
    transmission_velocity, 0.5, "spine",
    I_spike, spike_potential, resting_potential, threshold, leak_conductance,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 0.5, branch_spread * 0.5, // Reduced branching
    "L1" // Harris2013a, for cells in L2, L3, and L5
  };
  ct_map["pyramidal_L6"] = cell_type{ // No bursting 
    "pyramidal_L6", 1,
    tau_fast, tau_slow, tau_Vs, I_slow, U_Vs, max_spike_rate,
    transmission_velocity, 0.5, "spine",
    I_spike, spike_potential, resting_potential, threshold, leak_conductance,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 0.5, branch_spread * 0.5, // Reduced branching
    "L4" // Harris2013a
  };
  ct_map["spiny_stellate"] = cell_type{ // No bursting 
    "spiny_stellate", 1,
    tau_fast, tau_slow, tau_Vs, I_slow, U_Vs, max_spike_rate,
    transmission_velocity, 0.5, "spine",
    I_spike, spike_potential, resting_potential, threshold, leak_conductance,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.5, branch_spread * 1.5, // Increased branching
    apical_target_layer
  };
  // Inhibitory cells
  ct_map["Neurogliaform_cell"] = cell_type{
    "Neurogliaform_cell", -1,
    tau_fast, tau_slow, tau_Vs, I_slow * 3.5, U_Vs, max_spike_rate, // bursting
    transmission_velocity * 0.5, spine_density, axon_target, // Slower transmission
    I_spike, spike_potential, resting_potential, threshold, leak_conductance,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.5, branch_spread * 1.5, // Increased branching
    apical_target_layer
  };
  ct_map["PV"] = cell_type{ // Faster responders, ~5 ms; No bursting
    "PV", -1,
    tau_fast, tau_slow, tau_Vs, I_slow, U_Vs, max_spike_rate,
    transmission_velocity, spine_density, "soma",
    I_spike, spike_potential, resting_potential, threshold, leak_conductance,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.25, branch_spread * 1.25, // Increased branching
    apical_target_layer
  };
  ct_map["SST"] = cell_type{ // Slower responders, 10-30 ms
    "SST", -1,
    tau_fast, tau_slow, tau_Vs, I_slow * 3.5, U_Vs, max_spike_rate,
    transmission_velocity, spine_density, axon_target,
    I_spike, spike_potential, resting_potential, threshold, leak_conductance,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.5, branch_spread * 1.5, // Increased branching
    apical_target_layer
  };
  ct_map["VIP"] = cell_type{ // Slow responders, 15-40 ms
    "VIP", -1,
    tau_fast, tau_slow, tau_Vs, I_slow * 3.5, U_Vs, max_spike_rate,
    transmission_velocity, spine_density, axon_target,
    I_spike, spike_potential, resting_potential, threshold, leak_conductance,
    axon_branch_count, dendrite_branch_count,
    branch_independence * 1.25, branch_spread * 1.25, // Increased branching
    apical_target_layer
  };

  return ct_map;
}

// Meyers singleton: initialized once on first call, never reset by .onLoad
std::unordered_map<std::string, cell_type>& get_cell_types() {
  static std::unordered_map<std::string, cell_type> cell_types = make_default_cell_types();
  return cell_types;
}

// Internal helper: unpack a fully-specified named R List into a cell_type struct
cell_type build_cell_type_from_list(const List& params) {
  cell_type ct;
  ct.type_name             = as<std::string>(params["type_name"]);
  ct.valence               = as<int>(        params["valence"]);
  ct.tau_fast              = as<double>(     params["tau_fast"]);
  ct.tau_slow              = as<double>(     params["tau_slow"]);
  ct.tau_Vs                = as<double>(     params["tau_Vs"]);
  ct.I_slow                = as<double>(     params["I_slow"]);
  ct.U_Vs                  = as<double>(     params["U_Vs"]); 
  ct.max_spike_rate        = as<double>(     params["max_spike_rate"]);
  ct.transmission_velocity = as<double>(     params["transmission_velocity"]);
  ct.spine_density         = as<double>(     params["spine_density"]);
  ct.axon_target           = as<std::string>(params["axon_target"]);
  ct.I_spike               = as<double>(     params["I_spike"]);
  ct.spike_potential       = as<double>(     params["spike_potential"]);
  ct.resting_potential     = as<double>(     params["resting_potential"]);
  ct.threshold             = as<double>(     params["threshold"]);
  ct.leak_conductance      = as<double>(     params["leak_conductance"]);
  ct.axon_branch_count     = as<int>(        params["axon_branch_count"]);
  ct.dendrite_branch_count = as<int>(        params["dendrite_branch_count"]);
  ct.branch_independence   = as<double>(     params["branch_independence"]);
  ct.branch_spread         = as<double>(     params["branch_spread"]);
  ct.apical_target_layer   = as<std::string>(params["apical_target_layer"]);
  if (ct.spine_density < 0.0 || ct.spine_density > 1.0)
    Rcpp::stop("spine_density must be between 0 and 1");
  if (ct.branch_independence < 0.0 || ct.branch_independence > 1.0)
    Rcpp::stop("branch_independence must be between 0 and 1");
  if (ct.branch_spread < 0.0 || ct.branch_spread > 1.0)
    Rcpp::stop("branch_spread must be between 0 and 1");
  return ct;
}

// Print known cell types 
// [[Rcpp::export]]
void print_known_celltypes() {
  Rcpp::Rcout << "Known cell types:" << std::endl;
  for (const auto& pair : get_cell_types()) {
    const cell_type& ct = pair.second;
    Rcpp::Rcout << "\nType: "                                   << ct.type_name << std::endl
                << "  Valence: "                                << ct.valence << std::endl
                << "  Temporal modulation bias (ms): "          << ct.tau_fast << std::endl
                << "  Temporal modulation time constant (ms): " << ct.I_slow << std::endl
                << "  U_Vs: "                                   << ct.U_Vs << std::endl
                << "  Temporal modulation amplitude (ms): "     << ct.tau_slow << std::endl
                << "  Spike recovery rate (spikes/ms): "        << ct.max_spike_rate << std::endl
                << "  STD recovery time constant (spikes/ms): " << ct.tau_Vs << std::endl
                << "  Transmission velocity: "                  << ct.transmission_velocity << std::endl
                << "  Spine density: "                          << ct.spine_density << std::endl
                << "  Axon target: "                            << ct.axon_target << std::endl
                << "  Spike current (pA): "                     << ct.I_spike << std::endl
                << "  Spike potential (mV): "                   << ct.spike_potential << std::endl
                << "  Resting potential (mV): "                 << ct.resting_potential << std::endl
                << "  Threshold (mV): "                         << ct.threshold << std::endl
                << "  Leak conductance (nS): "                  << ct.leak_conductance << std::endl
                << "  Axon branch count: "                      << ct.axon_branch_count << std::endl
                << "  Dendrite branch count: "                  << ct.dendrite_branch_count << std::endl
                << "  Branch independence: "                    << ct.branch_independence << std::endl
                << "  Branch spread: "                          << ct.branch_spread << std::endl
                << "  Apical target layer: "                    << ct.apical_target_layer << std::endl;
  }
}

// Fetch cell type parameters 
// [[Rcpp::export]]
List fetch_cell_type_params(const std::string& type_name) {
  auto it = get_cell_types().find(type_name);
  if (it == get_cell_types().end()) {
    Rcpp::stop("Cell type not found in known cell types");
  }
  const cell_type& ct = it->second;
  return List::create(
    Named("type_name")             = ct.type_name,
    Named("valence")               = ct.valence,
    Named("tau_fast")              = ct.tau_fast,
    Named("tau_slow")              = ct.tau_slow,
    Named("tau_Vs")                = ct.tau_Vs,
    Named("I_slow")                = ct.I_slow,
    Named("U_Vs")                  = ct.U_Vs, 
    Named("max_spike_rate")        = ct.max_spike_rate,
    Named("transmission_velocity") = ct.transmission_velocity,
    Named("spine_density")         = ct.spine_density,
    Named("axon_target")           = ct.axon_target,
    Named("I_spike")               = ct.I_spike,
    Named("spike_potential")       = ct.spike_potential,
    Named("resting_potential")     = ct.resting_potential,
    Named("threshold")             = ct.threshold,
    Named("leak_conductance")      = ct.leak_conductance,
    Named("axon_branch_count")     = ct.axon_branch_count,
    Named("dendrite_branch_count") = ct.dendrite_branch_count,
    Named("branch_independence")   = ct.branch_independence,
    Named("branch_spread")         = ct.branch_spread,
    Named("apical_target_layer")   = ct.apical_target_layer
  );
}

// Add a new cell type; params must be a fully-specified named List
// [[Rcpp::export]]
void add_cell_type(const List& params) {
  cell_type ct = build_cell_type_from_list(params);
  if (get_cell_types().count(ct.type_name))
    Rcpp::warning("Cell type already exists in known cell types, overriding");
  get_cell_types()[ct.type_name] = ct;
}

// Modify an existing cell type; params must be a fully-specified named List
// (NULL-substitution for unspecified fields is handled on the R side)
// [[Rcpp::export]]
void modify_cell_type(const std::string& type_name, const List& params) {
  if (!get_cell_types().count(type_name))
    Rcpp::stop("Cell type not found in known cell types");
  get_cell_types()[type_name] = build_cell_type_from_list(params);
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
network::network() { 
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

// Expand all cell type scalar parameters into per-neuron Eigen vectors/matrix.
// Called once from set_network_structure after n_neurons and neuron_type_num are populated.
void network::resize_neuron_params() {
    v_bound               = VectorXd(n_neurons);
    dHdv_bound            = VectorXd(n_neurons);
    I_spike               = VectorXd(n_neurons);
    spike_potential       = VectorXd(n_neurons);
    resting_potential     = VectorXd(n_neurons);
    threshold             = VectorXd(n_neurons);
    leak_conductance      = VectorXd(n_neurons);
    max_spike_rate        = VectorXd(n_neurons);
    tau_fast              = VectorXd(n_neurons); 
    tau_slow              = VectorXd(n_neurons); 
    tau_Vs                = VectorXd(n_neurons);
    I_slow                = VectorXd(n_neurons); 
    U_Vs                  = VectorXd(n_neurons); 
    transmission_velocity = VectorXd(n_neurons);
    for (int i = 0; i < n_neurons; i++) {
        const cell_type& ct = neuron_types[neuron_type_num[i]];
        v_bound(i)               = std::abs(ct.resting_potential) * 1.01;
        dHdv_bound(i)            = std::abs(ct.I_spike)           * 1.01;
        I_spike(i)               = ct.I_spike;
        spike_potential(i)       = ct.spike_potential;
        resting_potential(i)     = ct.resting_potential;
        threshold(i)             = ct.threshold;
        leak_conductance(i)      = ct.leak_conductance;
        max_spike_rate(i)        = ct.max_spike_rate;
        tau_fast(i)              = ct.tau_fast; 
        tau_slow(i)              = ct.tau_slow; 
        tau_Vs(i)                = ct.tau_Vs;
        I_slow(i)                = ct.I_slow; 
        U_Vs(i)                  = ct.U_Vs; 
        transmission_velocity(i) = ct.transmission_velocity;
    }
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
      auto it = get_cell_types().find(nts);
      if (it == get_cell_types().end()) Rcpp::stop("Unknown neuron type: %s", nts);
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
    int n_nodes    = n_layers * n_columns * n_patches;
    n_neurons      = 0; // Compute total number of neurons as we go
    node_range_ends.assign(n_nodes, 0);
    node_coordinates_spatial.resize(n_nodes, 3);
    for (int p = 0; p < n_patches; p++) {
      for (int l = 0; l < n_layers; l++) {
        for (int c = 0; c < n_columns; c++) {
          int node_idx = p * (n_layers * n_columns) + l * n_columns + c;
          // Set global spatial coordinates for this node
          node_coordinates_spatial(node_idx, 0) = p * column_diameter/2.0 * patch_separation_factor;   // z
          node_coordinates_spatial(node_idx, 1) = l * layer_height   /2.0 * layer_separation_factor;   // y
          node_coordinates_spatial(node_idx, 2) = c * column_diameter/2.0 * column_separation_factor;  // x
          for (int t = 0; t < n_neuron_types; t++) {
            // Randomly select neuron numbers for each node
            int n = (int)R::rpois(neurons_per_node(l,t));
            // Keep track of the number of cells assigned so far
            n_neurons += n;
            // Record type identity for each cell
            for (int i = 0; i < n; i++) {
              neuron_type_name.push_back(neuron_types[t].type_name);
              neuron_type_num.push_back(t);
            }
          }
          // Save end-point index for this node
          node_range_ends[node_idx] = n_neurons - 1;
        }
      }
    }
    
    // Expand all cell type parameters into per-neuron vectors (now that n_neurons is known)
    resize_neuron_params();
    
    // Set length of the vectors holding cell processes
    arbors.resize(n_neurons);
    
    // Resize synapse index matrices and set all values to -1
    synapse_arbor_idx = MatrixXi::Constant(n_neurons, n_neurons, -1);
    synapse_node_idx  = MatrixXi::Constant(n_neurons, n_neurons, -1);
    
    // Resize network coordinate components 
    coordinates_spatial = MatrixXd::Zero(n_neurons, 3); 
    coordinates_node    = MatrixXi::Zero(n_neurons, 3); // patch (z), layer (y), column (x)
    
  }

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
      last_node = soma_coordinates;                         // Set initial point as the soma location
      arbors[cell_idx].coordinates.push_back({last_node});  // Initialize new coordinates vector (of Vector3d) with first row as spatial coordinates of the cell body 
      arbors[cell_idx].node_type.push_back({"soma"});       // ... and initialize node_type vector and mark that this first point is "soma"
      arbors[cell_idx].parents.push_back({-1});             // ... and initialize new vector to track node parents
      arbors[cell_idx].leafs.push_back({0});                // ... and initialize leafs vector and mark that this first point is not a leaf 
      arbors[cell_idx].synapses.push_back({0});             // ... and initialize synapses vector and mark that this first point is not a synapse
      parent_branch_idx = arbors[cell_idx].axon.size() - 1; // ... set as parent branch 
      parent_idx = 0;                                       // ... and set initial parent node idx 
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
  double lh = layer_height    * layer_separation_factor  / 2.0;
  double cd = column_diameter * column_separation_factor / 2.0;
  double pd = column_diameter * patch_separation_factor  / 2.0;
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
      cell_type pre_type = get_cell_types().at(pre_type_name);
      cell_type post_type = get_cell_types().at(post_type_name);
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
      int layer_pre  = Rwhich(pre_layer_exists)[0];
      int layer_post = Rwhich(post_layer_exists)[0];
      // ... and make masks 
      LogicalVector pre_layer_mask  = pre_type_mask  & Rmask(coordinates_node(Eigen::all,1), layer_pre); // column 1 is the layer
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
    NumericMatrix coordinates_node_R         = to_NumMat(coordinates_node);
    if (coordinates_node_R.size()       > 0) {colnames(coordinates_node_R)         = CharacterVector::create("patch_idx", "layer_idx", "column_idx");}
    NumericMatrix coordinates_spatial_R      = to_NumMat(coordinates_spatial);
    if (coordinates_spatial.size()      > 0) {colnames(coordinates_spatial_R)      = CharacterVector::create("z", "y", "x");}
    NumericMatrix node_coordinates_spatial_R = to_NumMat(node_coordinates_spatial);
    if (node_coordinates_spatial.size() > 0) {colnames(node_coordinates_spatial_R) = CharacterVector::create("z", "y", "x");}
    
    // Put into 1-indexed form
    for (double& v : coordinates_node_R) v++;
    
    return List::create(
      _["n_neurons"]                = n_neurons,
      _["n_neuron_types"]           = n_neuron_types,
      _["n_nodes"]                  = node_range_ends.size(),
      _["n_layers"]                 = n_layers,
      _["n_columns"]                = n_columns,
      _["n_patches"]                = n_patches,
      _["layer_names"]              = layer_names, 
      _["transconductances"]        = transconductance_matrices,
      _["node_coordinates_spatial"] = node_coordinates_spatial_R,
      _["coordinates_spatial"]      = coordinates_spatial_R,
      _["coordinates_node"]         = coordinates_node_R,
      _["neuron_type_name"]         = neuron_type_name,
      _["neuron_type_num"]          = neuron_type_num,
      _["node_range_ends"]          = node_range_ends,
      _["edge_idx_by_type"]         = edge_type_matrices, 
      _["edge_type_names"]          = edge_type_names,
      _["sim_dt"]                   = sim_dt,
      _["arbors"]                   = arbor_matrix
    );
   
  }

// Method to fetch SGT simulation results 
List network::fetch_sim_results() const {
    return List::create(
      _["slow_current_traces"]    = slow_current_traces,
      _["Ca_traces"]              = Ca_traces, 
      _["tau_slow_effect_traces"] = tau_slow_effect_traces, 
      _["Vs_traces"]              = Vs_traces, 
      _["T_traces"]               = T_traces, 
      _["v_traces"]               = v_traces,
      _["spike_counts"]           = spike_counts
    );
  }

// Function to compute pairwise lags based on axon path lengths and membrane velocity
MatrixXi network::find_pairwise_lags_by_axon(
    double dt // time step length in ms
  ) {
    
    // Initialize matrix to hold pairwise lags, with default value of 0 
    MatrixXi pairwise_lags = MatrixXi::Constant(n_neurons, n_neurons, 0);
    
    // Precompute reciprocals
    const VectorXd inv_vel = transmission_velocity.cwiseInverse();
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
    const NumericMatrix& stimulus_current_R,  // matrix of stimulus currents in pA, n_neurons x n_steps
    double dt,                                // time step length in ms; units: ms/step
    double initial_potential                  // start all neurons with this membrane potential
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
    MatrixXi pair_lags = find_pairwise_lags_by_axon(dt);
    
    // Resize matrix to hold simulated spike traces (membrane potential plus spike)
    v_traces.resize(n_neurons, n_steps);
    v_traces.setZero();
    v_traces.col(0).setConstant(initial_potential);
    
    // Resize matrices to hold temporal modulation terms 
    T_traces.resize(n_neurons, n_steps);
    T_traces.setOnes(); 
    Vs_traces.resize(n_neurons, n_steps);
    Vs_traces.setOnes(); 
    tau_slow_effect_traces.resize(n_neurons, n_steps);
    tau_slow_effect_traces.setOnes(); 
    Ca_traces.resize(n_neurons, n_steps);
    Ca_traces.setZero(); 
    slow_current_traces.resize(n_neurons, n_steps); 
    slow_current_traces.setOnes(); 
    
    // Initialize matrix to hold simulated sub-threshold membrane potential traces (without spike)
    MatrixXd v_sub = MatrixXd::Zero(n_neurons, n_steps);
    v_sub.col(0).setConstant(initial_potential);
    
    // Resize spike_counts vector
    spike_counts.resize(n_neurons);
    spike_counts.setZero();
    // ... and make a local copy for tracking recent spikes (proxy for spikes/ms)
    VectorXd spike_counts_recent = VectorXd::Zero(n_neurons);
    // ... initialize vector to keep track of spikes each time step
    VectorXd spikes              = VectorXd::Zero(n_neurons);
    
    // Divide spike_potential (minus threshold) by spike current to get transimpedance value necessary for that spike potential
    VectorXd transimpedance      = (spike_potential - threshold).cwiseQuotient(I_spike);
    
    // Convert max_spike_rate to simulation time steps 
    VectorXd max_spike_rate_dt   = max_spike_rate * dt;
    
    // Set threshold for membrane response to slow currents 
    double theta_low             = 0.1;
    double theta_high            = 0.9; 
    VectorXd theta_low_n         = VectorXd::Constant(n_neurons, std::pow(theta_low, 4.0));
    VectorXd theta_high_n        = VectorXd::Constant(n_neurons, std::pow(theta_high, 4.0));
    
    // Initialize vector to hold Schmitt trigger for whether Ca levels are rising or falling
    // ... = 1 if + flowing in, = 0 if + being pushed out
    VectorXd slow_current(n_neurons);
    
    // Set initial slow current and threshold
    slow_current.setOnes(); 
    VectorXd theta_n = theta_low_n; 
    
    // Initialize vectors to hold synaptic vesicle and intracellular calcium concentrations
    VectorXd Vs(n_neurons); 
    VectorXd Ca(n_neurons); 
    Vs.setOnes(); 
    Ca.setZero(); 
    
    // Simulate each time step after the initial
    for (int t = 1; t < n_steps; t++) {
      
      // Compute each cell's membrane potential state (rows) as seen by each other cell (columns)
      MatrixXd v_sub_lagged     = lagged_traces(t, pair_lags, v_sub);
     
      // Add leak current to stimulus current to get total membrane current 
      VectorXd membrane_current = stimulus_current.col(t - 1) + leak_conductance.cwiseProduct(resting_potential - v_sub.col(t - 1));
      
      // Compute rate of change for total metabolic power dissipation in the network, w.r.t. each neuron
      // ... units of dHdv are power/voltage, e.g., pico-Watts/mV = nA
      // ... key idea? if a change dv in voltage in any one cell causes a spike, then H increases as well
      VectorXd dHdv = network_power_dissipation_gradient(
        v_sub_lagged,
        v_sub.col(t - 1), 
        membrane_current, 
        transconductances_sum, 
        I_spike, 
        threshold
      );
      
      // For each neuron in network, at this time step, 
      // ... compute power to initiate a spike:
      VectorXd spike_initiation_power           = dHdv_bound.cwiseProduct(v_sub.col(t - 1));
      VectorXd rest_maintenance_power           = dHdv.cwiseProduct(v_bound);
      VectorXd spike_cost                       = spike_initiation_power - rest_maintenance_power;  
      // ... compute max power to initiate a spike
      VectorXd spike_initiation_power_from_rest = dHdv_bound.cwiseProduct(v_bound);
      VectorXd maintenance_power                = dHdv.cwiseProduct(v_sub.col(t - 1));
      VectorXd max_spike_cost                   = spike_initiation_power_from_rest - maintenance_power; 
      // ... normalize spike cost
      VectorXd normalized_spike_cost            = spike_cost.array()/max_spike_cost.array(); 
      
      // Multiple potential bound by normalized spike cost ... example units: mV * W/W = mV
      VectorXd v_bound_fraction                 = v_bound.cwiseProduct(normalized_spike_cost); 
      
      // Set dvdt based on v_bound fraction
      VectorXd dvdt                             = v_bound_fraction - v_sub.col(t - 1);
      
      // Compute temporal modulation term T
      VectorXd Ca_n            = slow_current - Ca;
      for (int i = 0; i < 2; ++i) Ca_n = Ca_n.cwiseProduct(Ca_n); 
      VectorXd tau_slow_effect = Ca_n.cwiseQuotient(Ca_n + theta_n);
      VectorXd T               = (Vs.cwiseProduct(tau_slow_effect)).cwiseQuotient(tau_fast);
      /*
       * T               = temporal modulation term, units of 1/ms
       * 
       * Vs              = synaptic vesicle concentration, models STD. Unitless, ratio [0,1].
       * tau_slow_effect = membrane response to slow currents, e.g., Ca2+ (calcium), for modeling bursting. Unitless. 
       * tau_fast        = membrane response to fast currents, e.g., Na+ (sodium). Units of ms.
       * 
       * Ca              = Intracellular calcium concentrations (or, whatever molecule controls the slow current). Unitless, ratio [0,1]. 
       * theta           = threshold level (of Ca concentration) for 50% effect of slow current. Unitless, ratio [0,1]. 
       * 
       * If slow_current = 1, then Ca_n represents the n-th power of the remaining capacity (i.e., (1.0 - Ca)^n). 
       *  If it's instead zero, then Ca_n represents the n-th power of the remaining intracellular calcium, assuming 
       *  n is even.
       */
      
      // Apply T to dvdt if not immediately after a spike
      dvdt                          = (spikes.array() == 1.0).select(dvdt, dvdt.cwiseProduct(T) * dt);
      
      // Find new subthreshold membrane potential by adding dvdt
      v_sub.col(t)                  = v_sub.col(t - 1) + dvdt;
      
      // Find spike barrier values
      VectorXd barrier_values       = v_barrier(v_sub.col(t), threshold, I_spike);
      // ... update spike counts
      spikes = (barrier_values.array() != 0.0).cast<double>();
      spike_counts.array()         += spikes.array();
      spike_counts_recent.array()  += spikes.array(); 
      spike_counts_recent           = (spike_counts_recent - max_spike_rate_dt).array().max(0.0);
      // ... update Vs and Ca
      Vs                           += dVsdt(Vs, spike_counts_recent, U_Vs, tau_Vs)     * dt; 
      Ca                           += dCadt(Ca, spike_counts_recent, I_slow, tau_slow) * dt;
      // ... and slow-current trigger 
      for (int i = 0; i < n_neurons; ++i) {
        if (slow_current(i)) {
          if (Ca(i) > theta_high) {
            slow_current(i) = 0.0; 
            theta_n(i)      = theta_high_n(i); 
          }
        } else if (Ca(i) < theta_low) {
          slow_current(i) = 1.0;
          theta_n(i)      = theta_low_n(i); 
        }
      }
      
      // Add spike to raw membrane potential and save to spike traces 
      v_traces.col(t)               = v_sub.col(t) + transimpedance.cwiseProduct(barrier_values);
      
      // Save temporal modulation terms 
      T_traces.col(t)               = T;
      Vs_traces.col(t)              = Vs;
      tau_slow_effect_traces.col(t) = tau_slow_effect;
      Ca_traces.col(t)              = Ca;
      slow_current_traces.col(t)    = slow_current;
      
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
  .constructor()
  .method("set_network_structure", &network::set_network_structure)
  .method("make_local_nodes", &network::make_local_nodes)
  .method("apply_circuit_motif", &network::apply_circuit_motif)
  .method("fetch_network_components", &network::fetch_network_components)
  .method("fetch_sim_results", &network::fetch_sim_results)
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
