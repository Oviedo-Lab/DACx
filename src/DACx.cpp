
// DACx.cpp
#include "DACx.h"

/*
 * Sections: 
 * - Helper functions
 * - Probability distribution functions
 * - Growth-transform helper functions
 * - Matrix and vector operations
 * - Network and related classes
 * - Network member function implementations
 */

/*
 * ***********************************************************************************
 * Helper functions
 */

// Define a pseudo-infinity value
const double Inf = 1e20;

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

// Find first neighbor 
std::vector<int> find_first_neighbor(
    const std::vector<Vector3d>& b_active, // Branch searching for neighbor
    const std::vector<Vector3d>& b_all,    // Branch being searched
    const double& neighborhood_radius,
    const bool& skip_origin = true
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
MatrixXi pairwise_lags_by_edges(
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
 * Network member function implementations
 */

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
    double synaptic_neighborhood_radius
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
    synaptic_neighborhood = synaptic_neighborhood_radius;
    
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
    
    // Resize synapse index matrix and set all values to 0
    synapse_idx = MatrixXi::Constant(n_neurons, n_neurons, 0);
    
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
    
    // Initialize pointer to appropriate cell_arbors structure
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
      // ... ensure it's not a synapse
      int safety_counter = 0; // To prevent infinite loop
      while (arbor.synapses[parent_branch_idx][branch_point] == 1) {
        branch_point = Rcpp::sample(parent_branch_length, 1)[0] - 1;
        safety_counter++;
        if (safety_counter > 1000) {
          Rcpp::stop("Unable to find a branch point that is not a synapse after 1000 iteractions");
        }
      }
      // ... and set as initial point
      initial_point = arbor.coordinates[parent_branch_idx][branch_point];
      // ... ensure this point not marked as a leaf 
      arbor.leafs[parent_branch_idx][branch_point] = 0;
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
      // Initialize leafs vector and mark that this first point is not a leaf 
      arbor.leafs.push_back({0});
      // Initialize synapses vector and mark that this first point is not a synapse
      arbor.synapses.push_back({0});
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
      
      // Mark whether this node is a leaf
      if (s < n_segments - 1) {
        arbor.leafs[parent_branch_idx].push_back(0);
      } else {
        arbor.leafs[parent_branch_idx].push_back(1);
      }
      
      // Mark that this node is not a synapse 
      arbor.synapses[parent_branch_idx].push_back(0);
      
    }
    
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
      // Set arbor ID number 
      arbors[cell_idx].arbor_id.push_back(n_existing_arbors);
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
    
    // For each branch to-be-made: 
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
            int process_node_count = neuron_types[t_num].process_node_count;
            int axon_branch_count = neuron_types[t_num].axon_branch_count;
            int dendrite_branch_count = neuron_types[t_num].dendrite_branch_count;
            
            // Create local axon arbor
            make_arbor(cell_idx, process_node_count, axon_branch_count, true);
            
            // Create local dendrite arbor
            make_arbor(cell_idx, process_node_count, dendrite_branch_count, false);
            
          }
          
          // For all combinations of pre- and post-synaptic neurons in this node
          for (int idx_pre = node_range_start; idx_pre <= node_range_end; idx_pre++) {
            
            // Get neuron types for pre-synaptic neurons
            int t_pre = neuron_type_num[idx_pre];
            
            // Get spike current potential, and power for pre-synaptic cells
            double I_spike = neuron_types[t_pre].I_spike;
            double spike_potential = neuron_types[t_pre].spike_potential;
            double spike_H = I_spike * spike_potential;
            
            // Get neuron valences for pre-synaptic neurons
            double val_pre = neuron_types[t_pre].valence;
            
            // Get coupling scaling factor for pre-synaptic cells
            double coupling_scaling_factor = neuron_types[t_pre].coupling_scaling_factor;
            
            // Initialize pointer to appropriate cell_arbors structure
            cell_arbors& arbor_pre = arbors[idx_pre];
            
            // Confirm first branch is the axon 
            if (!arbor_pre.axon[0]) {
              Rcpp::Rcout << "First branch in arbor_pre for cell index " << idx_pre << " is not an axon; check arbor structure." << std::endl;
              Rcpp::stop("First branch in arbor is not an axon");
            }
            
            // Grab all nodes along axon
            std::vector<Vector3d>& axon_coordinates = arbor_pre.coordinates[0];
            
            // Set transconductance into post-synaptic cells
            for (int idx_post = node_range_start; idx_post <= node_range_end; idx_post++) {
              
              // Get the post-synaptic arbor 
              cell_arbors& arbor_post = arbors[idx_post];
              
              // Assuming first dendrite branch is at index 1
              // ... for local connections, only one dendritic branch as been made so far
              std::vector<Vector3d>& dendrite_coordinates = arbor_post.coordinates[1]; 
              
              // Check for synapses 
              std::vector<int> neighbor_idx = find_first_neighbor(
                axon_coordinates, 
                dendrite_coordinates,
                synaptic_neighborhood
              );
              
              // If one is found, create it
              if (neighbor_idx[0] >= 0) {
                
                // Extend the axon
                // ... add coordinates
                axon_coordinates.push_back(dendrite_coordinates[neighbor_idx[1]]);
                // ... add parent 
                arbor_pre.parents[0].push_back(neighbor_idx[0]);
                // ... ensure old node not marked as leaf 
                arbor_pre.leafs[0][neighbor_idx[0]] = 0;
                // ... and mark new node as leaf
                arbor_pre.leafs[0].push_back(1);
                // ... and mark new node as synapse
                arbor_pre.synapses[0].push_back(1);
                // ... and mark in the synapse idx matrix 
                synapse_idx(idx_pre, idx_post) = axon_coordinates.size() - 1;
                
                // Get neuron types for post-synaptic neurons
                int t_post = neuron_type_num[idx_post];
                
                // Get recurrence factor for this connection type
                double rec_factor = recurrence_factor_matrix(t_post, t_pre);
                double transductance_bias = rec_factor * spike_H * coupling_scaling_factor;
                
                // Set transductance 
                double trans = R::runif(0.0, 2.0) * transductance_bias;
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

MatrixXi network::find_pairwise_lags_by_axon(
    const double& dt // time step length, in unit_time
  ) {
    
    // Initialize matrix to hold pairwise lags, with default value of 0 
    MatrixXi pairwise_lags = MatrixXi::Constant(n_neurons, n_neurons, 0);
    
    // Precompute reciprocals
    const VectorXd inv_vel = neuron_transmission_velocity.cwiseInverse();
    const double inv_dt = 1.0 / dt;
    
    // For each neuron, find the lag to each other neuron based on axonal path length, synapse location, and transmission velocity
    for (int idx_pre = 0; idx_pre < n_neurons; idx_pre++) {
      
      // Get the pre-synaptic arbor 
      cell_arbors& arbor_pre = arbors[idx_pre];
      
      // Confirm first branch is the axon 
      if (!arbor_pre.axon[0]) {
        Rcpp::Rcout << "First branch in arbor_pre for cell index " << idx_pre << " is not an axon; check arbor structure." << std::endl;
        Rcpp::stop("First branch in arbor is not an axon");
      }
      // Confirm no other branches are axons 
      for (int b = 1; b < arbor_pre.axon.size(); b++) {
        if (arbor_pre.axon[b]) {
          Rcpp::Rcout << "Branch index " << b << " in arbor_pre for cell index " << idx_pre << " is an axon; check arbor structure." << std::endl;
          Rcpp::stop("Multiple branches in arbor cannot be axons");
        }
      }
      
      // Grab all nodes along axon and their parents
      std::vector<Vector3d>& axon_coordinates = arbor_pre.coordinates[0];
      std::vector<int> axon_node_parents = arbor_pre.parents[0];
      
      // For each post-synaptic neuron, check for synapses and set lag if found
      for (int idx_post = 0; idx_post < n_neurons; idx_post++) {
        double dist = 0;
        int node_idx = synapse_idx(idx_pre, idx_post);
        if (node_idx >= 0) {
          int parent_node_idx = axon_node_parents[node_idx];
          while (parent_node_idx >= 0) {
            Vector3d node = axon_coordinates[node_idx];
            Vector3d parent_node = axon_coordinates[parent_node_idx];
            dist += std::sqrt(
              std::pow(node[0] - parent_node[0], 2) +
              std::pow(node[1] - parent_node[1], 2) +
              std::pow(node[2] - parent_node[2], 2)
            );
            node_idx = parent_node_idx; 
            parent_node_idx = axon_node_parents[node_idx];
          }
        }
        
        // Convert distance into simulation time-step lag
        const double lag = dist * inv_vel[idx_pre] * inv_dt;
        pairwise_lags(idx_pre, idx_post) = static_cast<int>(std::round(lag));
        
      }
      
    }
    
    return pairwise_lags;
    
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
    List arbor_list(n_neurons);
    if (include_arbors) {
      for (int n = 0; n < n_neurons; n++) {
        // Parse arbor sizes
        int n_arbors = arbors[n].axon.size();
        int n_segments = 0;
        for (int a = 0; a < n_arbors; a++) {
          n_segments += arbors[n].coordinates[a].size();
        }
        // Extract nodes
        NumericMatrix arbor_r(n_segments, 8);
        int last_seg_idx = 0;
        for (int a = 0; a < n_arbors; a++) {
          int n_segs = arbors[n].coordinates[a].size();
          double arbor_type = arbors[n].axon[a] ? 1.0 : 0.0; // 1 for axon, 0 for dendrite
          NumericVector parents_r = wrap(arbors[n].parents[a]);
          for (double &v : parents_r) v++; // put into 1-indexed form for R
          NumericVector leafs_r = wrap(arbors[n].leafs[a]);
          NumericVector synapses_r = wrap(arbors[n].synapses[a]);
          for (int i = 0; i < n_segs; i++) { 
            arbor_r(i + last_seg_idx, 0) = (double)arbors[n].arbor_id[a]; // arbor ID number
            arbor_r(i + last_seg_idx, 1) = arbor_type;
            arbor_r(i + last_seg_idx, 2) = parents_r[i];
            arbor_r(i + last_seg_idx, 3) = leafs_r[i];
            arbor_r(i + last_seg_idx, 4) = synapses_r[i];
            arbor_r(i + last_seg_idx, 5) = arbors[n].coordinates[a][i][0]; // z
            arbor_r(i + last_seg_idx, 6) = arbors[n].coordinates[a][i][1]; // y
            arbor_r(i + last_seg_idx, 7) = arbors[n].coordinates[a][i][2]; // x
          }
          last_seg_idx += n_segs;
        }
        colnames(arbor_r) = CharacterVector::create("arbor_id", "is_axon", "parent_idx", "is_leaf", "is_synapse", "z", "y", "x");
        arbor_list[n] = arbor_r;
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