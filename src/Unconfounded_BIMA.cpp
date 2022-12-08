#include <RcppArmadillo.h>

#include <Rcpp/Benchmark/Timer.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <progress.hpp>
#include <progress_bar.hpp>
// [[Rcpp::depends(RcppProgress)]]

using namespace Rcpp;
using namespace arma;

double square(double y){
  return y*y;
}
double adjust_acceptance(double accept,double sgm,double target = 0.1){
  double y = 1. + 1000.*(accept-target)*(accept-target)*(accept-target);
  if (y < .9)
    y = .9;
  if (y > 1.1)
    y = 1.1;
  sgm *= y;
  return sgm;
}
arma::uvec arma_setdiff(arma::uvec x, arma::uvec y){
  
  x = arma::unique(x);
  y = arma::unique(y);
  
  for (arma::uword j = 0; j < y.n_elem; j++) {
    arma::uvec q1 = arma::find(x == y[j]);
    if (!q1.empty()) {
      x.shed_row(q1(0));
    }
  }
  
  return x;
}
// [[Rcpp::export]]
arma::mat Low_to_high(arma::mat& Low_mat, int p, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  int n = Low_mat.n_cols;
  arma::mat High_mat(p,n);
  for(uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    High_mat.rows(p_idx) = Q*Low_mat.rows(L_range);
  }
  return High_mat;
}
arma::colvec Low_to_high_vec(const arma::colvec& Low_vec, int p,
                             const Rcpp::List& Phi_Q,
                             const Rcpp::List& region_idx, 
                             const Rcpp::List& L_idx){
  int num_region = region_idx.size();
  arma::colvec High_vec(p,1);
  for(uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    High_vec(p_idx) = Q*Low_vec(L_range);
  }
  return High_vec;
}

arma::vec High_to_low_vec(arma::vec& High_vec, int L, Rcpp::List& Phi_Q,
                          Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  arma::colvec Low_vec(L,1);
  for(uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    Low_vec(L_range) = Q.t()*High_vec(p_idx);
  }
  return Low_vec;
  
}

arma::mat High_to_low(const arma::mat& High_mat, int L, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  int n = High_mat.n_cols;
  arma::mat Low_mat = zeros(L,n);
  for(uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    Low_mat.rows(L_range) = Q.t()*High_mat.rows(p_idx);
  }
  return Low_mat;
  
}


arma::uvec complement(arma::uword start, arma::uword end, arma::uword n) {
  arma::uvec y1 = arma::linspace<arma::uvec>(0, start-1, start);
  arma::uvec y2 = arma::linspace<arma::uvec>(end+1, n-1, n-1-end);
  arma::uvec y = arma::join_cols(y1,y2);
  return y;
}

// [[Rcpp::export]]
List get_H_mat(const::mat G){
  mat G_null = null(G);
  mat H_inv = join_vert(G_null.t(),G);
  mat H = inv(H_inv);
  uword q = G.n_rows;
  uword n = G.n_cols;
  mat H1 = H.cols(0,n-q-1);
  mat H2 = H.cols(n-q,n-1);
  
  // get mean and variance for z1
  mat Lambda11 = H1.t()* H1;
  mat Lambda12 = H1.t()* H2;
  mat Lambda11_inv = inv(Lambda11);
  mat Lambda11_inv_sqrt = sqrtmat_sympd(Lambda11_inv);
  mat Left_mat = G_null.t() + Lambda11_inv * Lambda12 * G * H_inv;
  return List::create(Named("H") = H, 
                      Named("Left_mat") = Left_mat,
                      Named("Lambda11_inv_sqrt") = Lambda11_inv_sqrt);
}
// [[Rcpp::export]]
mat hyperplane_MVN_multiple(const::mat G,
                            const::List H_mat,
                            const::vec sigma2_vec,
                            const::mat mu_mat){
  // prepare H mat
  mat H = H_mat["H"];
  uword q = G.n_rows; 
  uword n = G.n_cols;
  mat Left_mat = H_mat["Left_mat"];
  mat Lambda11_inv_sqrt = H_mat["Lambda11_inv_sqrt"];
  
  mat mu_z1 = Left_mat * mu_mat; // n by m
  uword m = sigma2_vec.n_elem;
  mat z1 =   Lambda11_inv_sqrt *  randn(n-q,m);
  z1.each_row() %= sqrt(sigma2_vec.t());
  z1 += mu_z1;
  mat x = H.cols(0,n-q-1)*z1;
  return x.t();
}

class unconfounded_BIMA{
private: 
  int method;
  
  struct LinearRegData{
    vec Y; // n by 1
    mat M; // n by p
    mat M_t; // p by n
    vec X; // n by 1
    rowvec X_t;
    mat C; // q by n
    mat C_t;
    mat CC_t;
    
    int n;
    int p;
    int q;
    
    // additional summary stats
    vec X2_sum_allsample_q;
    mat XcXq_sumsq;
    vec XXq_sumsq;
    
  }dat;
  
  struct STGPParas{
    List Phi_Q;
    List Phi_Q_t;
    vec D_vec;
    vec D_sqrt;
    vec D_inv;
    List region_idx;
    double lambda_alpha;
    double lambda_beta;
    double lambda_nu;
    int L;
    List L_idx;
    int num_region;
  }STGP;
  
  struct MRegParas{
    vec alpha; 
    vec theta_alpha;
    mat xi; // p by q
    mat theta_xi; // L by q
    mat eta; // p by n
    mat eta_t;// n by p
    mat theta_eta; // L by n
    double inv_sigmasq_M;
    double inv_sigmasq_alpha;
    double inv_sigmasq_xi;
    double inv_sigmasq_eta;
  }m_paras;
  
  struct MRegIntermediate{
    mat Mstar_alpha_term; // residual after updating alpha
  } m_temp;

   struct YRegIntermediate{
    vec M_beta_term; // residual after updating alpha
    vec eta_nu_term;
  } y_temp;
  
  
  struct YRegParas{
    vec beta;
    vec theta_beta;
    double gamma;
    vec zeta; // q by 1
    vec nu; // p by 1
    vec theta_nu;
    double inv_sigmasq_Y;
    double inv_sigmasq_beta;
    double inv_sigmasq_gamma;
    double inv_sigmasq_zeta;
    double inv_sigmasq_nu;
  }y_paras;
  
  struct IGParas{
    double a;
    double b;
  }M_ig, alpha_ig, xi_ig, eta_ig, Y_ig, beta_ig, gamma_ig, zeta_ig, nu_ig ;
  
  struct AllProfiles{
    double loglik_m;
    double loglik_y;
  }profile;
  
  
  struct MALA_controls{
    vec alpha_new;
    vec step_alpha;
    vec step_beta;
    vec step_nu;
    vec emp_accept_alpha;
    vec emp_accept_beta;
    vec emp_accept_nu;
    vec target_acceptance_rate;
    mat accept_block_alpha;
    mat accept_block_beta;
    mat accept_block_nu;
    int accept_interval;
  }mala_controls;
  
  struct MCMCsample{
    mat theta_beta;
    mat theta_alpha;
    cube theta_xi;
    mat theta_nu;
    
    vec gamma;
    mat zeta;
    
    vec inv_sigmasq_M;
    vec inv_sigmasq_Y;
    vec loglik_M;
    vec loglik_Y;
    
    mat theta_eta_mean;
    
  }mcmc_sample;
  
  
  struct GibbsSamplerControl{
    int total_iter;
    int burnin;
    int mcmc_sample;
    int thinning;
    int verbose;
    int save_profile;
    int total_profile;
    int start_eta;
    int eta_interval;
    int start_save_eta;
  } gibbs_control;
  
  
  int iter;
  bool display_progress;
  
public:
  void set_method(CharacterVector in_method, bool in_display_progress){
    if(in_method(0)=="MALA"){
      std::cout << "MALA" << std::endl;
      method = 0;
    }
    display_progress = in_display_progress;
    
  };
  
  int get_method(){
    return method;
  }
  
  void load_data(const vec& in_y, const mat&in_M, const vec& in_X, const mat& in_C){
    dat.Y = in_y;
    dat.M = in_M;
    dat.M_t = trans(in_M);
    dat.X = in_X;
    dat.C = in_C;
    dat.C_t = in_C.t();
    dat.CC_t = in_C * dat.C_t;
    dat.n = dat.Y.n_elem;
    dat.p = dat.M.n_rows;
    dat.q = dat.C.n_rows;
    dat.X_t = in_X.t();
    Rcout<<"Loading data: success! n="<<dat.n<<"; p="<<dat.p<<"; q="<<dat.q<<std::endl;
    
    // compute summmary stats
    dat.X2_sum_allsample_q = arma::zeros(dat.q,1);
    dat.XcXq_sumsq = arma::zeros(dat.q-1,dat.q); // arma::sum_i C[-j]*C_j
    dat.XXq_sumsq = arma::zeros(dat.q,1); // arma::sum_i C*C_j
    for(int j=0; j<dat.q; j++){
      dat.X2_sum_allsample_q(j) += arma::sum(dat.C.row(j) %dat.C.row(j));
      dat.XXq_sumsq(j) += accu(dat.X_t %dat.C.row(j));
      arma::uvec c_j = complement(j, j, dat.q);
      dat.XcXq_sumsq.col(j) += dat.C.rows(c_j) * trans(dat.C.row(j));
    }
    
  }
  
  void load_STGP(const List& Kernel_params, 
                 double lambda_alpha, 
                 double lambda_beta,
                 double lambda_nu){
    Rcout<<"Loading Kernel_params, must contain: List Phi_Q, vec D_vec, List region_idx (from 0)."<<std::endl;
    List Phi_Q = Kernel_params["Phi_Q"];
    vec D_vec = Kernel_params["D_vec"];
    List region_idx = Kernel_params["region_idx"];
    
    STGP.Phi_Q = Phi_Q;
    STGP.region_idx = region_idx;
    STGP.D_vec = D_vec;
    STGP.D_sqrt = sqrt(D_vec);
    STGP.D_inv = 1.0/D_vec;
    
    STGP.lambda_alpha = lambda_alpha;
    STGP.lambda_beta = lambda_beta;
    STGP.lambda_nu = lambda_nu;
    
    STGP.L = D_vec.n_elem;
    STGP.num_region = Phi_Q.length();
    // create index list for basis coefficient
    uvec L_all(STGP.num_region);
    for(int r_int=0; r_int < STGP.num_region; r_int++){
      mat Q = Phi_Q[r_int];
      L_all(r_int) = Q.n_cols;
    }
    
    List L_idx(STGP.num_region);
    List Phi_Q_t(STGP.num_region);
    uvec L_cumsum = cumsum(L_all);
    for(int r_int=0; r_int < STGP.num_region; r_int++){
      arma::uvec L_idx_r;
      if(r_int==0){
        L_idx_r = arma::linspace<arma::uvec>(0,L_cumsum(r_int)-1,L_all(r_int));
      }else{
        L_idx_r = arma::linspace<arma::uvec>(L_cumsum(r_int-1),L_cumsum(r_int)-1,L_all(r_int));
      }
      L_idx[r_int] = L_idx_r; // is this legal?
      mat Q = Phi_Q[r_int];
      Phi_Q_t[r_int] = Q.t();
    }
    STGP.L_idx = L_idx;
    STGP.Phi_Q_t = Phi_Q_t;
    
    
  }
  
  void set_initial_params(const List& init_paras){
    Rcout<<"Setting initial params.. must contain:"<<std::endl;
    Rcout<<"alpha; xi(p by q); eta(p by n); sigma_M; sigma_alpha;sigma_xi;sigma_eta;"<<std::endl;
    Rcout<<"beta; gamma; zeta(q by 1); nu; sigma_Y; sigma_beta; sigma_gamma;sigma_zeta; sigma_nu;"<<std::endl;
    
    // --------- set up m_paras --------- //
    vec alpha = init_paras["alpha"]; m_paras.alpha = alpha;
    m_paras.theta_alpha = High_to_low_vec(alpha, STGP.L, STGP.Phi_Q, STGP.region_idx, STGP.L_idx);
    mat xi = init_paras["xi"]; m_paras.xi = xi;
    m_paras.theta_xi = High_to_low(xi,STGP.L, STGP.Phi_Q, STGP.region_idx, STGP.L_idx);
    mat eta = init_paras["eta"]; m_paras.eta = eta; m_paras.eta_t = eta.t();
    m_paras.theta_eta = High_to_low(eta,STGP.L, STGP.Phi_Q, STGP.region_idx, STGP.L_idx);
    
    
    double sigma_M = init_paras["sigma_M"];
    m_paras.inv_sigmasq_M = 1/sigma_M/sigma_M;
    double sigma_alpha = init_paras["sigma_alpha"];
    m_paras.inv_sigmasq_alpha = 1/sigma_alpha/sigma_alpha;
    double sigma_xi = init_paras["sigma_xi"];
    m_paras.inv_sigmasq_xi = 1/sigma_xi/sigma_xi;
    double sigma_eta = init_paras["sigma_eta"];
    m_paras.inv_sigmasq_eta = 1/sigma_eta/sigma_eta;
    
    // --------- set up y_paras --------- //
    vec beta = init_paras["beta"]; y_paras.beta = beta;
    y_paras.theta_beta = High_to_low_vec(beta, STGP.L, STGP.Phi_Q, STGP.region_idx, STGP.L_idx);
    double gamma = init_paras["gamma"]; y_paras.gamma = gamma;
    vec zeta = init_paras["zeta"]; y_paras.zeta = zeta;
    vec nu = init_paras["nu"]; y_paras.nu = nu;
    y_paras.theta_nu = High_to_low_vec(nu,STGP.L, STGP.Phi_Q, STGP.region_idx, STGP.L_idx);
    
    
    double sigma_Y = init_paras["sigma_Y"];
    y_paras.inv_sigmasq_Y = 1/sigma_Y/sigma_Y;
    double sigma_beta = init_paras["sigma_beta"];
    y_paras.inv_sigmasq_beta = 1/sigma_beta/sigma_beta;
    double sigma_gamma = init_paras["sigma_gamma"];
    y_paras.inv_sigmasq_gamma = 1/sigma_gamma/sigma_gamma;
    double sigma_zeta = init_paras["sigma_zeta"];
    y_paras.inv_sigmasq_zeta = 1/sigma_zeta/sigma_zeta;
    double sigma_nu = init_paras["sigma_nu"];
    y_paras.inv_sigmasq_nu = 1/sigma_nu/sigma_nu;
    
    
    M_ig.a =  alpha_ig.a =  xi_ig.a =  eta_ig.a =  Y_ig.a =  beta_ig.a =  gamma_ig.a =  zeta_ig.a =  nu_ig.a = 1.0;
    M_ig.b =  alpha_ig.b =  xi_ig.b =  eta_ig.b =  Y_ig.b =  beta_ig.b =  gamma_ig.b =  zeta_ig.b =  nu_ig.b = 1.0;
    
    Rcout<<"set_initial_params:successful!"<<std::endl;
  }
  
  void set_gibbs_control(int in_mcmc_sample, int in_burnin, int in_thinning, 
                         int in_verbose, int in_save_profile,
                         int theta_eta_interval,
                         int start_save_eta,
                         int start_update_eta){
    
    Rcout<<"set_gibbs_control..."<<std::endl;
    gibbs_control.mcmc_sample = in_mcmc_sample;
    gibbs_control.burnin = in_burnin;
    gibbs_control.thinning = in_thinning;
    gibbs_control.total_iter = gibbs_control.burnin;
    gibbs_control.total_iter += gibbs_control.mcmc_sample*gibbs_control.thinning; 
    gibbs_control.verbose = in_verbose;
    gibbs_control.save_profile = in_save_profile;
    if(gibbs_control.save_profile > 0){
      gibbs_control.total_profile = gibbs_control.total_iter/gibbs_control.save_profile;
    } else{
      gibbs_control.total_profile = 0;
    }
    
    gibbs_control.eta_interval = theta_eta_interval;
    gibbs_control.start_save_eta = start_save_eta;
    gibbs_control.start_eta = start_update_eta;
    Rcout<<"set_gibbs_control:successful!"<<std::endl;
  };
  
  void set_MALA_control(vec step, vec target_acceptance_rate, int accept_interval){
    mala_controls.step_alpha = mala_controls.step_beta = mala_controls.step_nu = step;
    mala_controls.target_acceptance_rate = target_acceptance_rate;
    mala_controls.accept_block_alpha.zeros(gibbs_control.total_iter, STGP.num_region);
    mala_controls.accept_block_beta.zeros(gibbs_control.total_iter, STGP.num_region);
    mala_controls.accept_block_nu.zeros(gibbs_control.total_iter, STGP.num_region);
    mala_controls.accept_interval = accept_interval;
    mala_controls.emp_accept_alpha.zeros(gibbs_control.total_iter/mala_controls.accept_interval,
                                         STGP.num_region);
    mala_controls.emp_accept_beta = mala_controls.emp_accept_alpha;
    mala_controls.emp_accept_nu = mala_controls.emp_accept_alpha;
  }
  
  // ------------------- M-regression ------------------- //
  void update_alpha(){
    
    for(int region_iter=0; region_iter < STGP.num_region; region_iter++){
      
      arma::uvec delta = find(abs(m_paras.alpha) > STGP.lambda_alpha);
      arma::uvec idx = STGP.region_idx[region_iter];
      arma::uvec delta_Q = find(abs(m_paras.alpha(idx))>STGP.lambda_alpha);
      arma::mat Q = STGP.Phi_Q[region_iter];
      arma::uvec L_idx = STGP.L_idx[region_iter];
      arma::uvec delta_in_block = intersect(idx,delta);
      
      arma::mat K_block_t = STGP.Phi_Q_t[region_iter];
      arma::mat M_star = K_block_t*dat.M.rows(idx) - m_paras.theta_xi.rows(L_idx) * dat.C - m_paras.theta_eta.rows(L_idx);
      arma::mat temp = M_star - K_block_t.cols(delta_Q)*(m_paras.alpha.rows(delta_in_block)-sign(m_paras.alpha.rows(delta_in_block))*STGP.lambda_alpha)*dat.X_t;
      arma::mat temp_X = temp;
      temp_X.each_row()%= dat.X_t;
      arma::vec  temp_sum = arma::sum(temp_X,1)*m_paras.inv_sigmasq_M;
      arma::vec  grad_f = -m_paras.theta_alpha(L_idx)/STGP.D_vec(L_idx)*m_paras.inv_sigmasq_alpha+
        K_block_t.cols(delta_Q)*Q.rows(delta_Q) *temp_sum; // L x 1, use smooth function in grad
      double step = mala_controls.step_alpha(region_iter);
      arma::vec   theta_alpha_diff = step*grad_f+sqrt(2*step)*arma::randn(size(grad_f));
      arma::vec  theta_alpha_new_block = m_paras.theta_alpha(L_idx) + theta_alpha_diff;
      
      // MH step
      double log_target_density = -0.5*square(norm(m_paras.theta_alpha(L_idx)/sqrt(STGP.D_vec(L_idx)),2))*m_paras.inv_sigmasq_alpha-
        0.5*square(arma::norm(temp,"fro"))*m_paras.inv_sigmasq_M;
      
      mala_controls.alpha_new = m_paras.alpha;// change this
      mala_controls.alpha_new(idx) += Q*theta_alpha_diff;
      // arma::uvec delta_new = find(abs(alpha_new)>lambda);// find for one region
      arma::uvec delta_Q_new = find(abs(mala_controls.alpha_new(idx))>STGP.lambda_alpha);
      arma::uvec delta_in_block_new = idx(delta_Q_new);
      // arma::uvec delta_in_block_new = intersect(idx,delta_new);
      arma::vec   alpha_new_region = mala_controls.alpha_new.rows(delta_in_block_new);
      arma::mat temp_new = M_star - K_block_t.cols(delta_Q_new)*(alpha_new_region-sign(alpha_new_region)*STGP.lambda_alpha)*dat.X_t;
      arma::mat temp_X_new = temp_new;
      temp_X_new.each_row()%= dat.X_t;
      arma::vec  temp_sum_new = arma::sum(temp_X_new,1)*m_paras.inv_sigmasq_M;
      arma::vec  grad_f_new = -theta_alpha_new_block/STGP.D_vec(L_idx)*m_paras.inv_sigmasq_alpha+
        K_block_t.cols(delta_Q_new)*Q.rows(delta_Q_new) *temp_sum_new; // L x 1, use smooth function in grad
      double log_target_density_new = -0.5*square(norm( theta_alpha_new_block/sqrt(STGP.D_vec(L_idx)),2))*m_paras.inv_sigmasq_alpha-
        0.5*square(arma::norm(temp_new,"fro"))*m_paras.inv_sigmasq_M;
      double log_q = -1/4/step * square(norm(-theta_alpha_diff-step*grad_f_new,2));
      double log_q_new = -1/4/step * square(norm(theta_alpha_diff-step*grad_f,2));
      double rho = log_target_density_new + log_q - log_target_density - log_q_new;
      if(log(arma::randu())<=rho){
        m_paras.theta_alpha(L_idx) = theta_alpha_new_block;
        m_paras.alpha = mala_controls.alpha_new;
        mala_controls.accept_block_alpha(iter,region_iter) = 1;
        temp = temp_new;
      }
    }// end of region loop
    
    // update acceptance rate
    if( (iter%mala_controls.accept_interval==0) & (iter>0) & (iter > mala_controls.accept_interval) ){
      arma::uvec u = arma::linspace<arma::uvec>(iter-mala_controls.accept_interval,
                                                iter-1,mala_controls.accept_interval);
      mala_controls.emp_accept_alpha.row(iter/mala_controls.accept_interval-1) = 
        mean(mala_controls.accept_block_alpha.rows(u),0);
      if(iter<gibbs_control.burnin){
        arma::vec  sigma_t = sqrt(2*mala_controls.step_alpha);
        for(arma::uword l = 0; l<STGP.num_region; l++){
          sigma_t(l) = adjust_acceptance(mala_controls.emp_accept_alpha(iter/mala_controls.accept_interval-1,l),sigma_t(l),
                  mala_controls.target_acceptance_rate(l));
          mala_controls.step_alpha(l) = sigma_t(l)*sigma_t(l)/2;
          if(mala_controls.step_alpha(l)>1){mala_controls.step_alpha(l)=1;}
        }
        
      }
    }
    
    // update sigma_alpha
    m_paras.inv_sigmasq_alpha = arma::randg( arma::distr_param(alpha_ig.a + STGP.L*0.5, 
                                                               1/(alpha_ig.b + dot(m_paras.theta_alpha,(m_paras.theta_alpha/STGP.D_vec)/2))) );
    
    // update m_temp
    arma::mat Mstar_alpha_term = arma::zeros(size(m_paras.theta_eta));
    for(arma::uword m=0; m<STGP.num_region; m++){
      arma::uvec delta = find(abs(m_paras.alpha)>STGP.lambda_alpha);
      arma::uvec idx = STGP.region_idx[m];
      arma::uvec delta_Q = find(abs(m_paras.alpha(idx))>STGP.lambda_alpha);
      arma::uvec delta_in_block = intersect(idx,delta);
      arma::mat Q = STGP.Phi_Q[m];
      arma::uvec L_idx = STGP.L_idx[m];
      arma::mat K_block_t = STGP.Phi_Q_t[m];
      Mstar_alpha_term.rows(L_idx) = K_block_t.cols(delta_Q)*(dat.M.rows(delta_in_block)-( m_paras.alpha.rows(delta_in_block)-sign(m_paras.alpha.rows(delta_in_block))*STGP.lambda_alpha)*dat.X_t );
    }
    m_temp.Mstar_alpha_term = Mstar_alpha_term;
  };
  
  void update_xi(){
    arma::mat xi_res = m_temp.Mstar_alpha_term - m_paras.theta_eta;// L by n
    arma::mat mean_xi = arma::mat(STGP.L,dat.q);
    double xi_b = 0;
    
    for(int j =0; j<dat.q; j++){
      arma::vec  Sigma_xi_j = 1/(1/STGP.D_vec*m_paras.inv_sigmasq_xi + m_paras.inv_sigmasq_M*dat.X2_sum_allsample_q(j));
      arma::uvec c_j = complement(j, j, dat.q);
      // change the following line when theta_eta needs to be updated
      arma::vec  mean_xi_j = xi_res*dat.C_t.col(j) - m_paras.theta_xi.cols(c_j) * dat.XcXq_sumsq.col(j);
      mean_xi_j %= Sigma_xi_j*m_paras.inv_sigmasq_M;
      mean_xi.col(j) = mean_xi_j;
      m_paras.theta_xi.col(j) = arma::randn(STGP.L,1)%sqrt(Sigma_xi_j) +  mean_xi_j;
      xi_b += dot(m_paras.theta_xi.col(j),m_paras.theta_xi.col(j)/STGP.D_vec);
    }
    
    // update sigma_xi
    m_paras.inv_sigmasq_xi = arma::randg( arma::distr_param(xi_ig.a + STGP.L*dat.q*0.5, 
                                                               1/(xi_ig.b + xi_b/2)) );
    
  }
  
  
  
  void update_sigma_M(){
    mat M_res = m_temp.Mstar_alpha_term - m_paras.theta_xi*dat.C - m_paras.theta_eta;
    double M_norm = norm(M_res,"fro");
    m_paras.inv_sigmasq_M = arma::randg( arma::distr_param(M_ig.a + dat.n*STGP.L/2,1/(M_ig.b + M_norm*M_norm/2) ) );
    
    // update loglik_m
    profile.loglik_m = -0.5*square(arma::norm(M_res,"fro"))*m_paras.inv_sigmasq_M;
    profile.loglik_m += 0.5*dat.n*STGP.L*log(m_paras.inv_sigmasq_M);
  }
  
  
  // ------------------- Y-regression ------------------- //
  
  void update_beta(){
    arma::colvec Y_star = dat.Y - y_paras.gamma*dat.X -  dat.C_t*y_paras.zeta -  m_paras.eta_t * y_paras.nu;
    vec temp,temp_new, tb_temp; double temp_2norm2,  log_target_density , log_target_density_new;
    for(int region_iter=0; region_iter < STGP.num_region; region_iter++){
        
        
        // ------------- update theta_beta ------------- //
        
        arma::uvec delta = find(abs(y_paras.beta)>STGP.lambda_beta);
        arma::uvec idx = STGP.region_idx[region_iter];
        arma::uvec delta_Q = find(abs(y_paras.beta(idx))>STGP.lambda_beta);
        arma::mat Q = STGP.Phi_Q[region_iter];
        arma::uvec L_idx = STGP.L_idx[region_iter];


        arma::mat K_block_t = STGP.Phi_Q_t[region_iter];
        arma::uvec delta_in_block = intersect(idx,delta);
        arma::colvec temp_idx = dat.M_t.cols(delta_in_block)*(y_paras.beta.rows(delta_in_block)-
          sign(y_paras.beta.rows(delta_in_block))*STGP.lambda_beta)/1.00;
        

        if(region_iter==0){
          temp = Y_star-dat.M_t.cols(delta)*(y_paras.beta.rows(delta)-sign(y_paras.beta.rows(delta))*STGP.lambda_beta)/1.00;
          // temp -= eta_t * nu;
          temp_2norm2 = dot(temp,temp);
        }

        arma::colvec grad_f = -y_paras.theta_beta(L_idx)/STGP.D_vec(L_idx)*y_paras.inv_sigmasq_beta+
          K_block_t.cols(delta_Q)*dat.M.rows(delta_in_block)*temp/1.00*y_paras.inv_sigmasq_Y;
        double step = mala_controls.step_beta(region_iter);
        arma::colvec theta_beta_diff = step*grad_f+sqrt(2*step)*arma::randn(size(grad_f));
        arma::colvec theta_beta_new_block = y_paras.theta_beta(L_idx)+theta_beta_diff;





        // MH step

        tb_temp = y_paras.theta_beta(L_idx)/STGP.D_sqrt(L_idx);
        log_target_density = -0.5*dot(tb_temp,tb_temp)*y_paras.inv_sigmasq_beta-
          0.5*temp_2norm2*y_paras.inv_sigmasq_Y;

        arma::colvec beta_new = y_paras.beta;
        beta_new(idx) += Q*theta_beta_diff;
        arma::uvec delta_new = arma::find(abs(beta_new)>STGP.lambda_beta);
        arma::uvec delta_Q_new = arma::find(abs(beta_new(idx))>STGP.lambda_beta);
        arma::uvec delta_in_block_new = intersect(idx,delta_new);

        arma::colvec x1 = arma::zeros(dat.n,1); arma::colvec x2 = arma::zeros(dat.n,1);
        bool b1 = delta_in_block.n_elem>0;
        bool b2 = delta_in_block_new.n_elem>0;
        if(b1){
          x1 = dat.M_t.cols(delta_in_block)*(y_paras.beta.rows(delta_in_block)-sign(y_paras.beta.rows(delta_in_block))*STGP.lambda_beta)/1.00;
        }
        if(b2){
          x2 = dat.M_t.cols(delta_in_block_new)*(beta_new.rows(delta_in_block_new)-sign(beta_new.rows(delta_in_block_new))*STGP.lambda_beta)/1.00;
        }

        if(!b1 && b2){
          temp_new = temp-x2;
        }
        if(b1 && !b2){
          temp_new = temp+x1;
        }
        if(b1 && b2){
          temp_new = temp+x1-x2;
        }
        if(!b1 && !b2){
          temp_new = temp;
        }

        tb_temp = theta_beta_new_block/STGP.D_sqrt(L_idx);
        log_target_density_new = -0.5*dot(tb_temp,tb_temp)*y_paras.inv_sigmasq_beta-
          0.5*dot(temp_new,temp_new)*y_paras.inv_sigmasq_Y;
        arma::colvec grad_f_new = -theta_beta_new_block/STGP.D_vec(L_idx)*y_paras.inv_sigmasq_beta+
          K_block_t.cols(delta_Q_new)*dat.M.rows(delta_in_block_new)*temp_new/1.00*y_paras.inv_sigmasq_Y; // L x 1

        double log_q = -1/4/step * square(norm(-theta_beta_diff-step*grad_f_new,2));//accu, change this part, %
        double log_q_new = -1/4/step * square(norm(theta_beta_diff-step*grad_f,2));
        double rho = log_target_density_new + log_q - log_target_density - log_q_new;
      
        
        if(log(arma::randu())<=rho){
          y_paras.theta_beta(L_idx) = theta_beta_new_block;
          y_paras.beta = beta_new;
          temp = temp_new;
          log_target_density=log_target_density_new;
          mala_controls.accept_block_beta(iter,region_iter) = 1;
        }


      }// end of block update

      // update M_beta for computing residual
      uvec delta = find(abs(y_paras.beta)>STGP.lambda_beta);
      y_temp.M_beta_term = dat.M_t.cols(delta)*(y_paras.beta.rows(delta)-sign(y_paras.beta.rows(delta))*STGP.lambda_beta)/1.00;

      // update sigma_beta
      double sigma_beta_a = beta_ig.a + y_paras.theta_beta.n_elem/2.0;
      double sigma_beta_b = beta_ig.b + dot(y_paras.theta_beta,y_paras.theta_beta/STGP.D_vec)/2.0;
      y_paras.inv_sigmasq_beta = arma::randg( arma::distr_param(sigma_beta_a,1.0/sigma_beta_b) );
      
  }

  
  void update_nu(){
    arma::colvec Y_star = dat.Y - dat.M_t * y_paras.beta - y_paras.gamma*dat.X -  dat.C_t*y_paras.zeta ;
    vec temp_nu, tb_temp_nu, temp_new_nu; double temp_2norm2_nu, log_target_density_nu, log_target_density_new_nu;
    for(int region_iter=0; region_iter < STGP.num_region; region_iter++){
    // ------------- update theta_nu ------------- //
          arma::uvec idx = STGP.region_idx[region_iter];
          arma::mat Q = STGP.Phi_Q[region_iter];
          arma::mat K_block_t = STGP.Phi_Q_t[region_iter];
          arma::uvec L_idx = STGP.L_idx[region_iter];

          arma::uvec delta_nu = find(abs(y_paras.nu)>STGP.lambda_nu);
          arma::uvec delta_Q_nu = find(abs(y_paras.nu(idx))>STGP.lambda_nu);
          
          
          arma::uvec delta_in_block_nu = intersect(idx,delta_nu);
          arma::colvec temp_idx_nu = m_paras.eta_t.cols(delta_in_block_nu)*(
            y_paras.nu.rows(delta_in_block_nu)-sign(y_paras.nu.rows(delta_in_block_nu))*STGP.lambda_nu )/1.00;
          
          
          if(region_iter==0){
            temp_nu = Y_star-m_paras.eta_t.cols(delta_nu)*(y_paras.nu.rows(delta_nu)-sign(y_paras.nu.rows(delta_nu))*STGP.lambda_nu)/1.00;
            // temp_nu -= M_t * beta;
            temp_2norm2_nu = dot(temp_nu,temp_nu);
          }
          
          arma::colvec grad_f_nu = -y_paras.theta_nu(L_idx)/STGP.D_vec(L_idx)*y_paras.inv_sigmasq_nu+
            K_block_t.cols(delta_Q_nu)*m_paras.eta.rows(delta_in_block_nu)*temp_nu/1.00*y_paras.inv_sigmasq_Y;
          double step_nu = mala_controls.step_nu(region_iter);
          arma::colvec theta_nu_diff = step_nu*grad_f_nu+sqrt(2*step_nu)*arma::randn(size(grad_f_nu));
          arma::colvec theta_nu_new_block = y_paras.theta_nu(L_idx)+theta_nu_diff;
          
          // MH step
          
          tb_temp_nu = y_paras.theta_nu(L_idx)/STGP.D_sqrt(L_idx);
          log_target_density_nu = -0.5*dot(tb_temp_nu,tb_temp_nu)*y_paras.inv_sigmasq_nu-
            0.5*temp_2norm2_nu*y_paras.inv_sigmasq_Y;
          
          arma::colvec nu_new = y_paras.nu;
          nu_new(idx) += Q*theta_nu_diff;
          arma::uvec delta_new_nu = arma::find(abs(nu_new)>STGP.lambda_nu);
          arma::uvec delta_Q_new_nu = arma::find(abs(nu_new(idx))>STGP.lambda_nu);
          arma::uvec delta_in_block_new_nu = intersect(idx,delta_new_nu);
          
          arma::colvec x1_nu = arma::zeros(dat.n,1); arma::colvec x2_nu = arma::zeros(dat.n,1);
          bool b1_nu = delta_in_block_nu.n_elem>0;
          bool b2_nu = delta_in_block_new_nu.n_elem>0;
          if(b1_nu){
            x1_nu = m_paras.eta_t.cols(delta_in_block_nu)*(y_paras.nu.rows(delta_in_block_nu)-sign(y_paras.nu.rows(delta_in_block_nu))*STGP.lambda_nu)/1.00;
          }
          if(b2_nu){
            x2_nu = m_paras.eta_t.cols(delta_in_block_new_nu)*(nu_new.rows(delta_in_block_new_nu)-sign(nu_new.rows(delta_in_block_new_nu))*STGP.lambda_nu)/1.00;
          }
          
          if(!b1_nu && b2_nu){
            temp_new_nu = temp_nu-x2_nu;
          }
          if(b1_nu && !b2_nu){
            temp_new_nu = temp_nu+x1_nu;
          }
          if(b1_nu && b2_nu){
            temp_new_nu = temp_nu+x1_nu-x2_nu;
          }
          if(!b1_nu && !b2_nu){
            temp_new_nu = temp_nu;
          }
          
          
          // ------------- time_seg4 ------------- //
          tb_temp_nu = theta_nu_new_block/STGP.D_sqrt(L_idx);
          log_target_density_new_nu = -0.5*dot(tb_temp_nu,tb_temp_nu)*y_paras.inv_sigmasq_nu-
            0.5*dot(temp_new_nu,temp_new_nu)*y_paras.inv_sigmasq_Y;
          arma::colvec grad_f_new_nu = -theta_nu_new_block/STGP.D_vec(L_idx)*y_paras.inv_sigmasq_nu+
            K_block_t.cols(delta_Q_new_nu)*m_paras.eta.rows(delta_in_block_new_nu)*temp_new_nu/1.00*y_paras.inv_sigmasq_Y; // L x 1
          
          double log_q_nu = -1/4/step_nu * square(norm(-theta_nu_diff-step_nu*grad_f_new_nu,2));//accu, change this part, %
          double log_q_new_nu = -1/4/step_nu * square(norm(theta_nu_diff-step_nu*grad_f_nu,2));
          double rho_nu = log_target_density_new_nu + log_q_nu - log_target_density_nu - log_q_new_nu;

          
          if(log(arma::randu())<=rho_nu){
            y_paras.theta_nu(L_idx) = theta_nu_new_block;
            y_paras.nu = nu_new;
            temp_nu = temp_new_nu;
            log_target_density_nu = log_target_density_new_nu;
            mala_controls.accept_block_nu(iter,region_iter) = 1;
          }

        }// end of block update

        // update residual term
        arma::uvec delta_nu = find(abs(y_paras.nu)>STGP.lambda_nu);
        y_temp.eta_nu_term = m_paras.eta_t.cols(delta_nu)*(y_paras.nu.rows(delta_nu)-sign(y_paras.nu.rows(delta_nu))*STGP.lambda_nu)/1.00;

        // update sigma_nu
        double sigma_nu_a = nu_ig.a + y_paras.theta_nu.n_elem/2.0;
        double sigma_nu_b = nu_ig.b + dot(y_paras.theta_nu,y_paras.theta_nu/STGP.D_vec)/2.0;
        y_paras.inv_sigmasq_nu = arma::randg( arma::distr_param(sigma_nu_a,1.0/sigma_nu_b) );

  }
  

  void update_gamma(){
    double post_sigma_gamma2 = 1/(arma::sum(arma::dot(dat.X,dat.X))*y_paras.inv_sigmasq_Y + y_paras.inv_sigmasq_gamma);
    double temp2 = dot(dat.X, dat.Y - y_temp.M_beta_term - dat.C_t * y_paras.zeta - y_temp.eta_nu_term);
    double mu_gamma = post_sigma_gamma2 * temp2*y_paras.inv_sigmasq_Y;
    y_paras.gamma = arma::randn() * sqrt(post_sigma_gamma2) + mu_gamma;

    // update sigma_gamma
    double sigma_beta_a = gamma_ig.a + 1/2.0;
    double sigma_beta_b = gamma_ig.b + y_paras.gamma*y_paras.gamma/2.0;
    y_paras.inv_sigmasq_gamma = arma::randg( arma::distr_param(sigma_beta_a,1.0/sigma_beta_b) );
  }
  
  void update_zeta(){
    arma::mat Sigma_zetay_inv = y_paras.inv_sigmasq_Y*dat.CC_t + y_paras.inv_sigmasq_zeta;
    arma::mat Sigma_zetay = inv_sympd(Sigma_zetay_inv);
    arma::colvec mu_zetay = Sigma_zetay* dat.C *(  dat.Y - y_temp.M_beta_term - y_temp.eta_nu_term - y_paras.gamma*dat.X )*y_paras.inv_sigmasq_Y;
    y_paras.zeta = arma::mvnrnd( mu_zetay, Sigma_zetay);

    // update sigma_gamma
    double sigma_beta_a = zeta_ig.a + y_paras.zeta.n_elem/2.0;
    double sigma_beta_b = zeta_ig.b + dot(y_paras.zeta,y_paras.zeta)/2.0;
    y_paras.inv_sigmasq_zeta = arma::randg( arma::distr_param(sigma_beta_a,1.0/sigma_beta_b) );
  }
  

  void update_sigma_Y(){
    arma::colvec temp_sigma_Y =  dat.Y - y_temp.M_beta_term - dat.C_t * y_paras.zeta - y_paras.gamma*dat.X - y_temp.eta_nu_term;
    double temp2 = dot(temp_sigma_Y,temp_sigma_Y);
    double sigma_Y_b = Y_ig.b + temp2/2.0;
    y_paras.inv_sigmasq_Y = arma::randg( arma::distr_param(Y_ig.a + dat.n/2.0, 1.0/sigma_Y_b) );

    // update loglik
    profile.loglik_y = -temp2/2.0 + 0.5*dat.n*log(y_paras.inv_sigmasq_Y);
  }
  

  // jointly
  void update_eta(const List& H_mat, const mat&G){
    
    vec res_y = dat.Y - y_temp.M_beta_term - dat.C_t * y_paras.zeta - y_paras.gamma*dat.X; // n by 1
    mat res_Mstar = m_temp.Mstar_alpha_term - m_paras.theta_xi*dat.C; // L by n

    // get posterior variance mat
    vec D = STGP.D_inv * m_paras.inv_sigmasq_eta + m_paras.inv_sigmasq_M;
    vec a = sqrt(y_paras.inv_sigmasq_Y) * y_paras.theta_nu;
    mat A = a * a.t();
    A.each_row() %= 1.0/D.t();
    A.each_col() %= 1.0/D;
    mat Var =  diagmat(1.0/D) - A/(1.0 +  dot(a,a/D)); // L by L

    // get posterior mean
    mat E = m_paras.inv_sigmasq_M * res_Mstar; // L by n
    E.each_row() += y_paras.inv_sigmasq_Y * res_y.t();
    E = Var * E;

    // update constrained theta_eta
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, Var);
    
    vec eta_star_sigma_pos = eigval;
    mat theta_eta_star_mean = E.t() * eigvec;
    mat theta_eta_star = hyperplane_MVN_multiple(G,H_mat,eta_star_sigma_pos,theta_eta_star_mean); // output L by n
    m_paras.theta_eta = eigvec * theta_eta_star;

    // update sigma_eta
    mat theta_eta_temp = m_paras.theta_eta;
    theta_eta_temp.each_col() %= 1/STGP.D_sqrt;
    double eta_norm = norm(theta_eta_temp,"fro");
    m_paras.inv_sigmasq_eta = arma::randg( arma::distr_param(eta_ig.a + 0.5*dat.n*STGP.L,
                                                1/(eta_ig.b + eta_norm*eta_norm/2)) );

    m_paras.eta = Low_to_high(m_paras.theta_eta, dat.p, STGP.Phi_Q, STGP.region_idx, STGP.L_idx);
    m_paras.eta_t = m_paras.eta.t();

    // update y_temp.eta_nu_term
    arma::uvec delta_nu = find(abs(y_paras.nu)>STGP.lambda_nu);
    y_temp.eta_nu_term = m_paras.eta_t.cols(delta_nu)*(y_paras.nu.rows(delta_nu)-sign(y_paras.nu.rows(delta_nu))*STGP.lambda_nu)/1.00;


  }

  void initialize_paras_sample(){
    mcmc_sample.theta_beta.zeros(STGP.L,gibbs_control.mcmc_sample);
    mcmc_sample.theta_alpha.zeros(STGP.L,gibbs_control.mcmc_sample);
    mcmc_sample.theta_xi.zeros(STGP.L,dat.q,gibbs_control.mcmc_sample);
    mcmc_sample.theta_nu.zeros(STGP.L,gibbs_control.mcmc_sample);

    mcmc_sample.gamma.zeros(gibbs_control.mcmc_sample);
    mcmc_sample.zeta.zeros(dat.q,gibbs_control.mcmc_sample);

    mcmc_sample.inv_sigmasq_M.zeros(gibbs_control.mcmc_sample);
    mcmc_sample.inv_sigmasq_Y.zeros(gibbs_control.mcmc_sample);
    mcmc_sample.loglik_M.zeros(gibbs_control.mcmc_sample);
    mcmc_sample.loglik_Y.zeros(gibbs_control.mcmc_sample);
    
    mcmc_sample.theta_eta_mean.zeros(size(m_paras.theta_eta));
  }

  void run_mcmc(){
    
    initialize_paras_sample();
    mat G = join_vert(dat.X_t, dat.C);
    List H_mat = get_H_mat(G);
    
    int eta_counter = 0;
    
    Progress prog(gibbs_control.total_iter, display_progress);
    for(iter=0; iter< gibbs_control.total_iter; iter++){
      prog.increment();
      
      // Rcout<<"run_mcmc:iter = "<<iter<<std::endl;
      if(iter ==0){
        // update y_temp.eta_nu_term
        arma::uvec delta_nu = find(abs(y_paras.nu)>STGP.lambda_nu);
        y_temp.eta_nu_term = m_paras.eta_t.cols(delta_nu)*(y_paras.nu.rows(delta_nu)-sign(y_paras.nu.rows(delta_nu))*STGP.lambda_nu)/1.00;
        
      }
      // update M-params
      update_alpha();
      update_xi();
      update_sigma_M();

      // update Y-param
      update_beta();
      update_zeta();
      update_gamma();
      update_sigma_Y();

      // update eta-related terms
      if(iter > gibbs_control.start_eta && (iter%gibbs_control.eta_interval) ==0 ){
        update_nu();
        update_eta(H_mat,G);
        if(iter > gibbs_control.start_save_eta){
          eta_counter += 1;
          mcmc_sample.theta_eta_mean += m_paras.theta_eta;
        }
        
      }
      save_paras_sample();
      monitor_gibbs();
      
    }// end of iterations
    
    mcmc_sample.theta_eta_mean /= eta_counter;
  }


  // additional profile functions
  void save_paras_sample(){
    if(iter > gibbs_control.burnin){
      if( (iter - gibbs_control.burnin)%gibbs_control.thinning==0 ){
        int mcmc_iter = (iter - gibbs_control.burnin)/gibbs_control.thinning;
        mcmc_sample.theta_beta.col(mcmc_iter) = y_paras.theta_beta;
        mcmc_sample.theta_alpha.col(mcmc_iter) = m_paras.theta_alpha;
        mcmc_sample.theta_xi.slice(mcmc_iter) = m_paras.theta_xi;
        mcmc_sample.theta_nu.col(mcmc_iter) = y_paras.theta_nu;


        mcmc_sample.gamma(mcmc_iter) = y_paras.gamma;
        mcmc_sample.zeta.col(mcmc_iter) = y_paras.zeta;

        mcmc_sample.inv_sigmasq_M(mcmc_iter) = m_paras.inv_sigmasq_M;
        mcmc_sample.inv_sigmasq_Y(mcmc_iter) = y_paras.inv_sigmasq_Y;
        
        mcmc_sample.loglik_M(mcmc_iter) = profile.loglik_m;
        mcmc_sample.loglik_Y(mcmc_iter) = profile.loglik_y;
        
      }
    }
  };
  

  void monitor_gibbs(){
    if(gibbs_control.verbose > 0){
      if(iter%gibbs_control.verbose==0){
        std::cout << "iter= " << iter << "; inv_sigmasq_M= "<< m_paras.inv_sigmasq_M  << "; inv_sigmasq_Y= "<< y_paras.inv_sigmasq_Y <<  std::endl;
        // std::cout << "inv_sigma_sq_beta:"<<paras.inv_sigma_sq_beta<<std::endl;
        // std::cout << "   inv_sigma_sq_gamma =" << paras.inv_sigma_sq_gamma << 
        //   " inv_sigma_sq_beta: "<< paras.inv_sigma_sq_beta << " inv_a_beta: "<< paras.inv_a_beta <<  std::endl;
      }
    }
  }


  List get_gibbs_sample(){
    return List::create(Named("theta_beta") = mcmc_sample.theta_beta,
                        Named("theta_alpha") = mcmc_sample.theta_alpha,
                        Named("theta_xi") = mcmc_sample.theta_xi,
                        Named("theta_nu") = mcmc_sample.theta_nu,
                        Named("theta_eta_mean") = mcmc_sample.theta_eta_mean,
                        Named("gamma") = mcmc_sample.gamma,
                        Named("zeta") = mcmc_sample.zeta,
                        Named("inv_sigmasq_M") = mcmc_sample.inv_sigmasq_M,
                        Named("inv_sigmasq_Y") = mcmc_sample.inv_sigmasq_Y,
                        Named("loglik_M") = mcmc_sample.loglik_M,
                        Named("loglik_Y") = mcmc_sample.loglik_Y);
  };

  
};


// [[Rcpp::export]]
List Unconfounded_BIMA(const vec& in_y, const mat&in_M, const vec& in_X, const mat& in_C,
                       const List& Kernel_params, const List& init_paras,
                       double lambda_alpha, 
                       double lambda_beta,
                       double lambda_nu,
                       vec step,
                       vec target_acceptance_rate,
                       CharacterVector method = "MALA",
                       int accept_interval=10,
                       int mcmc_sample = 500, 
                       int burnin = 5000, 
                       int thinning = 10,
                       int verbose = 5000,
                       int save_profile = 1,
                       int theta_eta_interval = 10,
                       int start_update_eta = 100,
                       int start_save_eta = 100,
                       bool display_progress = true){
  wall_clock timer;
  timer.tic();
  unconfounded_BIMA model;
  
  model.set_method(method, display_progress);
  model.load_data(in_y, in_M,  in_X,  in_C);
  model.load_STGP(Kernel_params, lambda_alpha, lambda_beta, lambda_nu);
  model.set_initial_params(init_paras);
  
  if(model.get_method()==0){
    model.set_gibbs_control(mcmc_sample,
                            burnin,
                            thinning,
                            verbose,
                            save_profile,
                            theta_eta_interval,
                            start_save_eta,
                            start_update_eta);
    model.set_MALA_control(step, target_acceptance_rate, accept_interval);
    model.run_mcmc();
  }
  
  double elapsed = timer.toc();
  
  List output;
  output = List::create(Named("mcmc") = model.get_gibbs_sample(),
                          Named("elapsed") = elapsed);
  
  return output;
}