#include <RcppArmadillo.h>
using namespace Rcpp;
#include <Rcpp/Benchmark/Timer.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <progress.hpp>
#include <progress_bar.hpp>
// [[Rcpp::depends(RcppProgress)]]

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
// -----------------------block update theta_beta--------------------------------- //
//' Scalar on image regression
//'
//' A basis decomposition is used. The main coefficient beta follows STGP prior.
//' Kernel matrices need to be prespeficified
//'
//' @param Y The scalar outcome, n by 1
//' @param M The image predictor, n by p
//' @param X The scalar exposure variable, n by 1
//' @param C The q confounders, n by q
//' @param L_all A vector of length num_region, each element is an integer to indicate the number of basis in each region
//' @param num_region An integer, the total number of regions
//' @param region_idx A list object of length num_region, each element is a vector of
//' the indices of each voxel in that region. Note that this index starts from 0.
//' @param n_mcmc An integer to indicate the total number of MCMC iterations
//' @param K A list object of length num_region, the r-th element is a p_r by L_r matrix for the basis function
//' @param stop_burnin An integer to indicate from which iteration to stop burnin period.
//' Note that during burinin, the step size in MALA is adjusted every interval_step iterations.
//' @param start_joint An integer to indicate from which iteration to start join update all parameters.
//' Note that this function allows to update beta using MALA alone at the beginning.
//' Users may wait for a few iterations for beta to stabilize and then jointly update all other parameters (using Gibbs Sampling)
//' @param lambda A numeric variable to indicate the thresholding parameter lambda in STGP prior
//' @param target_accept_vec A vector of length num_region. Each element is a numeric variable in (0,1).
//' This allows the user to define different target acceptance rate for each region in the MALA algorithm,
//' and the step size will be adjusted to meet the target acceptance rate.
//' @param a A numeric variable for the Inverse-Gamma(a,b), priors for \eqn{\sigma^2_Y,\sigma^2_\beta}
//' @param b A numeric variable for the Inverse-Gamma(a,b), priors for \eqn{\sigma^2_Y,\sigma^2_\beta}
//' @param init A list object that contains the following element
//' \itemize{
//'   \item theta_beta A vector of length L. Initial value for theta_beta
//'   \item D A vector of length L. Eigenvalues for all regions in the basis
//'   \item gamma A numeric scalar, initial value for gamma
//'   \item cb A numeric scalar, initial value for the intercept term
//'   \item zetay A vector of length q, intial value for zetay
//'   \item sigma_Y A numeric scalar, intial value for sigma_Y
//'   \item sigma_beta A numeric scalar, initial value for sigma_beta
//' }
//' @param step A numeric vector of length num_region, the initial step size for each region
//' @param interval_step An integer to denote how often to update the step size
//' @param interval_thin An integer to denote how often to save the thinned MCMC sample for theta_beta
//' @param display_progress True for displaying progress bar
//' @useDynLib BIMA, .registration=TRUE
//' @import Rcpp
//' @export
//' @return A List object with the following component
//' \itemize{
//' \item theta_beta_mcmc_thin
//' \item logll_mcmc_Y
//' \item track_step
//' \item accept_block
//' \item emp_accept
//' \item gs A list object with the following component
//'   \itemize{
//'     \item zetay_mcmc
//'     \item cy_mcmc
//'     \item sigma_beta2_mcmc
//'     \item sigma_Y2_mcmc
//'   }
//' \item time_seg
//' \item timer
//' }
// [[Rcpp::export(rng = false)]]
List Y_regression_region_block_fast(arma::colvec& Y, arma::mat& M,
                                    arma::colvec& X, arma::mat& C,arma::uvec L_all,
                                    arma::mat& eta, 
                                    arma::uword num_region, Rcpp::List& region_idx,
                                    int n_mcmc, Rcpp::List& K, int stop_burnin,
                                    int start_joint,
                                    double lambda, arma::colvec target_accept_vec,
                                    double a, double b,
                                    Rcpp::List& init,
                                    arma::colvec step,
                                    int interval_step,
                                    int interval_thin,
                                    bool display_progress = true){
    Rcpp::Timer timer;
    timer.step("start of precomputation");
    arma::uword p = M.n_rows;
    arma::uword n = M.n_cols;
    arma::uvec L_cumsum = cumsum(L_all);
    arma::uword L_max = sum(L_all);
    
    arma::mat eta_t = eta.t();
    
    // input
    arma::colvec theta_beta = init["theta_beta"];
    arma::colvec theta_nu = init["theta_nu"];
    arma::colvec D = init["D"];
    arma::colvec D_sqrt = sqrt(D);
    double gamma = init["gamma"], cy = init["cb"];
    arma::colvec zetay = init["zetay"];
    double sigma_Y = init["sigma_Y"], sigma_Y2 = sigma_Y*sigma_Y;
    double sigma_beta = init["sigma_beta"], sigma_beta2 = sigma_beta*sigma_beta;
    double sigma_nu2 = sigma_beta2;
    arma::colvec step_all;
    if(step.n_elem==1){
      step_all = step(0)*arma::ones(num_region);
    }else{
      step_all = step;
    }
    arma::colvec step_all_nu = step_all;
    // hyper parameter for inverse-Gamma
    double sigma_gamma2 = 1.0, sigma_zeta_y2 = 1.0,sigma_cy2=1.0;
    if(C.n_cols !=zetay.n_rows){
      Rcout<<"Error: dimensions of C and zetay don't match!"<<
        "dim of C = "<<size(C)<<"; dim of zetay = "<<size(zetay)<<std::endl;
      return Rcpp::List::create(Rcpp::Named("ERROR")=1);
    }
    
    arma::mat M_t = M.t(); arma::mat C_t = C.t();

    Rcpp::List Q_t(num_region) ;
    arma::colvec beta = arma::zeros(p,1);
    arma::colvec nu = arma::zeros(p,1);
    for(int l=0; l<num_region; l++){
      arma::uvec idx = region_idx[l];
      arma::mat Q = K[l];
      Q_t[l] = Q.t();
      arma::uvec L_idx;
      if(l==0){
        L_idx = arma::linspace<arma::uvec>(0,L_cumsum(l)-1,L_all(l));
      }else{
        L_idx = arma::linspace<arma::uvec>(L_cumsum(l-1),L_cumsum(l)-1,L_all(l));
      }
      beta(idx) = Q*theta_beta(L_idx);
      nu(idx) = Q*theta_nu(L_idx);
    }
    
    arma::colvec Y_star = Y- cy - gamma*X -  C*zetay ;
    
    Rcout<<" test 1"<<std::endl;

    //return
    // arma::mat theta_beta_mcmc = zeros(L_max,n_mcmc);
    arma::mat theta_beta_mcmc_thin = arma::zeros(L_max,n_mcmc/interval_thin);
    arma::mat theta_nu_mcmc_thin = arma::zeros(L_max,n_mcmc/interval_thin);
    arma::colvec logll_mcmc_Y = arma::zeros(n_mcmc);
    arma::colvec logll_mcmc_Y_nosigma = arma::zeros(n_mcmc);
    // arma::mat track_grad_f = zeros(L_max,n_mcmc);
    // mat track_rho = zeros(num_region, n_mcmc);
    arma::mat track_step = arma::zeros(num_region,n_mcmc);
    arma::mat track_step_nu = arma::zeros(num_region,n_mcmc);
    // cube track_rho_compo = zeros(num_region,4, n_mcmc);
    arma::mat emp_accept = arma::zeros( n_mcmc/interval_step,num_region);
    arma::mat emp_accept_nu = emp_accept;
    // arma::colvec accept = zeros(n_mcmc*num_region);
    arma::mat accept_block = arma::zeros(n_mcmc,num_region);
    arma::mat accept_block_nu = accept_block;
    arma::cube time_seg = arma::zeros(n_mcmc,num_region,5);
    // gs returns
    arma::colvec gamma_mcmc = arma::zeros(n_mcmc,1);
    arma::mat zetay_mcmc = arma::zeros(zetay.n_elem,n_mcmc);
    arma::colvec cy_mcmc = arma::zeros(n_mcmc,1);
    arma::colvec sigma_beta2_mcmc = arma::zeros(n_mcmc,1);
    arma::colvec sigma_nu2_mcmc = arma::zeros(n_mcmc,1);
    arma::colvec sigma_Y2_mcmc = arma::zeros(n_mcmc,1);
    
    Rcout<<" test 2"<<std::endl;
    // check if region_idx starts from 0!
    arma::uword min_region_idx=1;
    for(arma::uword l =0; l<num_region;l++){
      arma::uvec region_idx0 = region_idx[l];
      arma::uword min_l = min(region_idx0);
      if(min_l<min_region_idx){min_region_idx=min_l;}
    }

    if(min_region_idx>0){
      Rcout<<"Error: region_idx does not start from 0!"<<
        "min(region_idx[0]) = "<<min_region_idx<<std::endl;
      return Rcpp::List::create(Rcpp::Named("ERROR")=1,
                                Rcpp::Named("region_idx")=region_idx);
    }


    arma::uword all_iter=0;
    arma::uword num_block = num_region;
    // arma::colvec all_log_target_density = arma::zeros(num_region,1);
    arma::uvec delta_in_block, delta_in_block_nu;
    Progress prog(n_mcmc*num_region, display_progress);
    timer.step("start of iteration");
    // Rcout<<"1"<<std::endl;
    for(int iter=0; iter<n_mcmc; iter++){
      // Rcout<<"iter="<<iter<<std::endl;
      if(iter==stop_burnin){
        // start the timer
        timer.step("stop of burnin");        // record the starting point
      }
      // Y_star = Y- cy - gamma*X -  C*zetay;

      // colvec grad_f_all = zeros(L_max,1);
      // colvec rho_all = zeros(num_block,1);
      // mat rho_compo = zeros(num_block,4);
      // Rcout<<" test 3"<<std::endl;

      // start block update
      double log_target_density;
      double log_target_density_new;
      arma::colvec tb_temp;
      arma::colvec temp_new;
      arma::colvec temp;
      double temp_2norm2;
      
      double log_target_density_nu;
      double log_target_density_new_nu;
      arma::colvec temp_nu;
      arma::colvec tb_temp_nu;
      arma::colvec temp_new_nu;
      double temp_2norm2_nu;
      
      for(arma::uword m=0; m < num_region; m++){
        prog.increment();
        
        // ------------- update theta_beta ------------- //
        // ------------- time_seg0 ------------- //
        clock_t t0;t0 = clock();

        arma::uvec delta = find(abs(beta)>lambda);
        arma::uvec idx = region_idx[m];
        arma::uvec delta_Q = find(abs(beta(idx))>lambda);
        arma::mat Q = K[m];
        arma::uvec L_idx;
        if(m==0){
          L_idx = arma::linspace<arma::uvec>(0,L_cumsum(m)-1,L_all(m));
        }else{
          L_idx = arma::linspace<arma::uvec>(L_cumsum(m-1),L_cumsum(m)-1,L_all(m));
        }

        t0 = clock() - t0;
        double sys_t0 = ((double)t0)/CLOCKS_PER_SEC;
        time_seg(iter,m,0) = sys_t0;

        // ------------- time_seg1 ------------- //
        clock_t t1;t1 = clock();

        arma::mat K_block_t = Q_t[m];
        delta_in_block = intersect(idx,delta);
        arma::colvec temp_idx = M_t.cols(delta_in_block)*(beta.rows(delta_in_block)-sign(beta.rows(delta_in_block))*lambda)/1.00;
        

        t1 = clock() - t1;
        double sys_t1 = ((double)t1)/CLOCKS_PER_SEC;
        time_seg(iter,m,1) = sys_t1;

        // ------------- time_seg2 ------------- //
        clock_t t2;t2 = clock();
        if(m==0){
          temp = Y_star-M_t.cols(delta)*(beta.rows(delta)-sign(beta.rows(delta))*lambda)/1.00;
          temp -= eta_t * nu;
          temp_2norm2 = dot(temp,temp);
        }

        arma::colvec grad_f = -theta_beta(L_idx)/D(L_idx)/sigma_beta2+
          K_block_t.cols(delta_Q)*M.rows(delta_in_block)*temp/1.00/sigma_Y2;
        double step = step_all(m);
        arma::colvec theta_beta_diff = step*grad_f+sqrt(2*step)*arma::randn(size(grad_f));
        arma::colvec theta_beta_new_block = theta_beta(L_idx)+theta_beta_diff;
        // grad_f_all(L_idx) = grad_f;
        t2 = clock() - t2;
        double sys_t2 = ((double)t2)/CLOCKS_PER_SEC;
        time_seg(iter,m,2) = sys_t2;





          // MH step

          // ------------- time_seg3 ------------- //
          clock_t t3;t3 = clock();
          tb_temp = theta_beta(L_idx)/D_sqrt(L_idx);
          log_target_density = -0.5*dot(tb_temp,tb_temp)/sigma_beta2-
            0.5*temp_2norm2/sigma_Y2;

          arma::colvec beta_new = beta;
          beta_new(idx) += Q*theta_beta_diff;
          arma::uvec delta_new = arma::find(abs(beta_new)>lambda);
          arma::uvec delta_Q_new = arma::find(abs(beta_new(idx))>lambda);
          arma::uvec delta_in_block_new = intersect(idx,delta_new);

          arma::colvec x1 = arma::zeros(n,1); arma::colvec x2 = arma::zeros(n,1);
          bool b1 = delta_in_block.n_elem>0;
          bool b2 = delta_in_block_new.n_elem>0;
          if(b1){
            x1 = M_t.cols(delta_in_block)*(beta.rows(delta_in_block)-sign(beta.rows(delta_in_block))*lambda)/1.00;
          }
          if(b2){
            x2 = M_t.cols(delta_in_block_new)*(beta_new.rows(delta_in_block_new)-sign(beta_new.rows(delta_in_block_new))*lambda)/1.00;
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

          t3 = clock() - t3;
          double sys_t3 = ((double)t3)/CLOCKS_PER_SEC;
          time_seg(iter,m,3) = sys_t3;

          // ------------- time_seg4 ------------- //
          clock_t t4;t4 = clock();
          tb_temp = theta_beta_new_block/D_sqrt(L_idx);
          log_target_density_new = -0.5*dot(tb_temp,tb_temp)/sigma_beta2-
            0.5*dot(temp_new,temp_new)/sigma_Y2;
          arma::colvec grad_f_new = -theta_beta_new_block/D(L_idx)/sigma_beta2+
            K_block_t.cols(delta_Q_new)*M.rows(delta_in_block_new)*temp_new/1.00/sigma_Y2; // L x 1

          double log_q = -1/4/step * square(norm(-theta_beta_diff-step*grad_f_new,2));//accu, change this part, %
          double log_q_new = -1/4/step * square(norm(theta_beta_diff-step*grad_f,2));
          double rho = log_target_density_new + log_q - log_target_density - log_q_new;
        
          // Rcout<<"iter = "<<iter<<"log_target_density_new = "<<log_target_density_new<<std::endl;
          // Rcout<<"iter = "<<iter<<"log_q = "<<log_q<<std::endl;
          // Rcout<<"iter = "<<iter<<"log_target_density = "<<log_target_density<<std::endl;
          // Rcout<<"iter = "<<iter<<"log_q_new = "<<log_q_new<<std::endl;
          // Rcout<<"iter = "<<iter<<"rho = "<<rho<<std::endl;
          
          if(log(arma::randu())<=rho){
            theta_beta(L_idx) = theta_beta_new_block;
            beta = beta_new;
            temp = temp_new;
            log_target_density=log_target_density_new;
            //== below is the previous incorrect code, left as comment just for comparison
            // all_log_target_density(m) = log_target_density_new;
            accept_block(iter,m) = 1;
          }

          t4 = clock() - t4;
          double sys_t4 = ((double)t4)/CLOCKS_PER_SEC;
          time_seg(iter,m,4) = sys_t4;
          
          // ------------- update theta_nu ------------- //
          // ------------- time_seg0 ------------- //
          arma::uvec delta_nu = find(abs(nu)>lambda);
          arma::uvec delta_Q_nu = find(abs(nu(idx))>lambda);
          
          // ------------- time_seg1 ------------- //
          
          delta_in_block_nu = intersect(idx,delta_nu);
          arma::colvec temp_idx_nu = eta_t.cols(delta_in_block_nu)*(
            nu.rows(delta_in_block_nu)-sign(nu.rows(delta_in_block_nu))*lambda )/1.00;
          
          
          // ------------- time_seg2 ------------- //
          if(m==0){
            temp_nu = Y_star-eta_t.cols(delta_nu)*(nu.rows(delta_nu)-sign(nu.rows(delta_nu))*lambda)/1.00;
            temp_nu -= M_t * beta;
            temp_2norm2_nu = dot(temp_nu,temp_nu);
          }
          
          arma::colvec grad_f_nu = -theta_nu(L_idx)/D(L_idx)/sigma_nu2+
            K_block_t.cols(delta_Q_nu)*eta.rows(delta_in_block_nu)*temp_nu/1.00/sigma_Y2;
          double step_nu = step_all_nu(m);
          arma::colvec theta_nu_diff = step_nu*grad_f_nu+sqrt(2*step_nu)*arma::randn(size(grad_f_nu));
          arma::colvec theta_nu_new_block = theta_nu(L_idx)+theta_nu_diff;
          
          // MH step
          
          // ------------- time_seg3 ------------- //
          tb_temp_nu = theta_nu(L_idx)/D_sqrt(L_idx);
          log_target_density_nu = -0.5*dot(tb_temp_nu,tb_temp_nu)/sigma_nu2-
            0.5*temp_2norm2_nu/sigma_Y2;
          
          arma::colvec nu_new = nu;
          nu_new(idx) += Q*theta_nu_diff;
          arma::uvec delta_new_nu = arma::find(abs(nu_new)>lambda);
          arma::uvec delta_Q_new_nu = arma::find(abs(nu_new(idx))>lambda);
          arma::uvec delta_in_block_new_nu = intersect(idx,delta_new_nu);
          
          arma::colvec x1_nu = arma::zeros(n,1); arma::colvec x2_nu = arma::zeros(n,1);
          bool b1_nu = delta_in_block_nu.n_elem>0;
          bool b2_nu = delta_in_block_new_nu.n_elem>0;
          if(b1_nu){
            x1_nu = eta_t.cols(delta_in_block_nu)*(nu.rows(delta_in_block_nu)-sign(nu.rows(delta_in_block_nu))*lambda)/1.00;
          }
          if(b2_nu){
            x2_nu = eta_t.cols(delta_in_block_new_nu)*(nu_new.rows(delta_in_block_new_nu)-sign(nu_new.rows(delta_in_block_new_nu))*lambda)/1.00;
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
          tb_temp_nu = theta_nu_new_block/D_sqrt(L_idx);
          log_target_density_new_nu = -0.5*dot(tb_temp_nu,tb_temp_nu)/sigma_nu2-
            0.5*dot(temp_new_nu,temp_new_nu)/sigma_Y2;
          arma::colvec grad_f_new_nu = -theta_nu_new_block/D(L_idx)/sigma_nu2+
            K_block_t.cols(delta_Q_new_nu)*eta.rows(delta_in_block_new_nu)*temp_new_nu/1.00/sigma_Y2; // L x 1
          
          double log_q_nu = -1/4/step_nu * square(norm(-theta_nu_diff-step_nu*grad_f_new_nu,2));//accu, change this part, %
          double log_q_new_nu = -1/4/step_nu * square(norm(theta_nu_diff-step_nu*grad_f_nu,2));
          double rho_nu = log_target_density_new_nu + log_q_nu - log_target_density_nu - log_q_new_nu;

          // Rcout<<"iter = "<<iter<<"log_target_density_new_nu = "<<log_target_density_new_nu<<std::endl;
          // Rcout<<"iter = "<<iter<<"log_q_nu = "<<log_q_nu<<std::endl;
          // Rcout<<"iter = "<<iter<<"log_target_density_nu = "<<log_target_density_nu<<std::endl;
          // Rcout<<"iter = "<<iter<<"log_q_new_nu = "<<log_q_new_nu<<std::endl;
          // Rcout<<"iter = "<<iter<<"rho_nu = "<<rho_nu<<std::endl;
          
          if(log(arma::randu())<=rho_nu){
            theta_nu(L_idx) = theta_nu_new_block;
            nu = nu_new;
            temp_nu = temp_new_nu;
            log_target_density_nu = log_target_density_new_nu;
            //== below is the previous incorrect code, left as comment just for comparison
            // all_log_target_density(m) = log_target_density_new;
            accept_block_nu(iter,m) = 1;
          }


      }// end of block update
      if( (iter%interval_step==0) & (iter>0)  ){
        arma::uvec u = arma::linspace<arma::uvec>(iter-interval_step,iter-1,interval_step);
        emp_accept.row(iter/interval_step-1) = mean(accept_block.rows(u),0);
        emp_accept_nu.row(iter/interval_step-1) = mean(accept_block_nu.rows(u),0);
        if(iter<stop_burnin){

          for(arma::uword l = 0; l<num_block; l++){
            step_all(l)  = adjust_acceptance(emp_accept(iter/interval_step-1,l),step_all(l),target_accept_vec(l));
            if(step_all(l)>1){step_all(l)=1;}
            step_all_nu(l)  = adjust_acceptance(emp_accept_nu(iter/interval_step-1,l),step_all_nu(l),target_accept_vec(l));
            if(step_all_nu(l)>1){step_all_nu(l)=1;}
          }

        }

      }
      // Rcout<<"test 1-3"<<std::endl;
      // Rcout<<"stop_burnin = "<<stop_burnin<<std::endl;
      // Rcout<<"n_mcmc/100 = "<<n_mcmc/100<<std::endl;
      // Rcout<<"(n_mcmc - stop_burnin)/(n_mcmc/100)= "<<(n_mcmc - stop_burnin)/(n_mcmc/100.0)<<std::endl;
      // when stop burnin, choose the average of last few steps
      arma::uword back =  ceil( (n_mcmc - stop_burnin)/(n_mcmc/100.0) );
      // Rcout<<"back = "<<back <<std::endl;
      if(iter==stop_burnin & stop_burnin>back){
        arma::uvec u = arma::linspace<arma::uvec>(iter-back,iter-1,back);
        // arma::mat last_steps = track_step.cols(u);
        step_all = exp(mean(log(track_step.cols(u)),1));
        step_all_nu = exp(mean(log(track_step_nu.cols(u)),1));
        // uvec large_step_idx = arma::find(step_all>1e-4);
        // step_all(large_step_idx) = 1e-4*arma::ones(large_step_idx.n_elem);
      }
      if( (iter%interval_step==0) & (iter>0)  ){
        arma::uvec u = arma::linspace<arma::uvec>(iter-interval_step,iter-1,interval_step);
        emp_accept.row(iter/interval_step-1) = mean(accept_block.rows(u),0);
        emp_accept_nu.row(iter/interval_step-1) = mean(accept_block_nu.rows(u),0);
      }
      // Rcout<<"test 1-4"<<std::endl;

      track_step.col(iter) = step_all;
      track_step_nu.col(iter) = step_all_nu;
      all_iter = all_iter+1;
      
      // update nu


      // -------------- Update GS for all other parameters --------------
      if(iter>start_joint){
        arma::uvec delta = find(abs(beta)>lambda);
        arma::uvec delta_nu = find(abs(nu)>lambda);
        arma::colvec M_beta_term = M_t.cols(delta)*(beta.rows(delta)-sign(beta.rows(delta))*lambda)/1.00;
        // test
        arma::colvec eta_nu_term = eta_t.cols(delta_nu)*(nu.rows(delta_nu)-sign(nu.rows(delta_nu))*lambda)/1.00;
        
        
        // 2. gamma
        double post_sigma_gamma2 = 1/(arma::sum(arma::dot(X,X))/sigma_Y2 + 1/sigma_gamma2);
        double temp2 = dot(X, Y - cy - M_beta_term - C * zetay - eta_nu_term);
        double mu_gamma = post_sigma_gamma2 * temp2/sigma_Y2;
        gamma = arma::randn() * sqrt(post_sigma_gamma2) + mu_gamma;
        gamma_mcmc(iter) = gamma;
        //
        // // 3. zeta_y
        arma::mat Sigma_zetay_inv = 1/sigma_Y2*(C_t*C) + 1/sigma_zeta_y2;
        arma::mat Sigma_zetay = inv_sympd(Sigma_zetay_inv );
        arma::colvec mu_zetay = Sigma_zetay* C_t *(  Y - cy - M_beta_term - eta_nu_term - gamma*X )/sigma_Y2;
        zetay = arma::mvnrnd( mu_zetay, Sigma_zetay);
        zetay_mcmc.col(iter) = zetay;
        //
        // //   4. c_y
        double post_sigma_cy = 1/(Y.n_elem/sigma_Y2 + 1/sigma_cy2);
        arma::colvec temp_cy = Y - M_beta_term - C * zetay - gamma*X - eta_nu_term;
        double mu_cy = post_sigma_cy * sum(temp_cy)/sigma_Y2;
        cy = arma::randn() * sqrt(post_sigma_cy) + mu_cy;
        cy_mcmc(iter) = cy;
        //
        // //   5. sigma_beta
        double sigma_beta_a = a + theta_beta.n_elem/2;
        double sigma_beta_b = b + dot(theta_beta,theta_beta/D)/2;
        sigma_beta2 = 1/arma::randg( arma::distr_param(sigma_beta_a,1/sigma_beta_b) );
        sigma_beta2_mcmc(iter) = sigma_beta2;
        
        // // sigma_nu
        double sigma_nu_a = a + theta_nu.n_elem/2;
        double sigma_nu_b = b + dot(theta_nu,theta_nu/D)/2;
        sigma_nu2 = 1/arma::randg( arma::distr_param(sigma_nu_a,1/sigma_nu_b) );
        sigma_nu2_mcmc(iter) = sigma_nu2;
        
        // //   6. sigma_Y
        arma::colvec temp_sigma_Y = temp_cy-cy;
        double sigma_Y_b = b + dot(temp_sigma_Y,temp_sigma_Y)/2;
        sigma_Y2 = 1/arma::randg( arma::distr_param(a + Y.n_elem/2,1/sigma_Y_b) );
        sigma_Y2_mcmc(iter) = sigma_Y2;

        // 7. update lambda

      }

      // --------------------- summarize return --------------------- //

      // theta_beta_mcmc.col(iter) = theta_beta;
      arma::uvec delta_Y = find(abs(beta)>lambda);
      arma::uvec delta_nu = find(abs(nu)>lambda);
      Y_star = Y- cy - gamma*X -  C*zetay;
      arma::colvec temp_Y = Y_star - M_t.cols(delta_Y)*(beta.rows(delta_Y)-sign(beta.rows(delta_Y))*lambda)/1.00;
      temp_Y -= eta_t.cols(delta_nu)*(nu.rows(delta_nu)-sign(nu.rows(delta_nu))*lambda);
      logll_mcmc_Y(iter) = -dot(temp_Y,temp_Y)/2/sigma_Y2;
      logll_mcmc_Y_nosigma(iter) = logll_mcmc_Y(iter);
      logll_mcmc_Y(iter) += -0.5*n*log(sigma_Y2);

      if( (iter%interval_thin==0) & (iter>0) ){
        theta_beta_mcmc_thin.col(iter/interval_thin-1)=theta_beta;
        theta_nu_mcmc_thin.col(iter/interval_thin-1)=theta_nu;
      }
    }
    timer.step("end of iterations");
    List gs = Rcpp::List::create(Rcpp::Named("gamma_mcmc")=gamma_mcmc,
                                 Rcpp::Named("zetay_mcmc")= zetay_mcmc,
                                 Rcpp::Named("cy_mcmc")= cy_mcmc,
                                 Rcpp::Named("sigma_beta2_mcmc")= sigma_beta2_mcmc,
                                 Rcpp::Named("sigma_nu2_mcmc")= sigma_nu2_mcmc,
                                 Rcpp::Named("sigma_Y2_mcmc")= sigma_Y2_mcmc);

    return Rcpp::List::create(
      Rcpp::Named("theta_beta_mcmc_thin")=theta_beta_mcmc_thin,
      Rcpp::Named("theta_nu_mcmc_thin")=theta_nu_mcmc_thin,
      Rcpp::Named("logll_mcmc_Y")=logll_mcmc_Y,
      Rcpp::Named("logll_mcmc_Y_nosigma")=logll_mcmc_Y_nosigma,
      Rcpp::Named("track_step")=track_step,
      Rcpp::Named("track_step_nu")=track_step_nu,
      Rcpp::Named("accept_blcok")=accept_block,
      Rcpp::Named("emp_accept")=emp_accept,
      Rcpp::Named("emp_accept_nu")=emp_accept_nu,
      Rcpp::Named("accept_block_nu")=accept_block_nu,
      Rcpp::Named("gs") = gs,
      Rcpp::Named("time_seg") = time_seg,
      Rcpp::Named("Timer")=timer
    );
  }
// [[Rcpp::export(rng = false)]]
List get_H_mat(const::arma::mat G){
  arma::mat G_null = null(G);
  arma::mat H_inv = join_vert(G_null.t(),G);
  arma::mat H = inv(H_inv);
  return List::create(Named("H") = H,
                      Named("H_inv") = H_inv,
                      Named("G_null_t") = G_null.t());
}
// [[Rcpp::export(rng = false)]]
arma::mat hyperplane_MVN_multiple(const::arma::mat G,
                                  const::List H_mat,
                                  const::arma::vec sigma2_vec,
                                  const::arma::mat mu_mat){
  // prepare H mat
  arma::mat H = H_mat["H"];
  arma::mat H_inv = H_mat["H_inv"];
  arma::mat G_null_t = H_mat["G_null_t"];
  arma::uword q = G.n_rows;
  arma::uword n = G.n_cols;
  arma::mat H1 = H.cols(0,n-q-1);
  arma::mat H2 = H.cols(n-q,n-1);

  // get mean and variance for z1
  arma::mat Lambda11 = H1.t()* H1;
  arma::mat Lambda12 = H1.t()* H2;
  arma::mat Lambda11_inv = inv(Lambda11);
  arma::mat Lambda11_inv_sqrt = sqrtmat_sympd(Lambda11_inv);

  arma::mat mu_z1 = ( G_null_t + Lambda11_inv * Lambda12 * G * H_inv ) * mu_mat; // n by m
  arma::uword m = sigma2_vec.n_elem;
  arma::mat z1 =   Lambda11_inv_sqrt *  arma::randn(n-q,m);
  z1.each_row() %= sqrt(sigma2_vec.t());
  z1 += mu_z1;
  arma::mat x = H1*z1;
  return x.t();
}



arma::uvec complement(arma::uword start, arma::uword end, arma::uword n) {
  arma::uvec y1 = arma::linspace<arma::uvec>(0, start-1, start);
  arma::uvec y2 = arma::linspace<arma::uvec>(end+1, n-1, n-1-end);
  arma::uvec y = arma::join_cols(y1,y2);
  return y;
}
// -----------------------GS update to give initial values--------------------------------- //
//' Image on scalar regression with GP prior for alpha instead of STGP
//'
//' This is not a main function in BIMA package. It only provides an alternative to the STGP prior.
//' A basis decomposition is used. The main coefficient beta follows STGP prior.
//' Kernel matrices need to be prespeficified
//'
//' @param data A List that contains
//' \itemize{
//' \item M The image predictor, n by p
//' \item X The scalar exposure variable, n by 1
//' \item C The q confounders, q by n
//' }
//' @param region_idx_cpp A list object of length num_region, each element is a vector of
//' the indices of each voxel in that region. Note that this index starts from 0.
//' @param init A list object that contains the following element
//' \itemize{
//'   \item zetam A vector of length q, initial value for zetam
//'   \item sigma_alpha A numeric scalar, intial value for sigma_alpha
//'   \item sigma_M A numeric scalar, intial value for sigma_M
//'   \item theta_eta A n by L matrix, initial value for theta_eta
//'   \item sigma_eta A numeric scalar, intial value for sigma_eta
//'   \item theta_alpha A vector of length L, intial value for theta_alpha
//'   \item a A numeric variable for the Inverse-Gamma(a,b), priors for \eqn{\sigma^2_Y,\sigma^2_\beta}
//'   \item b A numeric variable for the Inverse-Gamma(a,b), priors for \eqn{\sigma^2_Y,\sigma^2_\beta}
//'   \item sigma_zetam A numeric scalar, intial value for sigma_zetam
//' }
//' @param kernel A List object
//' \itemize{
//'   \item D A list of length num_region, each element contains the eigen-values in one region
//'   \item Q A list of length num_region, each element contains the basis function (p_r by L_r) in one region
//' }
//' @param n_mcmc An integer to indicate the total number of MCMC iterations
//' @param display_progress True for displaying progress bar
//' @useDynLib BIMA, .registration=TRUE
//' @import Rcpp
//' @export
//' @return A list of
//' \itemize{
//'   \item theta_eta
//'   \item zetam_mcmc
//'   \item sigma_M2_inv_mcmc
//'   \item sigma_alpha2_inv_mcmc
//'   \item sigma_zetam2_inv_mcmc
//'   \item sigma_eta2_inv_mcmc
//'   \item logLL_mcmc
//'   \item Timer
//' }
// [[Rcpp::export(rng = false)]]
List M_regression_GS(Rcpp::List& data, Rcpp::List& init,
                     Rcpp::List& region_idx_cpp,
                     Rcpp::List& kernel, arma::uword n_mcmc,
                     bool display_progress = true){
  // input data
  Rcpp::Timer timer;
  timer.step("start of precomputation");
  arma::vec  X = data["X"];
  arma::mat  C = data["C"];
  arma::mat  M = data["M"];
  arma::uword n = X.n_elem;
  // input initial parameters
  arma::vec   zetam = init["zetam"];
  double sigma_alpha = init["sigma_alpha"];
  double sigma_M = init["sigma_M"];
  arma::mat theta_eta = init["theta_eta"];
  double sigma_eta = init["sigma_eta"];
  arma::vec  theta_alpha = init["theta_alpha"];
  double a = init["a"]; double b = init["b"];
  double sigma_zetam = init["sigma_zetam"];
  double sigma_alpha2_inv = 1/sigma_alpha/sigma_alpha;
  double sigma_M2_inv = 1/sigma_M/sigma_M;
  double sigma_zetam2_inv = 1/sigma_zetam/sigma_zetam;
  double sigma_eta2_inv = 1/sigma_eta/sigma_eta;
  double X2_sum = dot(X,X);

  // input basis functions
  Rcpp::List D = kernel["D"];Rcpp::List Q = kernel["Q"];


  // get M_star, D_vec, p, L from basis
  arma::uword num_region = region_idx_cpp.length();
  arma::uvec p_length; arma::uvec L_all;
  p_length.set_size(num_region); L_all.set_size(num_region);
  arma::mat M_star;
  arma::vec  D_vec; arma::vec  q_vec;
  for(arma::uword i=0; i<num_region; i++){
    arma::mat Q_i = Q[i];
    p_length(i) = Q_i.n_rows; L_all(i)=Q_i.n_cols;
    arma::uvec  idx = region_idx_cpp[i];
    M_star = arma::join_cols(M_star, Q_i.t()*M.rows(idx));
    arma::vec  D_i = D[i];
    D_vec = arma::join_cols(D_vec, D_i);
    arma::rowvec q_i = arma::sum(Q_i,0);
    q_vec = arma::join_cols(q_vec, q_i.t());
  }
  arma::uword L = arma::sum(L_all);
  // arma::uword p = arma::sum(p_length);
  double q2_sum = dot(q_vec,q_vec);
  // M_star = M_star/sqrt(p);
  arma::mat M_star_t = M_star.t();

  arma::uword m = zetam.n_elem;
  arma::mat C_t = C.t();

  // initialize mcmc sequences
  arma::mat theta_alpha_mcmc = arma::zeros(L,n_mcmc );
  arma::mat zetam_mcmc = arma::zeros(m,n_mcmc );
  arma::vec  sigma_M2_inv_mcmc = arma::zeros(n_mcmc,1);
  arma::vec  sigma_alpha2_inv_mcmc = arma::zeros(n_mcmc,1);
  arma::vec  sigma_zetam2_inv_mcmc = arma::zeros(n_mcmc,1);
  arma::vec  sigma_eta2_inv_mcmc = arma::zeros(n_mcmc,1);
  arma::vec  logLL_mcmc = arma::zeros(n_mcmc,1);
  arma::mat zeta_term =  q_vec*zetam.t()*C;
  Progress prog(n_mcmc*num_region, display_progress);
  timer.step("start of iteration");
  for(arma::uword iter=0; iter< n_mcmc; iter++){
    prog.increment();

    // 1. update theta_alpha
    arma::mat M_res = M_star - zeta_term - theta_eta;
    arma::vec  Sigma_theta_alpha = 1/(sigma_alpha2_inv/D_vec + sigma_M2_inv*X2_sum);
    M_res.each_row() %= X.t();
    arma::vec   mu_theta_alpha = arma::sum(M_res,1)%Sigma_theta_alpha*sigma_M2_inv;
    theta_alpha =  arma::randn(L,1)%sqrt(Sigma_theta_alpha) + mu_theta_alpha;
    theta_alpha_mcmc.col(iter) =  mu_theta_alpha;

    // 2. zetam

    arma::mat Sigma_zetam_inv = sigma_zetam2_inv + sigma_M2_inv*q2_sum*C*C_t;
    arma::mat Sigma_zetam = inv_sympd(Sigma_zetam_inv );
    arma::mat zeta_res = M_star - theta_alpha*X.t() - theta_eta;
    arma::rowvec temp_zeta = q_vec.t()*zeta_res;
    arma::rowvec mu_zetam = temp_zeta*C_t *Sigma_zetam* sigma_M2_inv;
    zetam = arma::mvnrnd( mu_zetam.t(), Sigma_zetam);
    zetam_mcmc.col(iter) = zetam;


    // 3. theta_eta
    zeta_term =  q_vec*zetam.t()*C;
    arma::mat  eta_res = M_star - theta_alpha*X.t() - zeta_term; //L x n
    arma::vec   eta_Sigma_vec = 1/(sigma_M2_inv + sigma_eta2_inv/D_vec);
    arma::vec   mu_eta_i;
    // arma::mat mu_eta;
    for(arma::uword i=0;i<n;i++){
      mu_eta_i = (eta_res.col(i)*sigma_eta2_inv)%eta_Sigma_vec;
      arma::vec   theta_eta_i = arma::randn(L)%arma::sqrt(eta_Sigma_vec)+ mu_eta_i;
      theta_eta.col(i) = theta_eta_i - arma::mean(theta_eta_i);
      // mu_eta = join_rows(mu_eta,mu_eta_i);
      // theta_eta.col(i) = mu_eta;
    }


    // 4. sigma_alpha
    sigma_alpha2_inv = arma::randg( arma::distr_param(a + L*0.5, 1/(b + dot(theta_alpha,theta_alpha/D_vec)/2)) );
    sigma_alpha2_inv_mcmc(iter) = sigma_alpha2_inv;
    // 5. sigma_M
    arma::mat M_reg_res = M_star - theta_alpha*X.t() - zeta_term - theta_eta;
    double M_norm = norm(M_reg_res,"fro");
    sigma_M2_inv = arma::randg( arma::distr_param(a + n*L/2,1/(b + M_norm*M_norm/2) ) );
    sigma_M2_inv_mcmc(iter) = sigma_M2_inv;

    //   6. sigma_zetam -> half-cauchy
    // sigma_zetam2_inv = arma::randg( arma::distr_param(a + zetam.n_elem/2,1/(b + dot(zetam,zetam)/2) ) );
    // sigma_zetam2_inv_mcmc(iter) = sigma_zetam2_inv;

    // //   7. sigma_eta -> giving too large variance
    double eta_norm = norm(theta_eta,"fro");
    sigma_eta2_inv = arma::randg( arma::distr_param(a + 0.5*n*L,1/(b + eta_norm*eta_norm/2)) );
    sigma_eta2_inv_mcmc(iter) = sigma_eta2_inv;

    // logLL
    arma::mat res = M_star - zeta_term - theta_eta - theta_alpha*X.t();
    double res_norm = norm(M_star,"fro");
    logLL_mcmc(iter) = -0.5*sigma_M2_inv*res_norm*res_norm;
  }
  timer.step("end of iterations");

  return Rcpp::List::create(Rcpp::Named("theta_alpha_mcmc")=theta_alpha_mcmc,
                            Rcpp::Named("theta_eta") = theta_eta,
                            Rcpp::Named("zetam_mcmc")= zetam_mcmc,
                            Rcpp::Named("sigma_M2_inv_mcmc")= sigma_M2_inv_mcmc ,
                            Rcpp::Named("sigma_alpha2_inv_mcmc")= sigma_alpha2_inv_mcmc,
                            Rcpp::Named("sigma_zetam2_inv_mcmc")=  sigma_zetam2_inv_mcmc,
                            Rcpp::Named("sigma_eta2_inv_mcmc")= sigma_eta2_inv_mcmc,
                            Rcpp::Named("logLL_mcmc")= logLL_mcmc,
                            Rcpp::Named("Timer")=timer);
}


// // -----------------------block update theta_alpha--------------------------------- //
//' Image on scalar regression
//'
//' A basis decomposition is used. The main coefficient alpha follows STGP prior.
//' Kernel matrices need to be pre-speficified
//'
//' @param M The image predictor, n by p
//' @param X The scalar exposure variable, n by 1
//' @param C The q confounders, n by q
//' @param L_all A vector of length num_region, each element is an integer to indicate the number of basis in each region
//' @param num_region An integer, the total number of regions
//' @param region_idx A list object of length num_region, each element is a vector of
//' the indices of each voxel in that region. Note that this index starts from 0.
//' @param n_mcmc An integer to indicate the total number of MCMC iterations
//' @param K A list object of length num_region, the r-th element is a p_r by L_r matrix for the basis function
//' @param stop_burnin An integer to indicate from which iteration to stop burnin period.
//' Note that during burinin, the step size in MALA is adjusted every interval_step iterations.
//' @param lambda A numeric variable to indicate the thresholding parameter lambda in STGP prior
//' @param target_accept_vec A vector of length num_region. Each element is a numeric variable in (0,1).
//' This allows the user to define different target acceptance rate for each region in the MALA algorithm,
//' and the step size will be adjusted to meet the target acceptance rate.
//' @param a A numeric variable for the Inverse-Gamma(a,b), priors for \eqn{\sigma^2_Y,\sigma^2_\beta}
//' @param b A numeric variable for the Inverse-Gamma(a,b), priors for \eqn{\sigma^2_Y,\sigma^2_\beta}
//' @param init A list object that contains the following element
//' \itemize{
//'   \item theta_alpha A vector of length L. Initial value for theta_beta
//'   \item theta_zetam A L by q matrix. Initial value for theta_zetam
//'   \item theta_eta A matrix (L by n). Initial value for theta_eta
//'   \item D A vector of length L. Eigenvalues for all regions in the basis
//'   \item sigma_M A numeric scalar, initial value for gamma
//'   \item zetay A vector of length q, intial value for zetay
//'   \item sigma_alpha A numeric scalar, intial value for sigma_alpha
//'   \item sigma_eta A numeric scalar, initial value for sigma_eta
//' }
//' @param step A numeric vector of length num_region, the initial step size for each region
//' @param interval An integer to denote how often to update the step size
//' @param interval_eta An integer to denote how often to update theta_eta
//' @param thinning An integer to indicate how often to save the MCMC samples for theta_alpha
//' @param display_progress True for displaying progress bar
//' @import Rcpp
//' @useDynLib BIMA, .registration=TRUE
//' @export
//' @return A List object with the following component
//' \itemize{
//' \item theta_alpha_mcmc
//' \item logll_mcmc
//' \item track_step
//' \item accept_block
//' \item emp_accept
//' \item gs A list object with the following component
//'   \itemize{
//'     \item theta_zetam_mcmc
//'     \item sigma_M2_inv_mcmc
//'     \item sigma_alpha2_inv_mcmc
//'     \item sigma_eta2_inv_mcmc
//'   }
//' \item Timer
//' }
// [[Rcpp::export(rng = false)]]
List M_regression_region_block( arma::mat& M,
                                arma::colvec& X, arma::mat& C,arma::uvec L_all,
                                arma::uword num_region, Rcpp::List& region_idx,
                                int n_mcmc, Rcpp::List& K, int stop_burnin,
                                double lambda, arma::colvec& target_accept_vec,
                                Rcpp::List& init,
                                int interval,arma::vec   step,
                                double a=1, double b=1,
                                int interval_eta = 10,
                                int thinning = 10,
                                bool display_progress = true){
  Rcpp::Timer timer;
  // set_seed(1);
  timer.step("start of precomputation");
  arma::uword p = M.n_rows;
  arma::uword n = M.n_cols;

  arma::uvec L_cumsum = cumsum(L_all);
  arma::uword L_max = L_cumsum(num_region-1);
  // input
  arma::vec   theta_alpha = init["theta_alpha"];
  arma::mat theta_zetam =  init["theta_zetam"]; //L by q
  arma::vec   D = init["D"];
  arma::uword q = theta_zetam.n_cols;

  double sigma_M = init["sigma_M"], sigma_M2 = sigma_M*sigma_M, sigma_M2_inv = 1/sigma_M2;
  double sigma_alpha = init["sigma_alpha"], sigma_alpha2 = sigma_alpha*sigma_alpha, sigma_alpha2_inv = 1/sigma_alpha2;
  double sigma_eta = init["sigma_eta"], sigma_eta2_inv = 1/sigma_eta/sigma_eta;
  arma::mat theta_eta = init["theta_eta"];double sigma_zetam2_inv = 100;

  // prepare for constrained eta update
  arma::mat G = (join_horiz(X,C)).t();
  // Rcout<<"size of G"<<G.n_rows<<","<<G.n_cols<<std::endl;
  List H_mat = get_H_mat(G);
  // Rcout<<"H arma::mat successful!"<<std::endl;

  arma::vec  step_all;
  if(step.n_elem==1){
    step_all = step(0)*arma::ones(num_region);
  }else{
    step_all = step;
  }

  // C should be n by q
  // if(C.n_cols !=q){
  //   Rcout<<"Error: dimensions of C and zetam don't match!"<<
  //     "dim of C = "<<size(C)<<"; dim of theta_zetam = "<<size(theta_zetam)<<std::endl;
  //   return Rcpp::List::create(Rcpp::Named("ERROR")=1);
  // }
  arma::mat C_t = C.t();
  List M_star_pre_eta(num_region);
  arma::vec  alpha = arma::zeros(p,1);
  arma::mat zetam = arma::zeros(p,q);
  arma::vec   X2_sum_allsample_q = arma::zeros(q,1);
  arma::mat XcXq_sumsq = arma::zeros(q-1,q); // arma::sum_i C[-j]*C_j
  arma::vec  XXq_sumsq = arma::zeros(q,1); // arma::sum_i C*C_j
  for(int j=0; j<q; j++){
    X2_sum_allsample_q(j) += arma::sum(C_t.row(j) %C_t.row(j));
    // XqYstar_term_allsample.col(j) += Y_star_b * trans(X_q.row(j));
    XXq_sumsq(j) += accu(X.t() %C_t.row(j));
    arma::uvec c_j = complement(j, j, q);
    XcXq_sumsq.col(j) += C_t.rows(c_j) * trans(C_t.row(j));
  }

  for(int l=0; l<num_region; l++){
    arma::uvec idx = region_idx[l];
    arma::mat Q = K[l];
    // M_star_pre_eta[l] = Q_t*(M.rows(idx) - arma::ones(idx.n_elem,1)*zetam.t()*C);
    arma::uvec L_idx;
    if(l==0){
      L_idx = arma::linspace<arma::uvec>(0,L_cumsum(l)-1,L_all(l));
    }else{
      L_idx = arma::linspace<arma::uvec>(L_cumsum(l-1),L_cumsum(l)-1,L_all(l));
    }

    alpha(idx) = Q*theta_alpha(L_idx);
    zetam.rows(idx) = Q*theta_zetam.rows(L_idx);


  }


  //return
  int total_mcmc = n_mcmc/thinning;
  arma::mat  theta_alpha_mcmc = arma::zeros(L_max,total_mcmc);
  arma::vec   logll_mcmc = arma::zeros(total_mcmc);
  arma::mat track_step = arma::zeros(num_region,n_mcmc);
  arma::mat emp_accept = arma::zeros( n_mcmc/interval,num_region);
  arma::vec   accept = arma::zeros(n_mcmc*num_region);
  arma::mat accept_block = arma::zeros(n_mcmc,num_region);
  arma::uword all_iter=0;
  arma::uword num_block = num_region;
  arma::mat theta_eta_res(L_max,n);
  arma::mat M_reg_res(L_max,n);

  // GS: initialize mcmc sequences
  arma::cube theta_zetam_mcmc = arma::zeros(L_max,q,total_mcmc );
  arma::vec  sigma_M2_inv_mcmc = arma::zeros(total_mcmc,1);
  arma::vec  sigma_alpha2_inv_mcmc = arma::zeros(total_mcmc,1);
  arma::mat theta_eta_temp(L_max,n);
  arma::vec  sigma_eta2_inv_mcmc = arma::zeros(total_mcmc,1);
  Progress prog(n_mcmc*num_region, display_progress);
  timer.step("start of iteration");
  for(int iter=0; iter<n_mcmc; iter++){
    double logll_M = 0;

    if(iter==stop_burnin){
      // start the timer
      timer.step("stop of burnin");        // record the starting point
    }
    // arma::vec  grad_f_all = arma::zeros(L_max,1);
    // arma::vec  rho_all = arma::zeros(num_block,1);
    // arma::mat rho_compo = arma::zeros(num_block,4);
    arma::vec   alpha_new = alpha;

    // Rcout<<"11111"<<std::endl;
    // check if region_idx starts from 0!
    // start block update
    for(arma::uword m=0; m < num_region; m++){
      prog.increment();
      // Rcout<<"iter="<<iter<<"; region="<<m<<std::endl;
      arma::uvec delta = find(abs(alpha)>lambda);
      arma::uvec idx = region_idx[m];
      arma::uvec delta_Q = find(abs(alpha(idx))>lambda);
      arma::mat Q = K[m];
      arma::uvec L_idx;
      if(m==0){
        L_idx = arma::linspace<arma::uvec>(0,L_cumsum(m)-1,L_all(m));
      }else{
        L_idx = arma::linspace<arma::uvec>(L_cumsum(m-1),L_cumsum(m)-1,L_all(m));
      }
      arma::uvec delta_in_block = intersect(idx,delta);

      arma::mat K_block_t = Q.t();
      // arma::mat M_star = K_block_t*(M.rows(idx) - ones(idx.n_elem,1)*zetam.t()*C_t) - theta_eta.rows(L_idx);
      arma::mat M_star = K_block_t*M.rows(idx) - theta_zetam.rows(L_idx) * C_t - theta_eta.rows(L_idx);
      arma::mat temp = M_star - K_block_t.cols(delta_Q)*(alpha.rows(delta_in_block)-sign(alpha.rows(delta_in_block))*lambda)*X.t();
      arma::mat temp_X = temp;
      temp_X.each_row()%= X.t();
      arma::vec  temp_sum = arma::sum(temp_X,1)*sigma_M2_inv;
      arma::vec  grad_f = -theta_alpha(L_idx)/D(L_idx)*sigma_alpha2_inv+
        K_block_t.cols(delta_Q)*Q.rows(delta_Q) *temp_sum; // L x 1, use smooth function in grad

      double step = step_all(m);
      arma::vec   theta_alpha_diff = step*grad_f+sqrt(2*step)*arma::randn(size(grad_f));
      arma::vec  theta_alpha_new_block = theta_alpha(L_idx) + theta_alpha_diff;


      // MH step
      double log_target_density = -0.5*square(norm(theta_alpha(L_idx)/sqrt(D(L_idx)),2))*sigma_alpha2_inv-
        0.5*square(arma::norm(temp,"fro"))*sigma_M2_inv;


      alpha_new = alpha;// change this
      alpha_new(idx) += Q*theta_alpha_diff;
      // arma::uvec delta_new = find(abs(alpha_new)>lambda);// find for one region
      arma::uvec delta_Q_new = find(abs(alpha_new(idx))>lambda);
      arma::uvec delta_in_block_new = idx(delta_Q_new);
      // arma::uvec delta_in_block_new = intersect(idx,delta_new);
      arma::vec   alpha_new_region = alpha_new.rows(delta_in_block_new);
      arma::mat temp_new = M_star - K_block_t.cols(delta_Q_new)*(alpha_new_region-sign(alpha_new_region)*lambda)*X.t();

      arma::mat temp_X_new = temp_new;
      temp_X_new.each_row()%= X.t();
      arma::vec  temp_sum_new = arma::sum(temp_X_new,1)*sigma_M2_inv;
      arma::vec  grad_f_new = -theta_alpha_new_block/D(L_idx)*sigma_alpha2_inv+
        K_block_t.cols(delta_Q_new)*Q.rows(delta_Q_new) *temp_sum_new; // L x 1, use smooth function in grad
      // Rcout<<"test 6-5"<<std::endl;
      double log_target_density_new = -0.5*square(norm( theta_alpha_new_block/sqrt(D(L_idx)),2))*sigma_alpha2_inv-
        0.5*square(arma::norm(temp_new,"fro"))*sigma_M2_inv;
      //         // Rcout<<"test 7"<<std::endl;
      double log_q = -1/4/step * square(norm(-theta_alpha_diff-step*grad_f_new,2));
      double log_q_new = -1/4/step * square(norm(theta_alpha_diff-step*grad_f,2));
      double rho = log_target_density_new + log_q - log_target_density - log_q_new;

      // Rcout<<"rho="<<rho<<std::endl;
      // Rcout<<"test 8"<<std::endl;
      if(log(arma::randu())<=rho){
        theta_alpha(L_idx) = theta_alpha_new_block;
        alpha = alpha_new;
        accept(all_iter) = 1;
        accept_block(iter,m) = 1;
        temp = temp_new;
      }





    }// true end of block update

    // Rcout<<"11222"<<std::endl;
    // Rcout<<"1"<<std::endl;
    if( (iter%interval==0) & (iter>0)  ){
      arma::uvec u = arma::linspace<arma::uvec>(iter-interval,iter-1,interval);
      emp_accept.row(iter/interval-1) = mean(accept_block.rows(u),0);
      // Rcout<<"1122-1"<<std::endl;
      if(iter<stop_burnin){
        arma::vec  sigma_t = sqrt(2*step_all);
        for(arma::uword l = 0; l<num_block; l++){
          sigma_t(l) = adjust_acceptance(emp_accept(iter/interval-1,l),sigma_t(l),
                  target_accept_vec(l));
          step_all(l) = sigma_t(l)*sigma_t(l)/2;
          if(step_all(l)>1){step_all(l)=1;}
        }

      }
    }
    //         // when stop burnin, choose the average of last few steps
    arma::uword back =  (n_mcmc - stop_burnin)/(n_mcmc/10) ;
    if(iter==stop_burnin & stop_burnin > back){

      arma::uvec u = arma::linspace<arma::uvec>(iter-back,iter-1,back);
      step_all = exp(mean(log(track_step.cols(u)),1));
    }

    if( (iter%interval==0) & (iter>0)  ){
      arma::uvec u = arma::linspace<arma::uvec>(iter-interval,iter-1,interval);
      emp_accept.row(iter/interval-1) = mean(accept_block.rows(u),0);
    }

    //     // Rcout<<"test 12"<<std::endl;
    track_step.col(iter) = step_all;
    all_iter = all_iter+1;

    //     // -------------- Update lambda using RW MCMC --------------
    // double lambda_new = arma::randn()*sigma_lambda + lambda;
    // if(lambda_new>0 && lambda_new<1){
    //   double logll_M_new = 0;
    //   for(arma::uword m=0; m<num_region; m++){
    //     arma::uvec delta = find(abs(alpha)>lambda_new);
    //     arma::uvec idx = region_idx[m];
    //     arma::uvec delta_Q = find(abs(alpha(idx))>lambda_new);
    //     arma::mat Q = K[m];
    //     arma::uvec L_idx;
    //     if(m==0){
    //       L_idx = arma::linspace<arma::uvec>(0,L_cumsum(m)-1,L_all(m));
    //     }else{
    //       L_idx = arma::linspace<arma::uvec>(L_cumsum(m-1),L_cumsum(m)-1,L_all(m));
    //     }
    //     arma::uvec delta_in_block = intersect(idx,delta);
    //
    //     arma::mat K_block_t = Q.t();
    //     arma::mat M_star = K_block_t*(M.rows(idx) - ones(idx.n_elem,1)*zetam.t()*C_t) - theta_eta.rows(L_idx);
    //     arma::mat temp = M_star - K_block_t.cols(delta_Q)*(alpha.rows(delta_in_block)-sign(alpha.rows(delta_in_block))*lambda_new)*X.t();
    //     logll_M_new += -0.5*square(arma::norm(temp,"fro"))*sigma_M2_inv;
    //   }
    //   double rho = logll_M_new - logll_M;
    //   if(log(arma::randu()) <= rho){
    //     lambda = lambda_new;
    //     accept_lambda(iter) = 1;
    //   }
    // }
    // lambda_mcmc(iter) = lambda;
    // if( (iter%interval==0) & (iter>0) &(iter<stop_burnin) ){
    //   arma::uvec u = arma::linspace<arma::uvec>(iter-interval,iter-1,interval);
    //   double rate = arma::mean(accept.elem(u));
    //   sigma_lambda = adjust_acceptance(rate,sigma_lambda,0.4);
    // }
    // Rcout<<"22222"<<std::endl;
    // Rcout<<"iter="<<iter<<std::endl;

    //     // -------------- Update all other parameters using GS --------------
    arma::mat Mstar_alpha_term = arma::zeros(size(theta_eta));
    for(arma::uword m=0; m<num_region; m++){
      arma::uvec delta = find(abs(alpha)>lambda);
      arma::uvec idx = region_idx[m];
      arma::uvec delta_Q = find(abs(alpha(idx))>lambda);
      arma::uvec delta_in_block = intersect(idx,delta);
      arma::mat Q = K[m];
      arma::uvec L_idx;
      if(m==0){
        L_idx = arma::linspace<arma::uvec>(0,L_cumsum(m)-1,L_all(m));
      }else{
        L_idx = arma::linspace<arma::uvec>(L_cumsum(m-1),L_cumsum(m)-1,L_all(m));
      }
      arma::mat K_block_t = Q.t();
      Mstar_alpha_term.rows(L_idx) = K_block_t.cols(delta_Q)*(M.rows(delta_in_block)-( alpha.rows(delta_in_block)-sign(alpha.rows(delta_in_block))*lambda)*X.t() );
    }
    // // 2. zetam
    arma::mat zetam_res = Mstar_alpha_term - theta_eta;// L by n
    arma::mat mean_zetam = arma::mat(L_max,q);

    for(int j =0; j<q; j++){
      arma::vec  Sigma_zetam_j = 1/(1/D*sigma_zetam2_inv + sigma_M2_inv*X2_sum_allsample_q(j));
      arma::uvec c_j = complement(j, j, q);
      // change the following line when theta_eta needs to be updated
      arma::vec  mean_zetam_j = zetam_res*C.col(j) - theta_zetam.cols(c_j) * XcXq_sumsq.col(j);
      mean_zetam_j %= Sigma_zetam_j*sigma_M2_inv;
      mean_zetam.col(j) = mean_zetam_j;
      theta_zetam.col(j) = arma::randn(L_max,1)%sqrt(Sigma_zetam_j) +  mean_zetam_j;
    }
    // arma::mat Sigma_zetam_inv = sigma_zetam2_inv + sigma_M2_inv*q2_sum*C_t*C;
    // arma::mat Sigma_zetam = inv_sympd(Sigma_zetam_inv );
    // arma::mat zeta_res = Mstar_alpha_term - theta_eta;
    // arma::rowvec temp_zeta = q_vec.t()*zeta_res;
    // arma::rowvec mu_zetam = temp_zeta*C *Sigma_zetam* sigma_M2_inv;
    // zetam = arma::mvnrnd( mu_zetam.t(), Sigma_zetam);


    // Rcout<<"1"<<std::endl;
    // 3. theta_eta
    if(iter%interval_eta ==0){
      theta_eta_res =  Mstar_alpha_term - theta_zetam*C_t;
      arma::vec   eta_Sigma_vec = 1/(sigma_M2_inv + sigma_eta2_inv/D);
      theta_eta_res.each_col() %= eta_Sigma_vec*sigma_M2_inv;

      theta_eta = hyperplane_MVN_multiple(G,H_mat,eta_Sigma_vec,theta_eta_res.t());

      // arma::mat theta_eta_new = arma::randn(L_max,n);
      // theta_eta_new.each_col() %= arma::sqrt(eta_Sigma_vec);
      // theta_eta_new += theta_eta_res;
      // theta_eta.rows(eta_cutoff_range) = theta_eta_new.rows(eta_cutoff_range);
    }

    // Rcout<<"2"<<std::endl;
    // 4. sigma_alpha
    sigma_alpha2_inv = arma::randg( arma::distr_param(a + L_max*0.5, 1/(b + dot(theta_alpha,theta_alpha/D)/2)) );

    // 5. sigma_M
    M_reg_res = Mstar_alpha_term - theta_zetam*C_t - theta_eta;
    double M_norm = norm(M_reg_res,"fro");
    sigma_M2_inv = arma::randg( arma::distr_param(a + n*L_max/2,1/(b + M_norm*M_norm/2) ) );


    //   6. sigma_zetam -> half-cauchy
    // sigma_zetam2_inv = arma::randg( arma::distr_param(a + zetam.n_elem/2,1/(b + dot(zetam,zetam)/2) ) );
    // sigma_zetam2_inv_mcmc(iter) = sigma_zetam2_inv;
    // Rcout<<"3"<<std::endl;
    //   7. sigma_eta -> giving too large variance
    if(iter%interval_eta ==0){
      theta_eta_temp = theta_eta;
      theta_eta_temp.each_col() %= 1/sqrt(D);
      double eta_norm = norm(theta_eta_temp,"fro");
      sigma_eta2_inv = arma::randg( arma::distr_param(a + 0.5*n*L_max,
                                                      1/(b + eta_norm*eta_norm/2)) );

    }

    //
    // Rcout<<"4"<<std::endl;
    //     // --------------------- arma::summarize return --------------------- //
    logll_M = -0.5*square(arma::norm(M_reg_res,"fro"))*sigma_M2_inv;
    if(iter%thinning == 0){
      int iter_mcmc = iter/thinning;
      theta_alpha_mcmc.col(iter_mcmc) = theta_alpha;
      logll_mcmc(iter_mcmc) = logll_M + 0.5*n*L_max*log(sigma_M2_inv);
      theta_zetam_mcmc.slice(iter_mcmc) = theta_zetam;
      sigma_alpha2_inv_mcmc(iter_mcmc) = sigma_alpha2_inv;
      sigma_M2_inv_mcmc(iter_mcmc) = sigma_M2_inv;
      sigma_eta2_inv_mcmc(iter_mcmc) = sigma_eta2_inv;
    }

  }



  List gs = Rcpp::List::create(Rcpp::Named("theta_eta") = theta_eta,
                               Rcpp::Named("theta_zetam_mcmc")= theta_zetam_mcmc,
                               Rcpp::Named("sigma_M2_inv_mcmc")= sigma_M2_inv_mcmc ,
                               Rcpp::Named("sigma_alpha2_inv_mcmc")= sigma_alpha2_inv_mcmc,
                               Rcpp::Named("sigma_eta2_inv_mcmc")=  sigma_eta2_inv_mcmc);
  timer.step("end of iterations");
  return Rcpp::List::create(Rcpp::Named("theta_alpha_mcmc")=theta_alpha_mcmc,
                            Rcpp::Named("logll_mcmc")=logll_mcmc,
                            Rcpp::Named("track_step")=track_step,
                            Rcpp::Named("accept_blcok")=accept_block,
                            Rcpp::Named("emp_accept")=emp_accept,
                            Rcpp::Named("gs")=gs,
                            Rcpp::Named("Timer")=timer
  );

}

