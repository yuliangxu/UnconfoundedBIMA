library(ggplot2)
library(viridis)
library(BayesGPfit)
library(RSpectra)

matern_kernel = function(x,y,nu,l=1){
  d = sqrt(sum((x-y)^2))/l
  y = 2^(1-nu)/gamma(nu)*(sqrt(2*nu)*d)^nu*besselK(sqrt(2*nu)*d,nu)
  return(y)
}
#‘ Generate a matern basis
#' @importFrom RSpectra eigs_sym
generate_matern_basis2 = function(grids, region_idx_list, L_vec,scale = 2,nu = 1/5,
                                  show_progress = FALSE){
  if(nu=="vec"){
    nu_vec = region_idx_list["nu_vec"]
  }
  num_block = length(region_idx_list)
  Phi_D = vector("list",num_block)
  Phi_Q = vector("list",num_block)
  Lt = NULL; pt = NULL
  for(i in 1:num_block){
    if(show_progress){
      print(paste("Computing basis for block ",i))
    }
    p_i = length(region_idx_list[[i]])
    kernel_mat = matrix(NA,nrow = p_i, ncol=p_i)
    for(l in 1:p_i){
      if(nu=="vec"){
        kernel_mat[l,] = apply(grids[region_idx_list[[i]],],1,matern_kernel,y=grids[region_idx_list[[i]],][l,],nu = nu_vec[i],l=scale)
      }else{
        kernel_mat[l,] = apply(grids[region_idx_list[[i]],],1,matern_kernel,y=grids[region_idx_list[[i]],][l,],nu = nu,l=scale)
      }
    }
    diag(kernel_mat) = 1
    K = eigs_sym(kernel_mat,L_vec[i])
    K_QR = qr(K$vectors)
    Phi_Q[[i]] = qr.Q(K_QR )
    Phi_D[[i]] = K$values
    Lt = c(Lt, length(Phi_D[[i]]))
    pt = c(pt, dim(Phi_Q[[i]])[1])
  }
  return(list(Phi_D = Phi_D,
              region_idx_block = region_idx_list,
              Phi_Q = Phi_Q,L_all = Lt,p_length=pt))
}

#' plot_img
#' 
#' @param img A vector of input image
#' @param grids_df A data frame to indicate the position of pixels
#' @import ggplot2
#' @import viridis
plot_img = function(img, grids_df,title="img",col_bar = NULL){
  ggplot(grids_df, aes(x=x1,y=x2)) +
    geom_tile(aes(fill = img)) +
    scale_fill_viridis_c(limits = col_bar, oob = scales::squish)+
    
    ggtitle(title)+
    theme(plot.title = element_text(size=20),legend.text=element_text(size=10))
}
Soft_threshold = function(x,lambda){
  return( (x-sign(x)*lambda)*(abs(x)>lambda))
}

Unconfounded_mediation_data_constrained_eta = function(beta, alpha,n,Q,D,
                                                       lambda,q=2,
                                           sigma_M = 0.1, 
                                           sigma_Y = 0.1, 
                                           gamma = 2){
  
  p = length(beta)
  X = rnorm(n)
  C = matrix(rnorm(n*q),q,n)*0.1 # q by n
  sigma_eta = 0.1
  
  # get constrained eta
  G = rbind(X,C)
  H_mat = get_H_mat(G); # from Unconfounded_BIMA.cpp
  theta_eta = hyperplane_MVN_multiple(G, H_mat, sigma_eta^2*D, matrix(0,n,L) ); # output L by n
  eta = Q %*% theta_eta
  
  zeta_y = sample(-10:10,q)
  zeta_m = matrix(rnorm(p*q),p,q)
  nu = rep(0,p)
  nu[sample(1:p, round(p*0.2))] = 0.5
  nu[beta > quantile(beta,0.9)] = 1
  
  alpha_STGP = Q%*%t(Q)%*%alpha; alpha_STGP = Soft_threshold(alpha_STGP[,1], lambda)
  beta_STGP = Q%*%t(Q)%*%beta; beta_STGP = Soft_threshold(beta_STGP[,1], lambda)
  nu_STGP = Q%*%t(Q)%*%nu; nu_STGP = Soft_threshold(nu_STGP[,1], lambda)
  
  # generate image
  M = alpha_STGP %*% t(X) + zeta_m %*% C + eta + matrix(rnorm(p*n, sd = sigma_M),p,n)
  
  # generate outcome
  Y = t(beta_STGP) %*% M + t(c(gamma, zeta_y)) %*% rbind(t(X),C) + t(nu_STGP) %*% eta  + rnorm(n, sd = sigma_Y)
  
  # true_params
  true_params = list(beta = beta_STGP, alpha = alpha_STGP, zeta_y = zeta_y, zeta_m = zeta_m,
                     gamma = gamma, sigma_M = sigma_M, sigma_Y = sigma_Y,nu = nu_STGP)
  
  return(list(Y = Y, M = t(M), X = X, C = C, eta = eta, theta_eta = theta_eta, nu = nu, 
              true_params = true_params))
}


Unconfounded_mediation_data_eta = function(beta, alpha,n,q=2,sigma_M = 0.1, 
                                           Q,D,
                                           Q_eta,D_eta,
                                       sigma_Y = 0.1, rho = 0.5, gamma = 2){
  
  p = length(beta)
  X = rnorm(n)
  C = matrix(rnorm(n*q),q,n)*0.1 # q by n
  L_eta = dim(Q_eta)[2]
  eta = Q_eta%*%matrix(rnorm(L_eta*n)*rep(sqrt(D_eta),n),nrow = L_eta, ncol = n)
  
  zeta_y = sample(-10:10,q)
  zeta_m = matrix(rnorm(p*q),p,q)
  nu = rep(0,p)
  nu[sample(1:p, round(p*0.2))] = 0.5
  nu[beta > quantile(beta,0.9)] = 1
  
  alpha_STGP = Q%*%t(Q)%*%alpha
  beta_STGP = Q%*%t(Q)%*%beta
  nu_STGP = Q%*%t(Q)%*%nu
  
  # generate image
  M = alpha_STGP %*% t(X) + zeta_m %*% C + eta + matrix(rnorm(p*n, sd = sigma_M),p,n)
  
  # generate outcome
  Y = t(beta_STGP) %*% M + t(c(gamma, zeta_y)) %*% rbind(t(X),C) + t(nu_STGP) %*% eta  + rnorm(n, sd = sigma_Y)
  
  # true_params
  true_params = list(beta = beta_STGP[,1], alpha = alpha_STGP[,1], zeta_y = zeta_y, zeta_m = zeta_m,
                     gamma = gamma, sigma_M = sigma_M, sigma_Y = sigma_Y,nu = nu_STGP[,1])
  
  return(list(Y = Y, M = t(M), X = X, C = C, eta = eta, nu = nu, 
              true_params = true_params))
}

scalar_Unconfounded_mediation_data = function(beta_img, alpha_img,n,q=2,
                                              sigma_M = 0.1, sigma_Y = 0.1, 
                                              gamma = 2, nu = -2){
  
  p = length(beta_img)
  
  # generate exposure
  X = rnorm(n)
  C = matrix(rnorm(n*q),q,n)*0.1 # q by n
  U = rnorm(n)
  
  zeta_y = sample(-10:10,q)
  zeta_m = matrix(rnorm(p*q),p,q)
  
  
  
  # generate image
  M = alpha_img %*% t(X) + zeta_m %*% C + t(replicate(p,U)) + 
      matrix(rnorm(p*n, sd = sigma_M),p,n)
  
  # generate outcome
  Y = t(beta_img) %*% M + t(c(gamma, zeta_y)) %*% rbind(t(X),C) + 
      nu * U  + rnorm(n, sd = sigma_Y)
  
  # true_params
  true_params = list(beta = beta_img, alpha = alpha_img, 
    zeta_y = zeta_y, zeta_m = zeta_m, nu = nu,
                     gamma = gamma, sigma_M = sigma_M, sigma_Y = sigma_Y)
  
  return(list(Y = Y, M = t(M), X = X, C = C, U = U,  true_params = true_params))
}

convert_to_long = function(datsim, M_res = NULL){

  n = length(datsim$Y)
  if(is.null(M_res)){
    wide_df = as.data.frame(cbind(id = 1:n,Y = datsim$Y[1,], M = datsim$M,
                                X = datsim$X, C = t(datsim$C)))
  }else{
    wide_df = as.data.frame(cbind(id = 1:n,Y = datsim$Y[1,], M_res = M_res,
                                X = datsim$X, C = t(datsim$C)))
  }
  data_long <- tidyr::gather(wide_df, "loc", "intensity", V3:V402, factor_key=TRUE)
  return(list(datsim_long = data_long, 
    datsim_wide = wide_df))

}

Unconfounded_mediation_data = function(beta_img, alpha_img,n,q=2,q_U = 3,sigma_M = 0.1,
                                       sigma_Y = 0.1, rho = 0.5, gamma = 2){
  
  p = length(beta_img)
  # X = rnorm(n)
  # C = matrix(rnorm(n*q),q,n)*0.1 # q by n
  # U = matrix(runif(n*q_U),q_U,n)*0.1 # q by n
  
  # generate exposure
  Sigma = matrix(rep(rho,(q+q_U+1)^2),q+q_U+1,q+q_U+1)
  diag(Sigma) =  rep(1,q+q_U+1)
  Exposure = mvtnorm::rmvnorm(n, sigma = Sigma)
  X = Exposure[,1]
  C = t(Exposure[,1:q+1])
  U = t(Exposure[,1:q_U+1+q])
  
  zeta_y = sample(-10:10,q)
  zeta_m = matrix(rnorm(p*q),p,q)
  zeta_mU = matrix(rnorm(p*q_U),p,q_U)*5
  zeta_yU = sample(-10:10,q_U)*5
  
  
  # generate image
  M = alpha_img %*% t(X) + zeta_m %*% C + zeta_mU %*% U + matrix(rnorm(p*n, sd = sigma_M),p,n)
  
  # generate outcome
  Y = t(beta_img) %*% M + t(c(gamma, zeta_y)) %*% rbind(t(X),C) + t(zeta_yU) %*% U  + rnorm(n, sd = sigma_Y)
  
  # true_params
  true_params = list(beta = beta_img, alpha = alpha_img, zeta_y = zeta_y, zeta_m = zeta_m,
                     gamma = gamma, sigma_M = sigma_M, sigma_Y = sigma_Y)
  
  return(list(Y = Y, M = t(M), X = X, C = C, U = U, zeta_mU = zeta_mU,zeta_yU = zeta_yU, true_params = true_params))
}

#' simulate_round_image
#' 
#' @import BayesGPfit
simulate_round_image = function(center_shift = c(0,0),lambda = 0.1,side = 30, 
                                range = c(0,1)){
  n_sqrt = side
  n = n_sqrt*n_sqrt
  grids = GP.generate.grids(d=2L,num_grids=n_sqrt)
  center = apply(grids,2,mean) + center_shift
  rad = apply(grids,1,function(x){sum((x-center)^2)})
  inv_rad = 2-rad
  inv_rad_ST = Soft_threshold(inv_rad,1.2)
  f_mu = Soft_threshold(log(inv_rad_ST^2+1),lambda)
  
  y = f_mu
  nonzero = y[abs(y)>0]
  a = range[1]; b = range[2]
  nonzero_mapped = (nonzero-min(nonzero))/(max(nonzero)-min(nonzero))*(b-a) + a
  y[abs(y)>0] = nonzero_mapped
  
  grids_df = as.data.frame(grids)
  
  
  return(list(img = y, grids_df = grids_df))
}


# previous help functions -------------------------------------------------

STGP_mcmc = function(theta_mcmc_sample,region_idx,basis,lambda){
  M = dim(theta_mcmc_sample)[2]
  S = max(unlist(region_idx))
  num_region = length(region_idx)
  est_mcmc = matrix(NA,nrow = S, ncol = M)
  dd = matrix(unlist(lapply(basis$Phi_Q,dim)),nrow=2)
  L_idx = cumsum(dd[2,])
  L_idx = c(rbind(L_idx,L_idx+1))
  L_idx = matrix(c(1,L_idx[-length(L_idx)]),nrow=2)
  for(l in 1:num_region){
    idx = region_idx[[l]]
    theta = theta_mcmc_sample[L_idx[1,l]:L_idx[2,l],]
    beta = basis$Phi_Q[[l]]%*%theta
    est_mcmc[idx,] = (beta-sign(beta)*lambda)*I(abs(beta)>lambda)
  }
  return(est_mcmc)
}

InclusionMap = function(mcmc_sample, true_beta, thresh = "auto", fdr_target = 0.1,
                        max.iter = 100){
  InclusionProb = 1 - apply(mcmc_sample, 1, function(x){mean(abs(x)==0)})
  true_beta = 1*(true_beta!=0)
  thresh_final = thresh
  fdr=NA
  if(thresh=="auto"){
    thresh = 0.5
    for(i in 1:max.iter){
      mapping = 1*(InclusionProb>thresh)
      fdr = FDR(mapping, true_beta)
      print(paste("fdr=",fdr,"thresh=",thresh))
      if(is.na(fdr)){
        print("fdr=NA, target FDR is too small")
        thresh = thresh/1.1
        mapping = 1*(InclusionProb>thresh)
        fdr = FDR(mapping, true_beta)
        print(paste("Use current fdr=",fdr,"thresh=",thresh))
        break
      }
      if(fdr<=fdr_target){
        thresh_final = thresh
        break
      }
      thresh = thresh*1.1
      if(thresh>1){
        print("New thresh>1, keep thresh at the current value and return result.")
        break
      }
    }
  }else{
    mapping = 1*(InclusionProb>thresh)
  }
  return(list(mapping = mapping, thresh=thresh_final,
              InclusionProb=InclusionProb))
}
FDR = function(active_region, true_region){
  sum(active_region!=0 & true_region==0)/sum(active_region!=0)
}
Precision = function(active_region, true_region){
  mean(I(active_region!=0) == I(true_region!=0))
}
Power = function(active_region, true_region){
  sum(active_region !=0 & true_region!=0)/sum(true_region!=0)
}
