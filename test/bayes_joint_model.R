setwd("/Volumes/GoogleDrive/My Drive/Research/Unconfounded_BIMA/code/UnconfoundedBIMA")
Rcpp::sourceCpp("src/Unconfounded_BIMA.cpp")
library(gtable)
library(grid)
in_sigma_M = 2
in_lambda = 0.2
# set up ------------------------------------------------------------------

source("R/help_functions.R")
# library(glmnet)
# side = 20
# p = side*side
n = 300
q = 2
# q_U = 2
# alpha_elnet = 1 # 0 for ridge, 1 for lasso
# CS_rho = 0.5
# n_rep = 30
# # set.seed(1)
# # ===== data generation =====
num_region = 1
side = 20 # > 50*50
p = side*side
region_idx = vector("list",num_region)
grids = GP.generate.grids(d=2L,num_grids=side)


idx_matr = matrix(1:(side*side),ncol = side)
side_per_region = side/sqrt(num_region)

for(r in 1:num_region){
  idx = rep(NA,(side_per_region)^2)
  colr = r - floor(r/sqrt(num_region))*sqrt(num_region);if(colr==0){
    colr = sqrt(num_region)
  }
  rowr = ceiling(r/sqrt(num_region));
  col_range = (max(colr-1,0):colr)*side_per_region;col_range[1] = col_range[1]+1;
  row_range = (max(rowr-1,0):rowr)*side_per_region;row_range[1] = row_range[1]+1;
  region_idx[[r]] = c(idx_matr[row_range[1]:row_range[2],col_range[1]:col_range[2]])
  
}

beta_img = simulate_round_image(side = side, range = c(0,1))
t1 = plot_img(beta_img$img, beta_img$grids_df, col_bar = c(0,1),"true beta")
alpha_img = simulate_round_image(center_shift = c(0.3,-0.3), side = side, range = c(0,1))
t2 = plot_img(alpha_img$img, alpha_img$grids_df,"true alpha")
grids_df = beta_img$grids_df



# basis = generate_matern_basis2(beta_img$grids_df, list(1:p), 0.3*p, scale = 2,nu = 1/5)
# saveRDS(basis,"./sim_data/basis_1region.rds")
basis = readRDS("./sim_data/basis_1region.rds")
L = length(basis$Phi_D[[1]])

L_idx = vector("list",num_region)

for(r in 1:num_region){
  if(r == 1){prev_ct = 0}else{prev_ct = sum(unlist(lapply(basis$Phi_D,length))[1:(r-1)])}
  L_idx[[r]] = 1:length(basis$Phi_D[[r]]) + prev_ct
}
L_idx_cpp = lapply(L_idx, function(x){x-1})

# generate data
beta=beta_img$img
datsim = Unconfounded_mediation_data_constrained_eta(beta_img$img, alpha_img$img,n,q = q, 
                                                     lambda = in_lambda,
                                         Q = basis$Phi_Q[[1]], 
                                         D = basis$Phi_D[[1]],
                                         sigma_M = in_sigma_M,
                                         sigma_Y=0.01, gamma = 2)

datsim$lambda = in_lambda

region_idx_cpp = lapply(region_idx, function(x){x-1})
n = dim(datsim$M)[1]
p = dim(datsim$M)[2]
m = dim(datsim$C)[1]


# run joint model ---------------------------------------------------------
Kernel_params = list(Phi_Q = basis$Phi_Q, 
                     D_vec = unlist(basis$Phi_D),
                     region_idx = lapply(basis$region_idx_block, function(x){x-1}) )
init_paras = list(alpha = rep(1,p), xi = matrix(0,p,q), eta = matrix(0,p,n),
                  sigma_M = 0.1, sigma_alpha = 0.1, sigma_xi = 0.1, sigma_eta = 0.1,
                  beta = rep(1,p), gamma = 1, zeta = rep(0,q), nu = rep(1,p), 
                  sigma_Y = 0.1, sigma_beta = 0.1, sigma_gamma = 0.1, sigma_zeta=0.1,sigma_nu=0.1)
num_region = length(basis$region_idx_block)
controls = list(step = rep(0.001,num_region) ,
                target_acceptance_rate = rep(0.2, num_region))
Rcpp::sourceCpp("src/Unconfounded_BIMA.cpp")
joint_bayes = Unconfounded_BIMA(datsim$Y, t(datsim$M), datsim$X, datsim$C,
                  Kernel_params, init_paras,
                  lambda_alpha = 0.5, 
                  lambda_beta = 0.5,
                  lambda_nu = 0.5,
                  step = controls$step,
                  target_acceptance_rate = controls$target_acceptance_rate,
                  accept_interval=10,
                  mcmc_sample = 1500, 
                  burnin = 2000, 
                  thinning = 1,
                  verbose = 0,
                  theta_eta_interval = 10,
                  start_update_eta = 500,
                  start_save_eta = 500)

par(mfrow = c(2,3))
plot(joint_bayes$mcmc$loglik_Y)
plot(joint_bayes$mcmc$theta_beta[1,])
# plot(joint_bayes$mcmc$gamma)
plot(1/joint_bayes$mcmc$inv_sigmasq_Y)

plot(joint_bayes$mcmc$loglik_M)
plot(joint_bayes$mcmc$theta_alpha[1,-1])
plot(1/joint_bayes$mcmc$inv_sigmasq_M)
eta_mean = Low_to_high(joint_bayes$mcmc$theta_eta_mean,p, Kernel_params$Phi_Q,
                       Kernel_params$region_idx, L_idx_cpp)
plot(datsim$eta, eta_mean)
abline(0,1,col="red")
par(mfrow = c(1,1))

# plot(joint_bayes$mcmc$)

summary(abs(c(rbind(datsim$X, datsim$C) %*% t(eta_mean)))) # check constraint


# visualize output --------------------------------------------------------


beta_gp_mcmc = Low_to_high(joint_bayes$mcmc$theta_beta,p, Kernel_params$Phi_Q, Kernel_params$region_idx, L_idx_cpp)
beta_stgp_mcmc = apply(beta_gp_mcmc,1,Soft_threshold, lambda = 0.5)
beta_stgp_mean = apply(beta_stgp_mcmc,2,mean)
e1=plot_img(beta_stgp_mean, grids_df,col_bar = c(0,1),"beta_stgp_mean")

alpha_gp_mcmc = Low_to_high(joint_bayes$mcmc$theta_alpha,p, Kernel_params$Phi_Q, Kernel_params$region_idx, L_idx_cpp)
alpha_stgp_mcmc = apply(alpha_gp_mcmc,1,Soft_threshold, lambda = 0.5)
alpha_stgp_mean = apply(alpha_stgp_mcmc,2,mean)
e2=plot_img(alpha_stgp_mean, grids_df,col_bar = c(0,1),"alpha_stgp_mean")

out_plot <- rbind(cbind(ggplotGrob(e1), ggplotGrob(e2) ),
                  cbind(ggplotGrob(t1), ggplotGrob(t2)), size = "first")

grid.draw(out_plot)