setwd("/Volumes/GoogleDrive/My Drive/Research/Unconfounded_BIMA/code/UnconfoundedBIMA")
Rcpp::sourceCpp("src/Unconfounded_BIMA.cpp")
# Rcpp::sourceCpp("src/Mediation_functions.cpp")
library(BIMA)
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

alpha_img = simulate_round_image(center_shift = c(0.3,-0.3), side = side, range = c(0,1))
grids_df = beta_img$grids_df


# basis = generate_matern_basis2(beta_img$grids_df, list(1:p), 0.3*p, scale = 2,nu = 1/5)
# saveRDS(basis,"./basis_1region.rds")
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
                                                     sigma_M = in_sigma_M,sigma_Y=0.01, gamma = 2)

datsim$lambda = in_lambda
t1 = plot_img(datsim$true_params$beta, beta_img$grids_df, col_bar = c(0,1),"true beta")
t2 = plot_img(datsim$true_params$alpha, alpha_img$grids_df, col_bar = c(0,1),"true alpha")
t3 = plot_img(datsim$true_params$beta*datsim$true_params$alpha, alpha_img$grids_df, col_bar = c(0,1),"true TIE")
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
                                lambda_alpha = datsim$lambda, 
                                lambda_beta = datsim$lambda,
                                lambda_nu = datsim$lambda,
                                step = controls$step,
                                target_acceptance_rate = controls$target_acceptance_rate,
                                accept_interval=10,
                                mcmc_sample = 20000, 
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
eta_mean_joint = Low_to_high(joint_bayes$mcmc$theta_eta_mean,p, Kernel_params$Phi_Q,
                       Kernel_params$region_idx, L_idx_cpp)
# plot(datsim$eta, eta_mean, main = "joint_eta")
# abline(0,1,col="red")
par(mfrow = c(1,1))

# plot(joint_bayes$mcmc$)

# summary(abs(c(rbind(datsim$X, datsim$C) %*% t(eta_mean)))) # check constraint

# M-reg -------------------------------------------------------------------
datsim2 = datsim
datsim2$M = t(datsim$M)
datsim2$Mstar = datsim$M %*% basis$Phi_Q[[1]]
datsim2$lambda = datsim$lambda
# note that in GS version, Image on scalar regression, zeta_m is implemented as a scalar instead of spatially-varying vector
gs64m = M_regression_GS(datsim2,
                        init = list(zetam=rep(1,m), sigma_alpha = 0.1,sigma_M=0.1,
                                    theta_eta = matrix(0,nrow = L, ncol=n),sigma_eta= 1,
                                    theta_alpha= rep(0.5,L),
                                    cb=1,a=1,b=2,sigma_zetam=0.1),
                        region_idx_cpp,
                        kernel = list(Q = basis$Phi_Q, D = basis$Phi_D),
                        n_mcmc = 400)
print("GS for M ....")
#> [1] "GS for M ...."
gs_burn = 370:400
init_m = list(
  theta_alpha =  apply(gs64m$theta_alpha_mcmc[,gs_burn],1,mean),
  D=unlist(basis$Phi_D), # recompute D
  theta_eta = gs64m$theta_eta,
  sigma_M = mean(1/sqrt(gs64m$sigma_M2_inv_mcmc[gs_burn])),
  sigma_alpha = mean(1/sqrt(gs64m$sigma_alpha2_inv_mcmc[gs_burn])),
  sigma_eta = mean(1/sqrt(gs64m$sigma_eta2_inv_mcmc[gs_burn])),
  theta_zetam = matrix(rep(0.1,L*m),L,m))
# n_mcmc = 1*1e5
n_mcmc = 2e4
num_block = 1
controls_m = list(lambda = datsim2$lambda,
                  n_mcmc = n_mcmc,
                  stop_adjust = 0.8*n_mcmc,
                  start_joint = 0,
                  interval_step = 10,
                  interval_eta = 10,
                  thinning = 1,
                  target_accept_vec = rep(0.2,num_region),
                  step = 1e-2/n)
print("MALA for M ....")

sim64m = M_regression_region_block(datsim2$M,
                                   datsim2$X, t(datsim2$C), basis$L_all,
                                   num_region = length(region_idx_cpp) ,
                                   region_idx = region_idx_cpp,
                                   n_mcmc = controls_m$n_mcmc ,
                                   basis$Phi_Q,
                                   stop_burnin = controls_m$stop_adjust,
                                   lambda = controls_m$lambda,
                                   target_accept_vec = controls_m$target_accept_vec,
                                   init = init_m,
                                   interval = controls_m$interval_step, # adjust step every 10 iter
                                   interval_eta = controls_m$interval_eta,
                                   thinning = controls_m$thinning,
                                   step = controls_m$step)
eta_bima = basis$Phi_Q[[1]] %*% sim64m$gs$theta_eta

par(mfrow = c(2,1))
plot(sim64m$logll_mcmc[(2.5*1e3):2e4])
plot(sim64m$theta_alpha_mcmc[1,-2e4])
par(mfrow = c(1,1))
# saveRDS(sim64m, "./sim64m_1region.rds")
# sim64m = readRDS("./sim64m_1region.rds")
# Y-reg -------------------------------------------------------------------

init_y = list(theta_beta = rep(1,L),
              theta_nu = rep(1,L),
              a_sigma_beta = 1, b_sigma_beta = 1,
              D=unlist(basis$Phi_D), # recompute D
              sigma_Y = 1,
              sigma_beta = 1,
              cb = 0 ,
              zetay = rep(1,dim(datsim$C)[1]),
              gamma = 1)# L by n
n_mcmc = 2e4
lambda = datsim2$lambda
controls_y = list(n_mcmc = n_mcmc, 
                  stop_burnin = 0.8*n_mcmc,
                  start_joint = 0.1*n_mcmc, 
                  lambda = datsim2$lambda,
                  interval_step = 10,
                  interval_thin = 1,
                  stop_adjust = 0.8*n_mcmc,
                  target_accept_vec = rep(0.2,num_region),
                  step = 1e-2/n)
region_idx_cpp = lapply(region_idx, function(x){x-1})

num_block = 1
# Rcpp::sourceCpp("src/Mediation_functions.cpp")
sim64y = BIMA::Y_regression_region_block_fast(Y = datsim2$Y, M = datsim2$M,
                                        X = datsim2$X, C = t(datsim2$C), 
                                        L_all = basis$L_all,
                                        num_region = num_block,
                                        region_idx = region_idx_cpp,n_mcmc = n_mcmc,
                                        K = basis$Phi_Q, 
                                        stop_burnin = controls_y$stop_burnin,
                                        start_joint = controls_y$start_joint, 
                                        lambda = controls_y$lambda, 
                                        target_accept_vec = rep(0.2,num_block),
                                        a=1,b=1,
                                        init = init_y,
                                        step = controls_y$step,
                                        interval_step = controls_y$interval_step, 
                                        interval_thin = controls_y$interval_thin)

par(mfrow = c(2,1))
plot(sim64y$logll_mcmc_Y[(2.5*1e3):2e4])
plot(sim64y$theta_beta_mcmc_thin[1,-2e4])
par(mfrow = c(1,1))

# # BIMA --------------------------------------------------------------------
# devtools::install_github("yuliangxu/BIMA")
# BIMA_mcmc = BIMA::BIMA(datsim2$Y, datsim2$X, datsim2$M, datsim2$C,
#                        init_y = init_y, init_m = init_m, 
#                        controls_m = controls_m, controls_y = controls_y,
#                        kernel_setting = list(method = "Self-defined",
#                                              Phi_Q = basis$Phi_Q, Phi_D = basis$Phi_D))
# BIMA_mcmc$TIE_sample
# e1_bima=plot_img(apply(BIMA_mcmc$beta_sample,1,mean),as.data.frame(grids_df),"est BIMA beta_mean")
# e2_bima=plot_img(apply(BIMA_mcmc$alpha_sample,1,mean),as.data.frame(grids_df),"est BIMA alpha_mean")
# 
# plot_img(apply(BIMA_mcmc$TIE_sample,1,mean),as.data.frame(grids_df),
#          "est BIMA TIE_mean")
# datsim$total_test_ST = datsim$true_params$alpha  * datsim$true_params$beta
# inclusion_map_tuned = InclusionMap(BIMA_mcmc$TIE_sample,datsim$total_test_ST,fdr_target=0.1)
# inclusion_map = InclusionMap(BIMA_mcmc$TIE_sample,datsim$total_test_ST,thresh=0.5)
# TIE = rep(0,p)
# S_idx = which(inclusion_map_tuned$mapping==1)
# S_null = which(datsim$total_test_ST==0)
# S_nonnull = which(datsim$total_test_ST!=0)
# TIE[S_idx] = apply(BIMA_mcmc$TIE_sample[S_idx,],1,function(a){mean(a[(abs(a))>0])})
# 
# e3=plot_img(TIE, grids_df,"estimated TIE")
# plot_img(datsim$total_test_ST, grids_df,"true TIE")
# 
# sum_TIE = matrix(NA, nrow=1,ncol=5)
# colnames(sum_TIE) = c("FDR","Power","Precision","MSE_null","MSE_nonnull")
# sum_TIE[1,1] = FDR(TIE,datsim$total_test_ST)
# sum_TIE[1,2] = Power(TIE,datsim$total_test_ST)
# sum_TIE[1,3] = Precision(TIE,datsim$total_test_ST)
# sum_TIE[1,4] = mean((TIE[S_null] - datsim$total_test_ST[S_null])^2)
# sum_TIE[1,5] = mean((TIE[S_nonnull] - datsim$total_test_ST[S_nonnull])^2)
# sim_result = NULL
# sim_result$sum_TIE = sum_TIE
# knitr::kable(sim_result)
# # summary -----------------------------------------------------------------

n_mcmc = dim(sim64y$theta_beta_mcmc_thin)[2]-1
theta_sample = sim64y$theta_beta_mcmc_thin[, ceiling(n_mcmc*0.8):n_mcmc]
beta_sample = STGP_mcmc(theta_sample,region_idx,basis,lambda = datsim2$lambda)
# theta_nu_sample = sim64y$theta_nu_mcmc_thin[, ceiling(n_mcmc*0.8):n_mcmc]
# nu_sample = STGP_mcmc(theta_nu_sample,region_idx,basis,lambda = datsim$lambda)


n_mcmc = dim(sim64m$theta_alpha_mcmc)[2]-1
theta_sample = sim64m$theta_alpha_mcmc[,ceiling(n_mcmc*0.9):n_mcmc]
alpha_sample = STGP_mcmc(theta_sample,region_idx,basis,lambda = datsim2$lambda)


beta_sample_thin = beta_sample[,seq(1,dim(beta_sample)[2],length.out = dim(alpha_sample)[2])]
total_sample = beta_sample_thin*alpha_sample
datsim$total_test_ST = datsim$true_params$alpha*datsim$true_params$beta

bima_mean = list(alpha = apply(alpha_sample,1,mean),
                 beta = apply(beta_sample_thin,1,mean),
                 gamma = mean(sim64y$gs$gamma_mcmc[ceiling(n_mcmc*0.8):n_mcmc]),
                 theta_eta = sim64m$gs$theta_eta)
bima_mean$TIE = bima_mean$alpha * bima_mean$beta

b1=plot_img(apply(beta_sample_thin,1,mean),
            as.data.frame(grids), col_bar = c(0,1),"BIMA: beta_mean")
b2=plot_img(apply(alpha_sample,1,mean),
            as.data.frame(grids), col_bar = c(0,1),"BIMA: alpha_mean")

b3=plot_img(apply(beta_sample_thin,1,mean)*apply(alpha_sample,1,mean),
            as.data.frame(grids), col_bar = c(0,1),"BIMA: TIE_mean")

# est_plot <- cbind(ggplotGrob(e1), ggplotGrob(e2),size = "first")
# bima_plot <- cbind(ggplotGrob(e1_bima), ggplotGrob(e2_bima),size = "first")
# true_plot <- cbind(ggplotGrob(t1), ggplotGrob(t2),size = "first")
# # g = rbind( est_plot,bima_plot, true_plot)
# grid.newpage()
# grid::grid.draw(g)
# 
# nu_true = plot_img(datsim$true_params$nu,
#                    as.data.frame(grids), "true nu")
# nu_est = plot_img(apply(nu_sample,1,mean),
#                   as.data.frame(grids), "est. Uncon nu_mean")
# g_nu = cbind(ggplotGrob(nu_true), ggplotGrob(nu_est),size = "first")
# grid::grid.draw(g_nu)


# visualize output --------------------------------------------------------


beta_gp_mcmc = Low_to_high(joint_bayes$mcmc$theta_beta,p, Kernel_params$Phi_Q, Kernel_params$region_idx, L_idx_cpp)
beta_stgp_mcmc = apply(beta_gp_mcmc,1,Soft_threshold, lambda = datsim$lambda)
beta_stgp_mean = apply(beta_stgp_mcmc,2,mean)
e1=plot_img(beta_stgp_mean, grids_df,col_bar = c(0,1),"joint:beta_mean")

alpha_gp_mcmc = Low_to_high(joint_bayes$mcmc$theta_alpha,p, Kernel_params$Phi_Q, Kernel_params$region_idx, L_idx_cpp)
alpha_stgp_mcmc = apply(alpha_gp_mcmc,1,Soft_threshold, lambda = datsim$lambda)
alpha_stgp_mean = apply(alpha_stgp_mcmc,2,mean)
e2=plot_img(alpha_stgp_mean, grids_df,col_bar = c(0,1),"joint:alpha_mean")

e3=plot_img(alpha_stgp_mean*beta_stgp_mean, grids_df,col_bar = c(0,1),"joint:TIE_mean")

joint_mcmc = length(joint_bayes$mcmc$gamma)
joint_mean = list(alpha = alpha_stgp_mcmc,
                  beta = beta_stgp_mean,
                  gamma = mean(joint_bayes$mcmc$gamma[ceiling(joint_mcmc*0.8):joint_mcmc]),
                  eta = joint_bayes$mcmc$theta_eta_mean)

out_plot <- rbind(cbind(ggplotGrob(e1), ggplotGrob(e2), ggplotGrob(e3) ),
                  cbind(ggplotGrob(b1), ggplotGrob(b2), ggplotGrob(b3) ),
                  cbind(ggplotGrob(t1), ggplotGrob(t2), ggplotGrob(t3) ), size = "first")



# par(mfrow = c(1,2))
# plot(c(datsim$eta), c(eta_bima),main="eta_bima"); abline(0,1,col="red")
# plot(c(datsim$eta), c(eta_mean), main = "joint_eta_mean"); abline(0,1,col="red")
# par(mfrow = c(1,1))

par(mfrow = c(2,1))
plot(c(datsim$theta_eta), c(joint_bayes$mcmc$theta_eta_mean),main="joint_theta_eta_mean"); abline(0,1,col="red")
plot(c(datsim$theta_eta), c(sim64m$gs$theta_eta), main = "theta_eta_bima"); abline(0,1,col="red")
par(mfrow = c(1,1))

grid.draw(out_plot)

# output bias, mse for alpha, beta, TIE, gamma for each method
get_mse = function(result_mean){
  # alpha
  alpha_diff = abs(result_mean$alpha - datsim$true_params$alpha)
  beta_diff = abs(result_mean$beta - datsim$true_params$beta)
  gamma_diff = abs(result_mean$gamma - datsim$true_params$gamma)
  TIE_diff = abs(result_mean$alpha*result_mean$beta - datsim$true_params$alpha*datsim$true_params$beta)
  true_TIE = datsim$true_params$alpha*datsim$true_params$beta
  
  c(alpha_null = mean(alpha_diff[datsim$true_params$alpha==0]),
    alpha_nonnull = mean(alpha_diff[datsim$true_params$alpha!=0]),
    beta_null = mean(beta_diff[datsim$true_params$beta==0]),
    beta_nonnull = mean(beta_diff[datsim$true_params$beta!=0]),
    TIE_null = mean(TIE_diff[true_TIE==0]),
    TIE_nonnull = mean(TIE_diff[true_TIE!=0]),
    gamma = mean(gamma_diff)) 
}
bias_out = rbind(bima_mean = get_mse(bima_mean),
                 joint_mean = get_mse(joint_mean))
knitr::kable(bias_out,digits = 4)

output = list(bima_mean = bima_mean,
              joint_mean = joint_mean)

saveRDS(output,paste("./sim_result/output_rep",rep,"_sigmaM",in_sigma_M,"_lambda",in_lambda,".rds") ) 


