setwd("/home/dan/genericObo")
source("helper.R")
source("generateData.R")


block_norm = function(y,y_hat,groups){
  K = length(unique(groups))
  groups = as.numeric(as.factor(groups)) ## in case we got strings for groups
  block_loss = matrix(0,ncol=K,nrow=1)
  for(k in groups){
    taskIdx = which(groups==k)
    block_loss[k] = sum((X[,j][taskIdx]-grad[taskIdx])^2) 
  }
  return(norm2(block_loss))
}


plotPath = function(res){ ## todo: add labels
  plot(res$betas_norm2[,1],type='line')
  points(res$betas_norm2[,2],type='line')
  points(res$betas_norm2[,3],type='line')
  points(res$betas_norm2[,4],type='line')
  points(res$betas_norm2[,5],type='line')
  points(res$betas_norm2[,6],type='line')
  points(res$betas_norm2[,7],type='line')
  points(res$betas_norm2[,8],type='line')
}

## for that we need to accumulate beta in every step
plotBlock = function(res,blockIndex){
  
}

## X - problems are concatenated row-wise
## y - responses are concatenated row-wise
## groups - the same length as y, with indexes 1->K where index i means the sample belongs to task i 
algo1 = function(X,y,groups,M=1000,eps=0.01){
X = as.matrix(X)
## standartize X:
X = scale(X)
groups = as.numeric(as.factor(groups)) ## in case we got strings for groups
groupnames = sort(unique(groups))
K = length(groupnames) ## number of tasks
p = ncol(X) ## number of covariates 
beta = matrix(0,nrow=K,ncol=p) ## row per task, column per coefficient across task
betas_norm2 = matrix(0,nrow=M,ncol=p) ### one column per group of covariates (subspace). one row per iteration
preds = eps*rep(0,nrow(y)) ## initial guess
for(m in 1:M){
  best_loss = matrix(0,ncol=K,nrow=1)
  j_star = -1 ## j_start will hold the chosen subspace to walk along (i.e block of covariates)
  u = matrix(0, ncol=K) ## we will advance in the chosen subspace by u/norm2(u)
  grad = -negative_gradient2(y=y,preds=preds,type="regression")  ### gradient after walking along the j-th subspace
  #print(norm2(neg_grad))
  for(j in 1:p){
    # if(j<6){
    #   print(paste0(j,"   ",cor(X[,j],grad)))  
    # }
    
    loss_grad_j = matrix(0,ncol=K,nrow=1) ### square loss = 0 <--> this subspace is parallel to the gradient
    for(k in 1:K){
      taskIdx = which(groups==k)
      #loss_grad_j[k] = sum((X[,j][taskIdx]-neg_grad[taskIdx])^2) ## square error vs gradient value. we choose the subspace whose mean square error across tasks is lowest, i.e it's the one which mostly apporoxmniates the pseudo-responses = mostly parallel
      loss_grad_j[k] = abs(cor(X[,j][taskIdx],grad[taskIdx])) ## finding who is mostly correlated with the negative gradient
      
      #loss_grad_j_signs[k] = s
    }
    
    ## we choose the j-th subspace if it's mostly reduceing the gradient: in coordinate descent language, we 
    ## choose the one which has the smallest square-loss when approximating the gradinet. adapting this to 
    ## MTL we take the j with the smallest norm2 square-loss across tasks
    if(norm2(loss_grad_j) >=  norm2(best_loss)){
      best_loss=loss_grad_j
      ## now we need to build a per-task step. in Obozinski we take a gradient step per task. either we need to approximate the 
      ## here, we know that hte gradient mostly parallel to j_star.
      for(k in 1:K){
        taskIdx = which(groups==k)
        #u[k] = sign(cor(X[,j][taskIdx],grad[taskIdx])) ## this works reasonably well
        u[k] = cor(X[,j][taskIdx],grad[taskIdx]) ## correlation ~~ gradient? 
        
        ### we need here that task k will advance in  ~dL/dX_(j_star)
      }
      j_star = j
      
    }
  }
  
  
  u = u/norm2(u)
  
  if(m %% 10 ==0){
    print(m)
    print(beta)
    print("------")
    print(u)

  }
  
  beta[,j_star] = beta[,j_star] - eps*u ## since we took the sign of the correlation with the *gradient*, we go *against* that sign, as we want to go in teh direction *opposting* the gradient
  allpreds =  X %*% t(beta) # column J of the result will hold the predictions for all samples by model J
  for(k in 1:K){
    preds[ which(groups == k) ] = allpreds[,k][which(groups == k)] ## update preds per task
  }
  
  #### todo: accumualte changes in beta to draw the regularization path. update only in j_star
  betas_norm2[m,] = betas_norm2[max(m-1,1),] ## copy all pervious norm2
  betas_norm2[m,j_star] = norm2(beta[,j_star]) ## the updated norm2 for the udpated subspace
  
}
ret=list()
ret[["beta"]]=beta
ret[["betas_norm2"]]=betas_norm2
return(ret)
}









data = GenerateLinearData(ntrain = 5000)
train = data$data[data$trainidx,]
X = train[,-which(colnames(train) %in% c("Label","Family"))]
y = train[,"Label"]
groups = train[,"Family"]






res = algo1(X,y,groups,M=600,eps=0.01)
## plotting tthe path from the reuslt
plotPath(res)
blockIndex=2
plotBlock(res,blockIndex)


if(FALSE){
  gplassotraindata = CreateGroupLassoDesignMatrix(train)
  gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
  gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
  gplassoy[gplassoy==-1]=0
  
  #mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=2, seed=777,family="binomial",trace=TRUE,penalty="grLasso")
  mgplasso = grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, seed=777,family="binomial",trace=TRUE,penalty="grLasso",nlambda = 20)
  plot(mgplasso,norm=TRUE,label=TRUE,log.l = FALSE, alpha = 1)
  
}
