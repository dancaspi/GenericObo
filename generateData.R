source("helper.R")
library(MASS)
library(grpreg)
library(xtable)
library(knitr)
source("helper.R")
### our boolean functino is, for example: x1 and X2 and X7 and X8 (or some other function, maybe less linear)
### we classify as 1 all cases when this is TRUE and 0 otherwise
### each family has a slightly different probability of Xi being 1. 
outdir="PandoSimulationResults"
dir.create(outdir,showWarnings = FALSE)

boolfunc = function(x){
  

  i=x[length(x)]
  
  #ret = (xor(x[1],x[6])|xor(x[2],x[7])|xor(x[3],x[8]))&(xor(x[4],x[9])|xor(x[5],x[11]) | xor(x[10],x[i]))
  ret = (xor(x[1],x[6])|xor(x[2],x[7])|xor(x[3],x[i]))&(xor(x[4],x[9])|xor(x[5],x[11]) | xor(x[10],x[i]))
  

  
  #return(ret)
  if(ret==0){
    return(-1)
  }else{
    return(1)
  }
}


GenerateNonLinearData=function(ntasks=5,d=50,ntrain=1000,ntest=300,seed=777){
  
  ## for each task, we generate a perturbation vector. this changes the probability of each bernoulli variable
  ## d bernulli variables overall with some independent distribution each to turn on/off
  ## each "family" is determined by whether some subset of flags was on
  ## if so it belogns to the fmaily, otherwise it's not
  ## 
  n=ntrain+ntest
  set.seed(seed)
  relevantdim = 11
  
  qs = runif(d,0.1,0.4) ## probability for each flag to be on
  #qs[c(1,2,3,4,5)]=runif(5,0.4,0.6)
  #qs[1:ntasks]=runif(ntasks,0.3,0.7)
  qs[1:ntasks]=runif(ntasks,0.4,0.7)
  allmatrix=c()
  groups=c()
  
  ## make sure we stay within (0,1)
  
  #tq=c(0.03741054 ,0.85404431 ,0.46901895 ,0.23391561 ,0.10628011 ,0.57462687 ,0.45676139, 0.20296465, 0.13046421, 0.31055056)
  flagsmatrix=matrix(nrow=ntasks,ncol=d)
  for(t in 1:ntasks){
    ### generate a sample of n=ntrain+ntest samples, each of dimension d
    ##generate a boolean vector using binomial distribution
    ## build the matrix for this task colum wise, we generate each column independently
    #per = rnorm(n=d,mean=0,sd=0.07) ## perturbation per task, intuitively: the more we perturb, the less "in common" the tasks have
    per = rnorm(n=d,mean=0,sd=0.07) ## perturbation per task, intuitively: the more we perturb, the less "in common" the tasks have
    tq=qs+per ## perturb probabilities for this task
    minval = min(tq[tq>0])
    maxval=max(tq[tq<1])
    tq[tq<=0]=minval
    tq[tq>=1]=maxval
    flagsmatrix[t,]=tq
    taskmatrix=matrix(nrow=n,ncol=relevantdim)
    for(dd in 1:relevantdim){
      
      p=tq[dd] ## the relevant probability for this column for this task
      taskmatrix[,dd] =rbinom(n,1,p) ## generate n samlpes with probability p
    }
    
    perturbedvars=d-relevantdim
    
    perturbmat = matrix( rnorm(perturbedvars*nrow(taskmatrix),mean=0,sd=1), nrow=nrow(taskmatrix)) 
    #perturbmat = cbind(matrix(0,nrow=nrow(taskmatrix),ncol=ncol(taskmatrix)-perturbedvars),perturbmat)
    #taskmatrix = taskmatrix+perturbmat ## add some noise 
    taskmatrix = cbind(taskmatrix,perturbmat) ## add many noise variables
    taskmatrix=cbind(taskmatrix,matrix(t,nrow=nrow(taskmatrix),ncol=1))
    allmatrix=rbind(allmatrix,taskmatrix)
    groups = rbind(groups,matrix(paste0("fam",t),nrow=nrow(taskmatrix),ncol=1))
  }
  y=apply(allmatrix,1,boolfunc)
  allmatrix=allmatrix[,-ncol(allmatrix)]
  df=data.frame(allmatrix)
  ret=list()
  df["Label"]=y
  df["Family"]=groups  
  ret[["data"]]=df
  ret[["groups"]]=groups
  # testidx = c()
  # for(fam in unique(df[,"Family"])){
  #   testidx = c(testidx, sample(which(df[,"Family"]==fam),ntest))
  # }
  
  testidx = c()
  validx = c()
  trainidx = c()
  for(fam in unique(df[,"Family"])){
    
    testidx = c(testidx, sample(which(df[,"Family"]==fam),ntest ))
    famtrain = setdiff(which(df[,"Family"]==fam),testidx)
    famvalidx = sample(famtrain,0.1*length(famtrain) ) #val 10% of train
    validx = c(validx,famvalidx)
    famtrain = setdiff(famtrain,validx) # remove validation from train
    trainidx = c(trainidx,famtrain)
  }
  
  ret[["testidx"]]=testidx
  ret[["trainidx"]]=trainidx
  ret[["validx"]]=validx
  ret[["qs"]]=qs
  ret[["per"]]=per
  ret[["tq"]]=tq
  ret[["flagsmatrix"]]=flagsmatrix
  ### do the same output as GenerateData to have everything run smoothly
  return(ret)    
}





GenerateLinearData = function(ntasks=5, d=50,ntrain=1000,ntest=300,seed=300){
  #ntasks=5 ### we generate a row for each task, so for each task we have a set of parameter weights
  #d=15 ### how many dimensions we have. we have 5 dimensions with actual value, and 5 which will be noise
  #n=100 #samples per task
  n=ntrain+ntest
  set.seed(seed)
  mu=rep(0,5)

  Sigma=diag(c(1,0.9,0.8,0.7,0.6)*0.5)
  W = mvrnorm(ntasks,mu,Sigma)  ### this generate a 5x5 matrix, a row of coefficients per task
  
  colnames(W)=NULL
  zz = matrix(0,ntasks,d-5)
  W = cbind(W,zz)
  
  
  ## each row of W is the "controller" (Wt) of each task
  
  ## now generate the random data X ~ uniform(0,1)^d
  ret=list()
  M = c()
  Y = c()
  groups = c()
  for(i in 1:ntasks){
    X = matrix(runif(n*d),nrow=n,ncol=d)
    offsets = c(1,1,1,1,1)
    w = t(W)[,i] + c(offsets,rep(0,d-5))
    y = X %*% w
    y = y + rnorm(n=length(y),mean=0,sd=0.0)
    ret[["rawscore"]]=y
    #y = 1/(1+exp(-0.5*y))
    # transform y to 0,1 uniformly
    #y = (y-min(y)) * (0.999/(max(y)-min(y)))+0.001 ### newValue = ((oldValue - oldMin) * newRange / oldRange) + newMin --> transform y to 0,1
    a=5
    y=(a*y)-median(a*y)
    #y=1/(1+exp(y))
    y=exp(y)/(1+exp(y))
    ret[["transformedscore"]]=y
    # perform logit
    
    

    # now we have a probability for each y to be in class 1 or 0
    
    y=rbinom(length(y),1,y) ## for each value of y, we took it as  aprobability to be in 1 or 0
    y[y==0]=-1 ## label 0 as -1
    ret[["y"]]=y
    #class1idx=which(y>0.6)
    #class0idx=which(y<=0.6)
    #y[class1idx]=1
    #y[class0idx]=-1
    #y[y==0]=-1
    
    M = rbind(M,X)
    Y = rbind(Y,matrix(y,nrow=length(y),ncol=1))
    groups = rbind(groups,matrix(paste0("fam",i),nrow=length(y),ncol=1))
  }
  
  ### clean = less than median
  ### mal = more than median
  df=data.frame(M)
  
  df["Label"]=Y
  
  df["Family"]=groups
  ## create test indexes
  testidx = c()
  validx = c()
  trainidx = c()
  for(fam in unique(df[,"Family"])){
    
    testidx = c(testidx, sample(which(df[,"Family"]==fam),ntest ))
    famtrain = setdiff(which(df[,"Family"]==fam),testidx)
    famvalidx = sample(famtrain,0.1*length(famtrain) ) #val 10% of train
    validx = c(validx,famvalidx)
    famtrain = setdiff(famtrain,validx) # remove validation from train
    trainidx = c(trainidx,famtrain)
  }
  
  ret[["testidx"]]=testidx
  ret[["trainidx"]]=trainidx
  ret[["validx"]]=validx
  
  ret[["data"]]=df
  ret[["groups"]]=groups
  ret[["W"]]=W
  
  
  return(ret)
}

####### NON LINEAR ######
nonLionearExmp = function(){
  d=75
  ntasks=5
  ntest=10000
  ntrain=1000 ## per task
  #controls=c(maxdepth=2,minbucket=1)
  set.seed(777)
  data=GenerateNonLinearData(d=d,ntasks=ntasks,ntrain=ntrain,ntest=ntest,seed=201)
  train = data$data[data$trainidx,]
  test = data$data[data$testidx,]
  val = data$data[data$validx,]
  cat("starting pando\n")
  mshared=TunePando(TrainMultiTaskClassificationGradBoost,train,val,fitTreeCoef="nocoef",fitLeafCoef="ridge",trainrate = 0.01,cviter=100,maxdepths=c(3,5,7),cps=c(0),cv=-1,cvrate=0.01)
  cat("starting pando2\n")
  mshared2=TunePando(TrainMultiTaskClassificationGradBoost2,train,val,fitTreeCoef="ridge",fitLeafCoef="nocoef",trainrate = 0.01,cviter=100,maxdepths=c(3,5,7),cps=c(0),cv=-1,cvrate=0.01)
  
  perTaskModels=list()
  logitModels=list()
  for(fam in unique(train[,"Family"])){
    
    tr = train[train[,"Family"]==fam,]
    tr.val = val[val[,"Family"]==fam,]
    
    m0=TunePando(vanillaboost2,tr,tr.val,trainrate = 0.01,cviter=100,maxdepths=c(3,5,7),cps=c(0),cv=-1,cvrate=0.01)
    perTaskModels[[toString(fam)]]=m0
    logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(tr[,-which(colnames(tr) %in% c("Family","Label"))]),y=tr[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
  }
  
  ### train binary model, ignoring multi tasking:
  binaryData = train
  binaryData["Family"]="1"
  binaryVal = val
  binaryVal["Family"]="1"
  mbinary=TunePando(vanillaboost2,binaryData,binaryVal,fitTreeCoef="nocoef",fitLeafCoef="nocoef",trainrate = 0.01,cviter=100,maxdepths=c(3,5,7),cps=c(0),cv=-1,cvrate=0.01)
  mlogitbinary = cv.glmnet(x=as.matrix(binaryData[,-which(colnames(tr) %in% c("Family","Label"))]),y=binaryData[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
  
  gplassotraindata = CreateGroupLassoDesignMatrix(train)
  gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
  gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
  gplassoy[gplassoy==-1]=0
  
  mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=2, seed=777,family="binomial",trace=TRUE)
  
  gplassotestdata = CreateGroupLassoDesignMatrix(test)
  gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
  gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]
  
  gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)   
  
  
  methods = c("PANDO","PANDO2","PTB","BB","PTLogit","BinaryLogit","GL")
  rc=list()
  tt=list()
  compmat = c()
  digitsfmt = matrix(-2,nrow=length(methods),ncol=length(methods))
  xtables=list()
  k=0
  
  
  allpreds = matrix(nrow=nrow(test),ncol=length(methods)+2)
  #colnames(allpreds)=c(methods,"Label","testnum")
  colnames(allpreds)=c(methods,"Label","Family")
  allpreds[,"Label"]=test[,"Label"]
  allpreds[,"Family"]=test[,"Family"]
  
  
  ##################### test:
  for(fam in unique(test[,"Family"])){
    k = k+1
    testidxs = which(test["Family"]==fam)
    compmatrix = matrix(NA,nrow=length(methods),ncol = length(methods))
    
    tr.test = test[test["Family"]==fam,]
    tr.test = tr.test[,-which(colnames(tr.test)=="Family")]
    
    if("PTB" %in% methods){
      tt[[methods[which(methods=="PTB")]]]= predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }
    if("BB" %in% methods){
      tt[[methods[which(methods=="BB")]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }
    if("BB2" %in% methods){
      tt[[methods[which(methods=="BB2")]]] = predict(mbinary2[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }  
    if("PTLogit" %in% methods){
      tt[[methods[which(methods=="PTLogit")]]] = predict(logitModels[[toString(fam)]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[toString(fam)]]$lambda.min)
    }
    if("BinaryLogit" %in% methods){
      tt[[methods[which(methods=="BinaryLogit")]]] = predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
    }
    if("PANDO" %in% methods){
      tt[[methods[which(methods=="PANDO")]]] = predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }
    if("PANDO2" %in% methods){
      tt[[methods[which(methods=="PANDO2")]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }
    
    if("GL" %in% methods){
      tt[[methods[which(methods=="GL")]]] = gplassoPreds[test[,"Family"]==fam]
    }
    
    for(i in 1:length(methods)){
      rc[[methods[i]]]=pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[i]]]))
      allpreds[testidxs,methods[i]] = as.matrix(tt[[methods[i]]],ncol=1)
    }
    
    for(i in 1:length(methods)){
      compmatrix[i,i]=rc[[methods[i]]]$auc[1]
      digitsfmt[i,i]=3
      for(j in 1:length(methods)){
        if(i >=j ){
          next
        }
        #cat("setting compmatrix",i," ",j,"\n")
        compmatrix[i,j] = signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)
        if(rc[[methods[i]]]$auc[1] < rc[[methods[j]]]$auc[1]){
          compmatrix[i,j] = compmatrix[i,j]*-1
        }
        cat("auc  for ",fam," ", methods[i]," VS ",methods[j],": ",round((rc[[methods[i]]]$auc[1]),4)," ",round((rc[[methods[j]]]$auc[1]),4),"\n")
      }
    }
    #compmat = rbind(compmat,compmatrix)
    compmat[[fam]] = compmatrix
    cat("***********\n")
    
    
    dft=data.frame(compmatrix)
    colnames(dft)=methods
    rownames(dft)=methods
    xtables[[toString(k)]]=xtable(dft,digits=cbind(rep(1,nrow(digitsfmt)),digitsfmt))
    
  }
  ####################################################################################################
}



###### LINEAR ########
linearExample = function(){
  d=25
  ntest=10000
  ntrain=1000
  controls=rpart.control()
  iter=200
  rate=0.01
  ridge.lambda=1 
  set.seed(777)
  data=GenerateLinearData(d=d,ntrain=ntrain,ntest=ntest)
  train = data$data[data$trainidx,]
  test = data$data[data$testidx,]
  val = data$data[data$validx,]
  cat("starting pando\n")
  mshared=TunePando(TrainMultiTaskClassificationGradBoost,train,val,fitTreeCoef="nocoef",fitLeafCoef="ridge",trainrate = 0.01,cviter=100,maxdepths=c(3,5,7),cps=c(0),cv=-1,cvrate=0.01)
  cat("starting pando2\n")
  mshared2=TunePando(TrainMultiTaskClassificationGradBoost2,train,val,fitTreeCoef="ridge",fitLeafCoef="nocoef",trainrate = 0.01,cviter=100,maxdepths=c(3,5,7),cps=c(0),cv=-1,cvrate=0.01)
  
  perTaskModels=list()
  logitModels=list()
  for(fam in unique(train[,"Family"])){
    
    tr = train[train[,"Family"]==fam,]
    tr.val = val[val[,"Family"]==fam,]
    
    m0=TunePando(vanillaboost2,tr,tr.val,trainrate = 0.01,cviter=100,maxdepths=c(3,5,7),cps=c(0),cv=-1,cvrate=0.01)
    perTaskModels[[toString(fam)]]=m0
    logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(tr[,-which(colnames(tr) %in% c("Family","Label"))]),y=tr[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
  }
  
  ### train binary model, ignoring multi tasking:
  binaryData = train
  binaryData["Family"]="1"
  binaryVal = val
  binaryVal["Family"]="1"
  mbinary=TunePando(vanillaboost2,binaryData,binaryVal,fitTreeCoef="nocoef",fitLeafCoef="nocoef",trainrate = 0.01,cviter=100,maxdepths=c(3,5,7),cps=c(0),cv=-1,cvrate=0.01)
  mlogitbinary = cv.glmnet(x=as.matrix(binaryData[,-which(colnames(tr) %in% c("Family","Label"))]),y=binaryData[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
  
  gplassotraindata = CreateGroupLassoDesignMatrix(train)
  gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
  gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
  gplassoy[gplassoy==-1]=0
  
  mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=2, seed=777,family="binomial",trace=TRUE)
  
  gplassotestdata = CreateGroupLassoDesignMatrix(test)
  gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
  gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]
  
  gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)   
  
  
  methods = c("PANDO","PANDO2","PTB","BB","PTLogit","BinaryLogit","GL")
  rc=list()
  tt=list()
  compmat = c()
  digitsfmt = matrix(-2,nrow=length(methods),ncol=length(methods))
  xtables=list()
  k=0
  
  
  allpreds = matrix(nrow=nrow(test),ncol=length(methods)+2)
  #colnames(allpreds)=c(methods,"Label","testnum")
  colnames(allpreds)=c(methods,"Label","Family")
  allpreds[,"Label"]=test[,"Label"]
  allpreds[,"Family"]=test[,"Family"]
  
  
  ##################### test:
  for(fam in unique(test[,"Family"])){
    k = k+1
    testidxs = which(test["Family"]==fam)
    compmatrix = matrix(NA,nrow=length(methods),ncol = length(methods))
    
    tr.test = test[test["Family"]==fam,]
    tr.test = tr.test[,-which(colnames(tr.test)=="Family")]
    
    if("PTB" %in% methods){
      tt[[methods[which(methods=="PTB")]]]= predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }
    if("BB" %in% methods){
      tt[[methods[which(methods=="BB")]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }
    if("BB2" %in% methods){
      tt[[methods[which(methods=="BB2")]]] = predict(mbinary2[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }  
    if("PTLogit" %in% methods){
      tt[[methods[which(methods=="PTLogit")]]] = predict(logitModels[[toString(fam)]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[toString(fam)]]$lambda.min)
    }
    if("BinaryLogit" %in% methods){
      tt[[methods[which(methods=="BinaryLogit")]]] = predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
    }
    if("PANDO" %in% methods){
      tt[[methods[which(methods=="PANDO")]]] = predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }
    if("PANDO2" %in% methods){
      tt[[methods[which(methods=="PANDO2")]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
    }
    
    if("GL" %in% methods){
      tt[[methods[which(methods=="GL")]]] = gplassoPreds[test[,"Family"]==fam]
    }
    
    for(i in 1:length(methods)){
      rc[[methods[i]]]=pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[i]]]))
      allpreds[testidxs,methods[i]] = as.matrix(tt[[methods[i]]],ncol=1)
    }
    
    for(i in 1:length(methods)){
      compmatrix[i,i]=rc[[methods[i]]]$auc[1]
      digitsfmt[i,i]=3
      for(j in 1:length(methods)){
        if(i >=j ){
          next
        }
        #cat("setting compmatrix",i," ",j,"\n")
        compmatrix[i,j] = signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)
        if(rc[[methods[i]]]$auc[1] < rc[[methods[j]]]$auc[1]){
          compmatrix[i,j] = compmatrix[i,j]*-1
        }
        cat("auc  for ",fam," ", methods[i]," VS ",methods[j],": ",round((rc[[methods[i]]]$auc[1]),4)," ",round((rc[[methods[j]]]$auc[1]),4),"\n")
      }
    }
    #compmat = rbind(compmat,compmatrix)
    compmat[[fam]] = compmatrix
    cat("***********\n")
    
    
    dft=data.frame(compmatrix)
    colnames(dft)=methods
    rownames(dft)=methods
    xtables[[toString(k)]]=xtable(dft,digits=cbind(rep(1,nrow(digitsfmt)),digitsfmt))
    
  }
  ####################################################################################################
  
}










