function performance = errormeasure(train_y,ptrain_y,test_y,ptest_y)

ymax = max(max(train_y),max(test_y));
ymin = min(min(train_y),min(test_y));
scalesize = ymax-ymin;
RMSEtrain = errperf (train_y,ptrain_y,'rmse');
RMSEtest = errperf (test_y,ptest_y,'rmse');
MSEtrain = errperf (train_y,ptrain_y,'mse');
MSEtest = errperf (test_y,ptest_y,'mse');
MAEtrain = errperf (train_y,ptrain_y,'mae');
MAEtest = errperf (test_y,ptest_y,'mae');
MAPEtrain = errperf (train_y,ptrain_y,'mape');
MAPEtest = errperf (test_y,ptest_y,'mape');
%
performance = [RMSEtrain,RMSEtest,MSEtrain,MSEtest,MAPEtrain,MAPEtest,MAEtrain,MAEtest];
