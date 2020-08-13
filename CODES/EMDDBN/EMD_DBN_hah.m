AEMO_importdata_hah;

%%EMD
options.MAXMODES=5;
IMF=emd(DATA, options);
IMF=IMF';
%% IMF TS2Matrix
for k = 1:options.MAXMODES+1
    if k<options.MAXMODES+1
    [scaled_IMF(:,k),max_IMF(k),min_IMF(k)]=scale_data(IMF(:,k),1,0,[],[]);
    else [scaled_IMF(:,k),max_IMF(k),min_IMF(k)]=scale_data(IMF(:,k),1,0,[],[]);
    end;
    for i=1:totalsize+samplesize-48;
        for j=1:48;
        scaled_IMF_x(i,j,k)=scaled_IMF(i+j-1,k);
        end;
    end;
    for i=1:trainsize;
        for j=1:48;          
            train_IMF_x(i,j,k)=scaled_IMF_x(i,j,k);
        end;
    end;

    train_IMF_y(:,k)=scaled_IMF(samplesize+1:samplesize+trainsize,k);

    for i=1:testsize;
        for j=1:48;
            test_IMF_x(i,j,k)=scaled_IMF_x(i+trainsize,j,k);
        end;
    end;

    test_IMF_y(:,k)= scaled_IMF(trainsize+samplesize+1:trainsize+testsize+samplesize,k);
end;

%% EMD-ANN
    for k=1:size(IMF,2)
        k
        rand('state',0)
        dbn.sizes = [48 48];
        opts.batchsize = 6;
        opts.momentum  = 0.4;
        opts.alpha     = 0.1;
        opts.numepochs = 20;
        dbn = dbnsetup(dbn, train_IMF_x(:,:,k), opts);
        dbn = dbntrain(dbn, train_IMF_x(:,:,k), opts);

%unfold dbn to nn
        nn = dbnunfoldtonn(dbn, 1);
        nn.activation_function = 'sigm';
        nn.learningRate                     = 1;   
%train nn
        opts.numepochs = 100;
        opts.batchsize = 6;
        nn = nntrain(nn,train_IMF_x(:,:,k), train_IMF_y(:,k), opts);
        [er, bad] = nntest(nn, test_IMF_x(:,:,k), test_IMF_y(:,k));
        
        predict_y(:,k) = nnpredicty (nn,train_IMF_x(:,:,k));
        ptest_y(:,k) = nnpredicty (nn, test_IMF_x(:,:,k));
       
        unscaled_predict_y(:,k)=predict_y(:,k)*(max_IMF(k)-min_IMF(k))+min_IMF(k);
        unscaled_ptest_y(:,k)=ptest_y(:,k)*(max_IMF(k)-min_IMF(k))+min_IMF(k);
        end;
        
        for j = 1:54,
            if j<49
                sumtrain_x(:,j) = train_x(:,j);
                sumtest_x(:,j) = test_x(:,j);
            else 
                sumtrain_x(:,j) = predict_y(:,j-48);
                sumtest_x(:,j) = ptest_y(:,j-48);
            end;
        end;       
 %%
 
        rand('state',0)
        dbn.sizes = [54 54];  
        opts.batchsize = 6;
        opts.momentum  = 0.4;
        opts.alpha     = 0.1;
        opts.numepochs = 20;
        dbn = dbnsetup(dbn, sumtrain_x, opts);
        dbn = dbntrain(dbn, sumtrain_x, opts);
        nn = dbnunfoldtonn(dbn, 1);
        nn.activation_function = 'sigm';
        nn.learningRate                     = 1;   
        opts.numepochs = 5000;
        opts.batchsize = 6;
        nn = nntrain(nn,sumtrain_x, train_y, opts);
        [er, bad] = nntest(nn, sumtest_x, test_y);
        
        EMDFNN_predict_train= nnpredicty (nn,sumtrain_x);
        EMDFNN_predict_test = nnpredicty (nn, sumtest_x);


%% error measure
unscaledpredict_y2 = EMDFNN_predict_train * scalesize + minvalue;
unscaledptest_y2 = EMDFNN_predict_test * scalesize + minvalue;
unscaledtrain_y2 = DATA(samplesize+1:samplesize+trainsize)' * scalesize + minvalue;
unscaledtest_y2 = DATA(samplesize+trainsize+1:samplesize+trainsize+testsize)' * scalesize + minvalue;

figure;
plot(unscaledtrain_y2,'b');
hold on;
plot(unscaledpredict_y2,'r');
legend('Original Data','Regression Data');
grid on;


figure;
plot(unscaledtest_y2,'b');
hold on;
plot(unscaledptest_y2,'r');
legend('Original Data','Regression Data');
grid on;


performance = errormeasure(unscaledtrain_y2,unscaledpredict_y2',unscaledtest_y2,unscaledptest_y2');