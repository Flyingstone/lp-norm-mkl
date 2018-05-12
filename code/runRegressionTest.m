if (testtype == 1)
    % random test
    c = 3;
    p = 5;
    thetaDim = 100;   % alphaDimension of theta
    alphaDim = 100;   % alphaDimension of alpha

    Q_thetaDim = zeros(alphaDim,alphaDim,thetaDim);
    myridge = 1e-8;

    for i = 1:thetaDim
        randVector = rand(alphaDim,1);
        Q_thetaDim(:,:,i) = randVector*randVector' + eye(alphaDim)*myridge;    
    end

    y = rand(alphaDim,1) - rand(alphaDim,1);
elseif (testtype == 2)
    % fitting a gaussian data with mean at 0 and std dev 1
    % using polynomial kernels
    c = 3;
    p = 5;
    thetaDim = 3;
    alphaDim = 40;
    
    x = randn(alphaDim, 1);
    x = sort(x);
    y = (1/sqrt(2*pi))*exp(-0.5 * x.^2);
    
    Q_thetaDim = zeros(alphaDim,alphaDim,thetaDim);
    myridge = 1e-8;
    for i = 1:thetaDim
        for j = 1:alphaDim
            for k = 1:alphaDim
                Q_thetaDim(j,k,i) = (x(j)*x(k) + 1)^i;
            end
        end
        Q_thetaDim(:,:,i) = Q_thetaDim(:,:,i) + eye(alphaDim)*myridge;
    end
elseif (testtype == 3)
    % fitting a gaussian data with mean at 0 and std dev 1
    % using exponential kernels
    c = 3;
    p = 5;
    thetaDim = 3;
    alphaDim = 40;
    
    x = randn(alphaDim, 1);
    x = sort(x);
    y = (1/sqrt(2*pi))*exp(-0.5 * x.^2);
    
    Q_thetaDim = zeros(alphaDim,alphaDim,thetaDim);
    
    for i = 1:thetaDim
        for j = 1:alphaDim
            for k = 1:alphaDim
                Q_thetaDim(j,k,i) = exp(-(1/i)*norm(x(j) - x(k)));
            end
        end
    end
elseif (testtype == 4)
    % fitting a bimodal gaussian data with means 0 and 3 and std dev 1 each
    % using exponential kernels
    c = 3;
    p = 5;
    thetaDim = 3;
    alphaDim = 40;
    
    x1 = randn(alphaDim, 1); x2 = randn(alphaDim, 1) + 3;
    x = [x1; x2];
    x = x(randperm(2*alphaDim));
    x = x(1:alphaDim);
    x = sort(x);
    y1 = (1/sqrt(2*pi))*exp(-0.5 * x.^2);
    y2 = (1/sqrt(2*pi))*exp(-0.5 * (x - 3).^2);
    y = max(y1, y2);

    Q_thetaDim = zeros(alphaDim,alphaDim,thetaDim);
    
    for i = 1:thetaDim
        for j = 1:alphaDim
            for k = 1:alphaDim
                Q_thetaDim(j,k,i) = exp(-(1/i)*norm(x(j) - x(k)));
            end
        end
    end
elseif (testtype == 5)
    % fitting a bimodal gaussian data with means 0 and 3 and std dev 1 each
    % using a mixture of polynomial and gaussian kernels
    c = 3;
    p = 3;
    thetaDim = 4;
    alphaDim = 40;
    
    x1 = randn(alphaDim, 1); x2 = randn(alphaDim, 1) + 3;
    x = [x1; x2];
    x = x(randperm(2*alphaDim));
    x = x(1:alphaDim);
    x = sort(x);
    y1 = (1/sqrt(2*pi))*exp(-0.5 * x.^2);
    y2 = (1/sqrt(2*pi))*exp(-0.5 * (x - 3).^2);
    y = max(y1, y2);

    Q_thetaDim = zeros(alphaDim,alphaDim,thetaDim);
    
    myridge = 1e-8;
    for i = 1:thetaDim/2
        for j = 1:alphaDim
            for k = 1:alphaDim
                Q_thetaDim(j,k,i) = exp(-(1/i)*(norm(x(j) - x(k))^2));
            end
        end
        Q_thetaDim(:,:,i) = Q_thetaDim(:,:,i) + eye(alphaDim)*myridge;
    end
    for i = thetaDim/2:thetaDim
        for j = 1:alphaDim
            for k = 1:alphaDim
                Q_thetaDim(j,k,i) = (x(j)*x(k) + 1)^(thetaDim + 1 - i);
            end
        end
        Q_thetaDim(:,:,i) = Q_thetaDim(:,:,i) + eye(alphaDim)*myridge;
    end
end

if (codevector(1) == 1)
    [eta1, theta1, alpha1] = full_cvx(c, p, Q_thetaDim, y);
    ypred1 = zeros(alphaDim,1);
    for i = 1:alphaDim
       for j = 1:thetaDim
           ypred1(i) = ypred1(i) + Q_thetaDim(:,i,j)'*alpha1*theta1(j);
       end
       ypred1(i) = -1*ypred1(i);
    end
end

if (codevector(2) == 1)
    [eta2, theta2, alpha2] = cuttingplane_cvx(c, p, Q_thetaDim, y, 200);
    ypred2 = zeros(alphaDim,1);
    for i = 1:alphaDim
       for j = 1:thetaDim
           ypred2(i) = ypred2(i) + Q_thetaDim(:,i,j)'*alpha2*theta2(j);
       end
       ypred2(i) = -1*ypred2(i);
    end
end

if (codevector(3) == 1)
    [eta3, theta3, alpha3,etaTimeline] = cuttingplane_sedumi(Q_thetaDim, y, p, c, 200);
    ypred3 = zeros(alphaDim,1);
    for i = 1:alphaDim
       for j = 1:thetaDim
           ypred3(i) = ypred3(i) + Q_thetaDim(:,i,j)'*alpha3*theta3(j);
       end
       ypred3(i) = -1*ypred3(i);
    end
end
