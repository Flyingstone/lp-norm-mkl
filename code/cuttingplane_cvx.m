function [eta, theta, alpha] = cuttingplane_cvx(c, p, Q_thetaDim, y, maxIterations)

% c 
% p
% Q_thetaDim

thetaDim = size(Q_thetaDim, 3);
alphaDim = size(y, 1);

% Program starts
satisfied = 0;
thetaStar = thetaDim^(-1/p) * ones(thetaDim,1);  % initialize so that ||thetastar|| = 1
prevAlpha = [];
doneFlag = 0;
Qvector = [];
etaTimeline = []; 

while ((satisfied < maxIterations) && (doneFlag < 1))
    % Inner loop : Given theta, find alpha
    weightedQ = zeros(alphaDim);
    for i = 1:thetaDim
         weightedQ = weightedQ + thetaStar(i) * Q_thetaDim(:,:,i);
    end    

    A = ((eye(alphaDim)/(2*c)) + weightedQ);
    
    cvx_begin
        variable alphaStar(alphaDim)
        maximize (-alphaStar'*y - 0.5*alphaStar'*A*alphaStar)
    cvx_end    
    
    prevAlpha = [prevAlpha alphaStar];
    
    % Outer loop : Given alpha find theta
    Qvector = [Qvector zeros(thetaDim,1)];
    for i = 1:thetaDim       
        Qvector(i,size(Qvector,2)) = prevAlpha(:,size(Qvector,2))'* Q_thetaDim(:,:,i) *prevAlpha(:,size(Qvector,2));       
    end
    
    if(satisfied > 0)
        doneFlag = 1;
        for i = 1:size(prevAlpha,2)
            if( etaStar + prevAlpha(:,i)'*y + (0.25/c) * prevAlpha(:,i)'* prevAlpha(:,i) + 0.5 * thetaStar' * Qvector(:,i) < 0)                
                disp('at least 1 constraint violated!');
                doneFlag = 0;
                break;
            end
        end    
    end
    
    if(doneFlag == 0)
      cvx_begin
            variable thetaStar(thetaDim); 
            variable etaStar(1);        
            minimize ( etaStar );
            subject to                                    
                for i = 1:size(prevAlpha,2)
                    etaStar + prevAlpha(:,i)'*y + (0.25/c) * prevAlpha(:,i)'* prevAlpha(:,i) + 0.5 * thetaStar' * Qvector(:,i) >= 0;                
                end
                norm(thetaStar,p) <= 1;
                thetaStar >= 0;
        cvx_end
        etaTimeline = [etaTimeline etaStar];
        satisfied = satisfied + 1;
    else
        disp('EARLY EXIT! ');
        satisfied
    end
end
eta = etaStar;
theta = thetaStar;
alpha = alphaStar;