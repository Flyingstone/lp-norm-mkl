function [eta, theta, alpha,etaTimeline] = cuttingplane_sedumi(Q, y, p, c, maxIterations)
%CUTTING PLANE ALGORITHM WITH SeDuMi
%   This is the implementation of the cutting plane algorithm for problem 4
%   in the project.
%
%   The objective function is given as,
%       min_{\theta} max_{\alpha} -\alpha'y - (1/4c)\alpha'\alpha -
%       (1/2)\alpha'(\sum_m \theta_m Q_m)\alpha
%           subject to \theta >= 0, norm(\theta, p) <= 1
%
%   Q - a 3-dimensional matrix which contains all the 2-dimensional kernel
%   matrices. Each kernel matrix is indexed by the third dimension. For e.g.
%   Q(:,:,1) is the first kernel matrix, Q(:,:,2) is the second kernel
%   matrix, so on and so forth.
%
%   y - the list of y values for regression
%   
%   p - a constant indicating the norm to be used in the constraint
%
%   c - simply a constant of the objective function
%
%   maxIterations - the cutting plane algorithm is iterative in nature. This
%   variable puts a limit on the number of iterations, in case the
%   termination condition is not met.
%
%   [eta, theta, alpha] = cuttingplane_sedumi(Q, y, p, c, maxIterations)
%   
%   eta - is the optimum objective value achieved
%   theta - is the optimal theta value
%   alpha - is the optimal alpha value
%

% computing dimension of theta and alpha
thetaDim = size(Q, 3);
alphaDim = size(y, 1);

% a variable to keep track of iterations
iteration = 0;

% initialize so that ||thetastar|| = 1
thetaStar = thetaDim^(-1/p) * ones(thetaDim,1);

% a book-keeping parameter that retains all the alpha that forms our
% cutting plane
prevAlpha = [];

% a variable to check termination condition
doneFlag = 0;

% alphaQAlpha corresponds to a vector of the form
% [\alpha'Q1\alpha \alpha'Q2\alpha ... \alpha'Qm\alpha]'
% 
% However, we will have one alpha per iteration (cutting plane constraint), 
% so alphaQAlpha will actually hold
%
% [\alpha1'Q1\alpha1 \alpha1'Q2\alpha1 ... \alpha1'Qm\alpha1]'
% [\alpha2'Q1\alpha2 \alpha2'Q2\alpha2 ... \alpha2'Qm\alpha2]'
% ...
% ...
% [\alpha_iteration'Q1\alpha_iteration \alpha_iteration'Q2\alpha_iteration ... \alpha_iteration'Qm\alpha_iteration]'
% (P.S : Note the transpose)
alphaQAlpha = [];

% alphaYAlphaAlpha is simply a constant of the form 
% \alpha'*y + 1/4c*(alpha'*alpha)
%
% Again, since we will have one alpha per iteration, so alpha per
% iteration, so alphaYAlphaAlpha will actually hold
% 
% [\alpha1'*y + 1/4c*(alpha1'*alpha1) \alpha2'*y + 1/4c*(alpha2'*alpha2)
% ... \alpha_iteration'*y + 1/4c*(alpha'*alpha_iteration)]'
alphaYAlphaAlpha = [];

%%%%%
% Both alphaQAlpha and alphaYAlphaAlpha are needed to express SOCP
% constraints for SeDuMi
%%%%

%%%%
% Some debugging variables
% they can be commented out
%%%%
etaTimeline = []; 
thetaTimeline = [thetaStar'];

% do until termination condition reached
% or max iteration has been reached
while ((doneFlag < 1) && (iteration < maxIterations))
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Optimization problem 1
    %   Given a fixed theta,
    %   max_{\alpha}
    %       -\alpha'y - (1/4c)\alpha'\alpha -
    %       (1/2)\alpha'(\sum_m \theta_m Q_m)\alpha
    %   subject to
    %       No Constraints
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % computing weightedQ = \sum_m {\theta_m * Q_m}
    weightedQ = zeros(alphaDim);
    for i = 1:thetaDim
         weightedQ = weightedQ + thetaStar(i) * Q(:,:,i);
    end

    % compute the most violated constraint
    A = ((eye(alphaDim)/(2*c)) + weightedQ);
    alphaStar = -(A\y);
    
    % push the alpha into the book-keeping parameter
    prevAlpha = [prevAlpha alphaStar];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Optimization problem 2
    %   Given a fixed alpha,
    %   min_{\eta, \theta} 
    %       \eta
    %   subject to 
    %       \theta >= 0, norm(\theta, p) <= 1,
    %       \eta >= -\alpha'y - (1/4c)\alpha'\alpha -
    %       (1/2)\alpha'(\sum_m \theta_m Q_m)\alpha
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Update alphaQAlpha with the new alpha found above
    alphaQAlpha = [alphaQAlpha zeros(thetaDim,1)];
    for i = 1:thetaDim       
        alphaQAlpha(i, size(alphaQAlpha, 2)) = prevAlpha(:, size(alphaQAlpha, 2))'* Q(:, :, i) *prevAlpha(:, size(alphaQAlpha, 2));   
    end
    
    % Update alphaYAlphaAlpha with the new alpha found above
    alphaYAlphaAlpha = [alphaYAlphaAlpha; alphaStar'*y + 0.25/c*(alphaStar'*alphaStar)];
    
    % See if any of the current and previous constraints are getting
    % violated. If not, then we are done. If yes, we need to improve our
    % theta estimate
    % P.S. -> Note that in the first iteration, we don't have any estimate
    % for the objective value \eta, thereby we skip the constraint
    % violation check for the first iteration.
    if(iteration > 0)
        doneFlag = 1;
        for i = 1:size(prevAlpha,2)
            if( etaStar + prevAlpha(:,i)'*y + (0.25/c) * prevAlpha(:,i)'* prevAlpha(:,i) + 0.5 * thetaStar' * alphaQAlpha(:,i) < 0)                
                disp('at least 1 constraint violated!');                
                doneFlag = 0;
                break;
            end
        end    
    end

    % if atleast one of the constraints got violated
    % solve for new \eta, \theta
    if(doneFlag == 0)
        % Solving using SeDuMi
        % Objective min b'x
        b = [zeros(thetaDim,1);1];
        
        % \eta >= -\alpha'y - (1/4c)\alpha'\alpha -
        %       (1/2)\alpha'(\sum_m \theta_m Q_m)\alpha
        % can be written as a linear constraint D'x + f >= 0
        % for a fixed alpha
        % Hence with every iteration, 1 linear constraint gets added (cutting plane)
        % Also, \theta >= 0 adds thetaDim linear constraints
        D = [0.5*alphaQAlpha eye(thetaDim); ones(1, size(alphaQAlpha, 2)) zeros(1,thetaDim)];
        f = [alphaYAlphaAlpha; zeros(thetaDim,1)];
        
        % Norm constraint (SOC) can be written as 
        % ||A1'x + c1|| <= b1'x + d1
        % However, for p = 1, the second order norm constraint reduces to a
        % linear constraint and hence has to be handled separately
        if (p > 1)
            % if p > 1, write out the SOCP form
            P = diag(thetaStar.^(p - 2));
            q = thetaStar.^(p - 1);
            r = norm(thetaStar, p)^p;

            A1 = sqrt(p*(p-1)/2) * [sqrt(P) ;  zeros(1,thetaDim)];
            c1 = -sqrt(p/(2*(p - 1))) * (p-2) * (sqrt(P)\q);
            b1 = zeros(thetaDim+1, 1);
            d1 = sqrt( (p*(p-2)^2)/(2*(p - 1)) * q'*(P\q) + 1 - (1 + p*(p - 3)/2)*r );
        else
            % if p == 1, write out the linear form
            q = thetaStar.^(p - 1);
            r = norm(thetaStar, p)^p;
            
            D = [D [p*(p-2)*q; 0]];
            f = [f; 1-((1+p*(p-3)/2)*r)];
            A1 = []; c1 = []; b1 = []; d1 = [];
        end
        
        % Constructing SeDuMi matrices
        A11 = -[b1 A1];
        At = [-D A11];
        bt = -b;

        c11 = [d1; c1];
        ct = [f; c11];

        % K.l = no. of linear constraints
        % K.q = dimension of the SOCP constraints
        if (p > 1)
            K.l = size(alphaQAlpha, 2) + thetaDim;
            K.q = [thetaDim+1];
        else
            K.l = size(alphaQAlpha, 2) + thetaDim + 1;
            K.q = 0;
        end

        % Calling Mr. SeDuMi
        [xs,ys,info] = sedumi(At,bt,ct,K);
        
        % Our problem is in the dual form, hence the dual solution gives
        % us our theta and eta.
        thetaStar = ys(1:thetaDim);
        etaStar = ys(thetaDim+1);
        
        iteration = iteration + 1;
        
        % Updating the debugging variables, can be commented
        etaTimeline = [etaTimeline; etaStar];
        thetaTimeline = [thetaTimeline; thetaStar'];
    else
        % if no constraint violated, then exit
        disp('EARLY EXIT! ');
        iteration
    end
end
% Output Variable Values
eta = etaStar;
theta = thetaStar;
alpha = alphaStar;
