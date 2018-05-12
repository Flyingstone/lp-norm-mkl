function [eta, theta, alpha] = full_cvx(c, p, Q_thetaDim, y)

thetaDim = size(Q_thetaDim, 3);
alphaDim = size(y, 1);

cvx_begin
    variable alphaStar(alphaDim)
    variable t(1)
    variable A(thetaDim)
    
    maximize (-alphaStar'*y - 0.25/c*alphaStar'*alphaStar - 0.5*t)    
    
    subject to
        for i = 1:thetaDim
            A(i) >= alphaStar'*Q_thetaDim(:,:,i)*alphaStar;
        end        
        t >= norm(A, p/(p-1));
cvx_end

eta = cvx_optval;
alpha = alphaStar;

sum = 0;
for i = 1:thetaDim
    theta(i) = (alpha'*Q_thetaDim(:,:,i)*alpha)^(1/(p-1));
    sum = sum + theta(i)^p;
end
sum = sum^(1/p);
theta = theta/sum;
theta = theta';

weightedQ = zeros(alphaDim);
for i = 1:thetaDim
	weightedQ = weightedQ + theta(i) * Q_thetaDim(:,:,i);
end
B = ((eye(alphaDim)/(2*c)) + weightedQ);
finalval = -alpha'*y - 0.5*alpha'*B*alpha