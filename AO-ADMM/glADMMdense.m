function [ H, U, Yt, V, GG, itr ] = glADMMdense( Y, W, H, U, Yt, V, d, GG, ops)
% ADMM iterates to solve
%       minimize l( Y - W*H' ) + r(H)
% for dense tensor factorization
%   Y is the approapriate matrix unfolding
%   W is the khatri-rao product of the rest of the factors

[ ~, k ] = size(H);
G = ones(k,k); prod = [ 1:d-1, d+1:length(GG) ];
for dd = prod
    G = G .* GG{dd}; 
end
rho = trace(G)/k;
L = chol( G + (rho+ops.mu)*eye(k), 'lower' );

Hp = H;
tol = 1e-2;
for itr = 1:10
    H0 = H;
    
    Ht = L'\ ( L\ ( W'*(Yt+V) + rho*(H+U)' + ops.mu*Hp' ) );
    H  = proxr( Ht'-U, ops, d, rho);
    Yt = proxl( Y, W*Ht-V, ops );
    U  = U + H - Ht';
    V  = V + Yt - W*Ht;
    
    r = H - Ht';
    s = H - H0;
    if norm(r(:)) < tol*norm(H(:)) && norm(s(:)) < tol*norm(U(:))
        break
    end
end
GG{d} = H'*H;
end


function H = proxr( Hb, ops, d, rho )
    switch ops.constraint{d}
        case 'nonnegative'
            H = max( 0, Hb );
        case 'simplex_col'
            H = ProjectOntoSimplex(Hb, 1);
        case 'simplex_row'
            H = ProjectOntoSimplex(Hb', 1);
            H = H';
        case 'bias'
            H = max( 0, Hb );
            H(:,d) = 1;
        case 'l2'
            H = (rho/(ops.l2{d}+rho))*Hb;
    end
end

function Yt = proxl( Y, Yb, ops )
    switch ops.loss
        case 'KL'
            Yt = ( (Yb-1) + sqrt( (Yb-1).^2 + 4*Y ) )/2;
        case 'l1'
            Yt =  Y .* ( abs( Yb - Y ) <= 1 ) + ...
                 (Yb-1) .* ( Yb - Y > 1 ) + ...
                 (Yb+1) .* ( Yb - Y < -1);
        case 'Huber'
            Yt = (Yb+Y)/2 .* ( abs( Yb - Y ) <= 2*ops.Huber ) + ...
                 (Yb-ops.Huber) .* ( Yb - Y >  2*ops.Huber ) + ...
                 (Yb+ops.Huber) .* ( Yb - Y < -2*ops.Huber);
        case 'missing_ls'
            Yt = Yb;
            Yt( ~isnan(Y) ) = Y( ~isnan(Y) );
            Yt = .5 * ( Yt + Yb );
        case 'missing_KL'
            Yt = ( (Yb-1) + sqrt( (Yb-1).^2 + 4*Y ) )/2;
            Yt( isnan(Yt) ) = Yb( isnan(Yt) );
    end
end