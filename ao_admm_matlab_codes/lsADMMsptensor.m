function [ H, U, GG, itr ] = lsADMMsptensor( Y, H, U, d, GG, ops)
% ADMM iterates to solve
%       minimize (1/2)*|| Y - ktensor(H) ||^2 + r(H{d})
% for sparse tensor factorization
%   Y is the sptensor
%   H is a cell of the matrix factors
%   d is the index of the matrix factor to be updated

[ ~, k ] = size(H{d});
G = ones(k,k); prod = [ 1:d-1, d+1:length(GG) ];
for dd = prod
    G = G .* GG{dd}; 
end
rho = min(1e-3,trace(G)/k);
L = chol( G + (rho+ops.mu)*eye(k), 'lower' );

F = mttkrp( Y, H, d );
tol = 1e-2;
Hd = H{d}; Ud = U{d};
for itr = 1:5
    H0 = Hd;
    
    Ht   = L'\ ( L\ ( F + rho*(Hd+Ud) + ops.mu*H{d} )' );
    Hd = proxr( Ht'-Ud, ops, d, rho);
    Ud = Ud + Hd - Ht';
    
    r = Hd - Ht';
    s = Hd - H0;
    if norm(r(:)) < tol*norm(Hd(:)) && norm(s(:)) < tol*norm(Ud(:))
        break
    end
end
H{d} = Hd;
U{d} = Ud;
GG{d} = Hd'*Hd;
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
        case 'l1'
            H = sign( Hb ) .* max( 0, abs(Hb) - (ops.l1{d}/rho) );
        case 'l1n'
            H = max( 0, Hb - ops.l1{d}/rho );
        case 'l2'
            H = ( rho/(ops.l2{d}+rho) ) * Hb;
        case 'l2n'
            H = ( rho/(ops.l2{d}+rho) ) * max(0,Hb);
        case 'l2-bound'
           nn = sqrt( sum( Hb.^2 ) );
            H = Hb * diag( 1./ max(1,nn) );
        case 'l2-boundn'
            H = max( 0, Hb );
           nn = sqrt( sum( H.^2 ) );
            H = H * diag( 1./ max(1,nn) );
        case 'l0'
            T = sort(Hb,2,'descend');
            t = T(:,4); T = repmat(t,1,size(T,2));
            H = Hb .* ( Hb >= T );
    end
end