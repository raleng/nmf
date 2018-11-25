function [ H, U, GG, itr ] = lsADMMdense( Y, W, H, U, d, GG, ops)
% ADMM iterates to solve
%       minimize (1/2)*|| Y - W*H' ||^2 + r(H)
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

F = W'*Y; Hp = H;
tol = 1e-2;
for itr = 1:5
    H0 = H;
    
    Ht = L'\ ( L\ ( F + rho*(H+U)' + ops.mu*Hp' ) );
    H  = proxr( Ht'-U, ops, d, rho);
    U  = U + H - Ht';
    
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