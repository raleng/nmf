function [ H, his ] = AOadmm( Y, k, ops )
% AO-ADMM for constrained matrix/tensor factorizations with least-squares loss
% input: Y, data; k, rank;
%        ops.loss          = 'ls'          -- default
%                          = 'l1'
%                          = 'Huber'   -- ops.Huber lambda
%                          = 'KL'
%                          = 'missing_ls'  -- default if Y has NaN's
%                           -NaN's in Y will be treated as missing.
%                           -if Y is stored as a sparse data structure,
%                           zeros will be treated as missing.
%        ops.constraint{d} = 'nonnegative' -- default
%                          = 'simplex_row'
%                          = 'simplex_col'
%                          = 'l1'      -- ops.l1{d}  = penalty
%                          = 'l1n'          l1 and nonnegative
%                          = 'l2'      -- ops.l2{d}  = penalty
%                          = 'l2n'          l2 and nonnegative
%                          = 'l2-bound'-- row norm less equal to 1
%                          = 'l2-boundn'    l2bound and nonnegative
%                          = 'bias'    -- let the d-th column being 1
%                                               + nonnegativity
%        ops.init          = H_init
%        ops.maxitr        = max number of iterations
%        ops.maxtime       = max time
%        ops.mu            = BSUM Tikhonov regularization parameter
%       for ls loss and nonnegative constraint, you can simply set ops=[]
% output: H, matrix cell of the factors
%        his.time          = time elapsed per iteration
%        his.err           = objective value per iteration
%        his.admm          = number of admm iterates per iteration
tic

dim = length(size(Y)); nn = size(Y);
H = cell( dim, 1 ); U = cell( dim, 1 );
for d = 1:dim
    H{d} =  rand( nn(d), k ); H{d} = H{d} / diag( sqrt( sum( H{d}.^2 ) ) );
    U{d} = zeros( nn(d), k );
end
if ~isfield(ops, 'loss'),       ops.loss = 'ls';                end
if ~isfield(ops, 'constraint'), ops.constraint = cell(dim,1);
    for d=1:dim, ops.constraint{d} = 'nonnegative'; end;        end
if  isfield(ops, 'init'),       H = ops.init;                   end
if ~isfield(ops, 'maxitr' ),    ops.maxitr = 200;               end
if ~isfield(ops, 'maxtime'),    ops.maxtime = 600;              end
if ~isfield(ops, 'tol'),        ops.tol = 1e-7;                 end
GG = cell( dim, 1 );
for d = 1:dim
    GG{d} = H{d}'*H{d};
end
ops
ops.mu = 0;

if ~strcmp( ops.loss, 'ls' ), ops.maxitr = round(ops.maxitr/2); end
his.time    = zeros(1,ops.maxitr);
his.err     = zeros(1,ops.maxitr);
his.admm    = zeros(dim,ops.maxitr);


if dim == 2
%% matrix case
    normY = norm(Y(~isnan(Y)),'fro');
    if strcmp( ops.loss, 'ls' )
        for itr = 1:ops.maxitr
           [ H{2}, U{2}, GG, i ] = ...
               lsADMMdense( Y , H{1}, H{2}, U{2}, 2, GG, ops);
           his.admm(2,itr) = i;
           [ H{1}, U{1}, GG, i ] = ...
               lsADMMdense( Y', H{2}, H{1}, U{1}, 1, GG, ops);
           his.admm(1,itr) = i;
           his.time(itr) = toc;
           his.err(itr) = Loss( Y, H, ops, normY, dim );

                if toc > ops.maxtime || ( itr>1 && ...
                        abs( his.err(itr) - his.err(itr-1) ) < ops.tol )
                    his.err(itr+1:end) = [];
                    his.time(itr+1:end) = [];
                    his.admm(:,itr+1:end) = [];
                    break; 
                end
        end
    
    else
        Yt = zeros( size(Y) ); V = zeros( size(Y) );
        Yt( ~isnan(Y) ) = Y( ~isnan(Y) );
        for itr = 1:2*ops.maxitr
        [ H{2}, U{2}, Yt, V, GG, i ] = ...
           glADMMdense( Y , H{1}, H{2}, U{2}, Yt , V , 2, GG, ops);
        his.admm(2,itr) = i;
        [ H{1}, U{1}, Yt, V, GG, i ] = ...
           glADMMdense( Y', H{2}, H{1}, U{1}, Yt', V', 1, GG, ops);
        Yt = Yt'; V = V';
        his.admm(1,itr) = i;
        his.time(itr) = toc;
        his.err(itr) = Loss( Y, H, ops, normY, dim );
%         ops.mu = 1e-7 + .1*his.err(itr);

            if toc > ops.maxtime || ( itr>1 && ...
                    abs( his.err(itr) - his.err(itr-1) ) < ops.tol )
                his.err(itr+1:end) = [];
                his.time(itr+1:end) = [];
                his.admm(:,itr+1:end) = [];
                break; 
            end
        end
    end
    
    
elseif ~isa( Y, 'sptensor' );
%% dense tensor case
    normY = norm(Y(:));
    Ym = matricize( Y );
    for itr = 1:ops.maxitr
        for d = 1:dim
            dc = [ 1:d-1, d+1:dim ];
            W = khatrirao( H{ dc }, 'r' );
            [ H{d}, U{d}, GG, i ] = ...
                lsADMMdense( Ym{d}, W, H{d}, U{d}, d, GG, ops);
            his.admm(d,itr) = i;
        end
        his.time(itr) = toc;
        his.err(itr) = Loss( Y, H, ops, normY, dim );
%         ops.mu = 1e-7 + .1*his.err(itr);

        if toc > ops.maxtime || ( itr>1 && ...
                    abs( his.err(itr) - his.err(itr-1) ) < ops.tol )
            his.err(itr+1:end) = [];
            his.time(itr+1:end) = [];
            his.admm(:,itr+1:end) = [];
            break; 
        end
    end
    
    if ~strcmp( ops.loss, 'ls' )
        Yt = Ym; V = cell( size(Ym) );
        for d = 1:dim, V{d} = zeros( size( Ym{d} ) );   end
        for itr = ops.maxitr+1:2*ops.maxitr
            for d = 1:dim
            dc = [ 1:d-1, d+1:dim ];
            W = khatrirao( H{ dc }, 'r' );
            [ H{d}, U{d}, Yt{d}, V{d}, GG, i ] = ...
               glADMMdense( Ym{d}, W, H{d}, U{d}, Yt{d}, V{d}, d, GG, ops);
            his.admm(d,itr) = i;
            end
            his.time(itr) = toc;
            his.err(itr) = Loss( Y, H, ops, normY, dim );
       
            if toc > ops.maxtime || ( itr>1 && ...
                    abs( his.err(itr) - his.err(itr-1) ) < ops.tol )
                his.err(itr+1:end) = [];
                his.time(itr+1:end) = [];
                his.admm(:,itr+1:end) = [];
                break; 
            end
        end
    end
    
    
    
    
elseif isa( Y, 'sptensor' )
%% sparse tensor case, using Kolda's tensor_toolbox, sptensor
    normY = norm(Y);
    for itr = 1:ops.maxitr
        for d = 1:dim
            [ H, U, GG, i ] = lsADMMsptensor( Y, H, U, d, GG, ops);
            his.admm(d,itr) = i;
        end
        his.time(itr) = toc;
        his.err(itr) = Loss( Y, H, ops, normY, dim );
        ops.mu = 1e-7 + .1*his.err(itr);

        if toc > ops.maxtime || ( itr>1 && ...
                    abs( his.err(itr) - his.err(itr-1) ) < ops.tol )
            his.err(itr+1:end) = [];
            his.time(itr+1:end) = [];
            his.admm(:,itr+1:end) = [];
            break; 
        end
    end
    
    % non-least-squares loss case to be added
end



end











function err = Loss( Y, H, ops, normY, dim )
    switch ops.loss
        case 'ls'
            if dim == 2
                err = norm( Y - H{1}*H{2}', 'fro' ) / normY;
            elseif ~isa( Y, 'sptensor' )
                err = normY^2 + norm( ktensor(H) )^2 ...
                    - 2*innerprod( tensor(Y), ktensor(H) );
                err = sqrt(err)/normY;
            elseif  isa( Y, 'sptensor' )
                err = normY^2 + norm( ktensor(H) )^2 ...
                    - 2*innerprod( Y, ktensor(H) );
                err = sqrt(err)/normY;
            end
        case 'missing_ls'
            Err = ( Y - H{1}*H{2}' );
            Err( isnan(Y) ) = 0;
            err = norm( Err, 'fro' ) / normY;
        case 'KL'
            if dim == 2
                Yt = H{1}*H{2}';
                Err = Y .* log( Y+eps ) - Y .* log( Yt+eps ) - Y + Yt;
            else
                Yt = max( double( ktensor(H) ), eps ); Err = Yt;
                KL = Y .* log( Y./Yt ) - Y + Yt;
                Err( Y~=0 ) = KL( Y~=0 );
            end
            err = sum( Err(:) );
        case 'missing_KL'
            Yt = H{1}*H{2}';
            Err = Y .* log( Y+eps ) - Y .* log( Yt+eps ) - Y + Yt;
            Err( isnan(Y) ) = 0;
            err = sum(Err(:));
        case 'l1'
            Err = abs( Y - double(ktensor(H)) );
            err = sum( Err(:) );
        case 'Huber'
            Err = Y - double(ktensor(H));
            Err = Err.^2 .* ( Err <= ops.Huber ) + ...
             ( ops.Huber*abs(Err) - ops.Huber^2/2 ) .* (Err > ops.Huber);
            err = sum( Err(:) );
    end
end
