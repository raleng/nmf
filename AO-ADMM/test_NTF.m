clear

n = 100;
k = 10;

Ht{1} = rand( n, k ).*( rand( n, k ) < .5 ); 
Ht{2} = rand( n, k ); 
Ht{3} = rand( n, k ); 

X = ktensor( Ht );
Y = double(X) + 1e-1*randn(n,n,n);
norm(tensor(Y)-tensor(X))/norm(X)

% ops = []; % default is NTFHinit = cell( 3, 1 );
for d = 1:3
    Hinit{d} = rand( n, k );
    Hinit{d} = Hinit{d} / diag( sqrt( sum( Hinit{d}.^2 ) ) );
    ops.l1{d} = 1;
    ops.l2{d} = 1e-1;
end
ops.init = Hinit;
ops.constraint{1} = 'l1';
ops.constraint{2} = 'l2';
ops.constraint{3} = 'l2';
[ H, his ] = AOadmm( Y, k, ops );

plot( his.time, his.err )