clear

n = 100;
k = 10;

Ht{1} = -log( rand( n, k ) ).*( rand( n, k ) < .5 ); 
Ht{2} = -log( rand( n, k ) ).*( rand( n, k ) < .5 ); 

X = Ht{1}*Ht{2}';
Y = double(X) + 1e-1*randn(n,n);
norm(Y-X,'fro')/norm(X,'fro')

ops = []; 
[ H, his ] = AOadmm( Y, k, ops );

plot( his.time, his.err )