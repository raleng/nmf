function H = ProjectOntoSimplex(Hb, b)
% PROJECTONTOSIMPLEX Projects point onto simplex of specified radius.
%
% w = ProjectOntoSimplex(v, b) returns the vector w which is the solution
%   to the following constrained minimization problem:
%
%    min   ||w - v||_2
%    s.t.  sum(w) == b, w >= 0.
%
%   That is, performs Euclidean projection of v to the positive simplex of
%   radius b.
%
% Author: John Duchi (jduchi@cs.berkeley.edu)
% updated by Kejun Huang to handle multiple vectors

[n,k] = size(Hb);
if (b < 0)
  error('Radius of simplex is negative: %2.3f\n', b);
end
Hb = (Hb > 0) .* Hb;
Hsort = sort(Hb,'descend');
Hsum = cumsum(Hsort);
j = sum( Hsort > (Hsum - b) ./ cumsum( ones( size(Hsum) ) ) );

theta = zeros(1,k);
for kk = 1:k
    if j(kk)>0, theta(kk) = Hsum(j(kk),kk); end
end
theta = ( theta - b )./j;
% theta = max(0, ( theta - b )./j );
H = max(Hb - ones(n,1)*theta, 0);