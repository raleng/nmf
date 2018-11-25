function Ym = matricize( Y )

Y = tensor(Y);
Ym = cell( ndims(Y), 1 );
for d = 1:ndims(Y)
    temp  = tenmat(Y,d);
    Ym{d} = temp.data';
end