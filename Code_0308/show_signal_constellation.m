T = csvread('data.csv');

Real = T(:,1:208);
Image = T(:, 209:416);

Complex = complex(Real, Image);



plot(Complex(4*6+1:(6+1)*4), 'o')  %4*6+1:(6+1)*4
ylabel('Imag')
xlabel('Real')
title('Signal Constellation for One Message')
