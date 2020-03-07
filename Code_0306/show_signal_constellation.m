T = csvread('data.csv');

Real = T(:,1:52);
Image = T(:, 53:104);

Complex = complex(Real, Image);



plot(Complex(41:44), 'o')
ylabel('Imag')
xlabel('Real')
title('Signal Constellation for One Message')
