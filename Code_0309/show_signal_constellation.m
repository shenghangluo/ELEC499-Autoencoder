T = csvread('data_0309.csv');

%Real = T(:,1:208);
%Image = T(:, 209:416);

%Complex = complex(Real, Image);

one = T(:, 1:104)
one_real = one(:, 1:52);
one_image = one(:, 53:104),
one_complex = complex(one_real, one_image);

two = T(:, 105:208);
two_real = two(:, 1:52);
two_image = two(:, 53:104),
two_complex = complex(two_real, two_image);

%plot(one_complex(4*2+1:(2+1)*4), 'o')  %4*6+1:(6+1)*4
plot(two_complex(4*1+1:(1+1)*4), 'o')
ylabel('Imag')
xlabel('Real')
title('Signal Constellation for One Message')


