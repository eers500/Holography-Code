clear all
clc

syms x y z k
expression = exp(1i*k*sqrt(x^2+y^2+z^2))/sqrt(x^2+y^2+z^2);

h = diff(expression, z);
H = 1/(2*pi) * diff(h, z);

% vpa(subs(H, [x,y,z,k], [1,1,1,1]), 4)

%%
hh = matlabFunction(1/(2*pi)*h);
HH = matlabFunction(H);

e = hh(K,xx,yy,100);
E = HH(K,xx,yy,100);

subplot(1,2,1); imagesc(real(e)); colormap gray
subplot(1,2,2); imagesc(real(E)); colormap gray

%%
I = imread('/home/erick/Documents/PhD/Holography-Code/Code/MF1_30Hz_200us_awaysection.png');
IB = imread('/home/erick/Documents/PhD/Holography-Code/Code/MED_MF1_30Hz_200us_awaysection.png');
I = double(I);
IB = double(IB);


IB(IB==0)=mean(IB(:));
IN = I./IB;

N = 1.3236;
LAMBDA = 0.642;
FS = 0.711;
[NI,NJ]=size(IN);
SZ = 10;
Z = SZ*[0:149];
K = 2*pi*N/LAMBDA;

E = fftshift(fft2(IN-1));

%%
Q = complex(zeros(NI,NJ));
for i=1:NI
    for j=1:NJ
        Q(i,j) = ((LAMBDA*FS)/(max([NI,NJ])*N))^2*((i-NI/2)^2+(j-NJ/2)^2);
    end
end
Q = sqrt(1-Q)-1;

IZ = zeros(NI,NJ,length(Z));
for k=1:length(Z)
    R = exp(-1j*K*Z(k).*Q);
    IZ(:,:,k) = real(1+ifft2(ifftshift(E.*R)));
end

%%
for i=1:length(Z)
    imagesc(IZZ(:,:,i)); colormap gray; axis square;
    pause(0)
end