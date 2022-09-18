close all;
clear all;

pkg load signal

printf ("%s", program_name ());
arg_list = argv ();
for i = 1:nargin
  printf (" %s", arg_list{i});
endfor
printf ("\n");

fprintf("Calculating fir coeffs for low pass...\n")
f1 = input("Please enter the start frequency of the stop band (-3dB): ") %2000;
f2 = input("Please enter the stop frequency of the stop band (-40dB): ") %3117;
delta_f = f2-f1;
Fs = input("Please enter the sample frequency: ") %49880;
dB  = 40;
N = dB*Fs/(22*delta_f);
N = 80%N + mod(N,2)


f =  [f1 ]/(Fs/2);
hc = fir1(round(N)-1, f,'low');
%hc = hc(:) .* 8
save firCoeffsLpf.txt hc

figure
subplot(5,1,1)
plot((-0.5:1/4096:0.5-1/4096)*Fs,20*log10(abs(fftshift(fft(hc,4096)))))
axis([0 20000 -60 20])
title('Filter Frequency Response')
grid on

x = sin(2*pi*[1:1000]*600/Fs) +  sin(2*pi*[1:1000]*2000/Fs) + sin(2*pi*[1:1000]*5000/Fs)  + sin(2*pi*[1:1000]*13000/Fs);

sig = 20*log10(abs(fftshift(fft(x,4096))));
xf = filter(hc,1,x);

subplot(5,1,2)
plot(x)
title('Sinusoid with frequency components 600, 2000, 5000, and 13000 Hz')


subplot(5,1,3)
plot(xf)
title('Filtered Signal')
xlabel('time')
ylabel('amplitude')


x= (x/sum(x))/20;
sig = 20*log10(abs(fftshift(fft(x,4096))));
xf = filter(hc,1,x);

subplot(5,1,4)
plot((-0.5:1/4096:0.5-1/4096)*Fs,sig)
hold on
plot((-0.5:1/4096:0.5-1/4096)*Fs,20*log10(abs(fftshift(fft(hc,4096)))),'color','r')
hold off
axis([0 20000 -60 10])
title('Input to filter - 4 Sinusoids')
grid on
subplot(5,1,5)
plot((-0.5:1/4096:0.5-1/4096)*Fs,20*log10(abs(fftshift(fft(xf,4096)))))
axis([0 20000 -60 10])
title('Output from filter')
xlabel('Hz')
ylabel('dB')
grid on
