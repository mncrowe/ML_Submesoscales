function F_avg = Data_Average(F,t,range,window,method)
% Time averages a given input over a given window
%
% - F: field to average, t should correspond to the final dimension of F
% - t: time vector, set t = 0 to get t = 1:Nt (default: 0)
% - range: vector [i1 i2], indices of time points where average is calculated (default: [1 Nt])
% - window: vector [w1 w2], window to average over, relative to each index (default: [-1 1])
% - method: averaging method;
%   - 0: central Riemann sum/trapezoidal rule (default)
%   - 1: left Riemann sum
%   - 2: right Riemann sum

S = size(F); Nt = S(end);

if nargin < 2; t = 1:Nt; end
if nargin < 3; range = [1 Nt]; end
if nargin < 4; window = [-1 1]; end
if nargin < 5; method = 0; end

if t == 0; t = 1:Nt; end
if range == 0, range = [1 Nt]; end
if window == 0; window = [-1 1]; end

Nr = range(2)-range(1)+1;
P_dims = prod(S(1:end-1));

F = reshape(F,[P_dims Nt]);
F_avg = zeros([P_dims Nr]);

for i = range(1):range(2)
    j1 = int32(max(1,i+window(1)));
    j2 = int32(min(Nt,i+window(2)));

    dt = reshape(t(j1+1:j2)-t(j1:j2-1),[1 j2-j1]);
    T = t(j2)-t(j1);

    if method == 0; M = ([dt 0]+[0 dt])/2; end
    if method == 1; M = [dt 0]; end
    if method == 2; M = [0 dt]; end

    F_avg(:,i-range(1)+1) = sum(F(:,j1:j2).*M,2)/T;
end

F_avg = reshape(F_avg,[S(1:end-1) Nr]);

end