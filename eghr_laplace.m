
%
% Demo of 2-dimensional ICA by the EGHR
%
% Copyright (C) 2016 Takuya Isomura
% (RIKEN Brain Science Institute)
%
% 2016-04-24
%

% Simulation parameters
%--------------------------------------------------------------------------
dim_s = 2;       % dimension of source
dim   = 2;       % dimension of input & output
T     = 2*10^6;  % total training steps
tau_W = 1*10^3;  % learning rate
samp  = 1000;    % #samples for testing

A     = randn(2); % mixing matrix
E0    = dim*1+1; % EGHR parameter (for exponential distribution)
sqrt2 = sqrt(2);

% Initialization
%--------------------------------------------------------------------------
s      = zeros(dim_s,1);     % input
W      = eye(dim,dim) * 1.5; % EGHR unmixing matrix
E0_Egx = eye(dim,dim) * 0;   % expectation

% Training
%--------------------------------------------------------------------------
for t = 1 : T
    %source: White Laplase noise (Langevin equation)
    s = sqrt2 * -log(rand(dim_s,1)) .* (2*randi(2,dim_s,1)-3) ./ 2;
    x = A * s; %input

    % EGHR
    u = W * x; % output
    g = sqrt2 * tanh(u*100); % nonlinearization (approximated sign function)
    E = u' * g; % prediction error
    E0_Egx = E0_Egx + 10/tau_W * (-E0_Egx + (E0 - E) * g * x'); % averaging
    W = W + 1/tau_W * E0_Egx; % update
end

% Testing
%--------------------------------------------------------------------------
W * A % Showing final state

% Testing samples
S = sqrt2 * -log(rand(dim_s,samp)) .* (2*randi(2,dim_s,samp)-3) ./ 2; %sources
X = A * S; % input
U = W * X; % output

% Show distributions
a = 5; % scale
figure(1); clf;
subplot(1,3,1); plot(S(1,:), S(2,:), 'x'); axis([-a a -a a]); axis square; title('Source');
subplot(1,3,2); plot(X(1,:), X(2,:), 'x'); axis([-a a -a a]); axis square; title('Input');
subplot(1,3,3); plot(U(1,:), U(2,:), 'x'); axis([-a a -a a]); axis square; title('Output');
