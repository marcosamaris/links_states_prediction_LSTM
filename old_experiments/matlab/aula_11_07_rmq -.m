clear;			% Limpa memória da área de trabalho
clc;			% Limpa tela


%load 'F0707.dat'			% Lê o arquivo de dados
%x = F0707(28:end,1);			% Série de entrada (independente)
%y = F0707(28:end,2);			% Série de saída (dependente)
%N = numel(y);				% Tamanho da serie

% Dados simulados
N = 300;
v = 0.01*randn(N,1); 		% Processo aleatório gaussiano para gerar ruído
e = zeros(N,1);			% Série de ruído contaminante
x = randn(N,1); 		% Série de entrada
y = zeros(N,1);			% Série de saída	
	

theta = [0;0];			% Valor inicial dos parâmetros
P = 1e4*eye(2);			% Matriz de covariância inicial
THETA = theta';			% Registro das estimativas
k = [0;0];			% Ganhos de correção
K = k';				% Registro dos ganhos
sigma = diag(P)';		% Variâncias
lambda = .95;			% Fator de esquecimento


% Simula N passos de um sistema dinâmico e da predição recursiva
for t=2:N

    % Simulação de dinâmica	
    %e(t,1) = 0.2*e(t-1)+v(t);				% Simulação do ruído					
    %y(t,1) = 0.7*y(t-1)+0.5*x(t-1)+1*e(t);		% Simulação da saída
    
    % Estimador de MQ recursivo
    phi = [y(t-1) x(t-1)]';				% Regressores
    k= P*phi/(phi'*P*phi+lambda);		  	% Atualização dos ganhos
    theta = theta + k*(y(t)-phi'*theta);		% Atualização dos parâmetros
     
    P = (P - P*phi*phi'*P/(phi'*P*phi+lambda))/lambda;	% Atualização da matriz de covariância

    sigma(t,:) = diag(P)';				% Armazena as variâncias
    THETA(t,:) = theta';				% Armazena os parâmetros
    K(t,:) = k';					% Armazena os ganhos
end

clf			% Limpa a figura
plot(THETA)		% Plota a evolução dos parâmetros
