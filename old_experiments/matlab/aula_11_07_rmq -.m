clear;			% Limpa mem�ria da �rea de trabalho
clc;			% Limpa tela


%load 'F0707.dat'			% L� o arquivo de dados
%x = F0707(28:end,1);			% S�rie de entrada (independente)
%y = F0707(28:end,2);			% S�rie de sa�da (dependente)
%N = numel(y);				% Tamanho da serie

% Dados simulados
N = 300;
v = 0.01*randn(N,1); 		% Processo aleat�rio gaussiano para gerar ru�do
e = zeros(N,1);			% S�rie de ru�do contaminante
x = randn(N,1); 		% S�rie de entrada
y = zeros(N,1);			% S�rie de sa�da	
	

theta = [0;0];			% Valor inicial dos par�metros
P = 1e4*eye(2);			% Matriz de covari�ncia inicial
THETA = theta';			% Registro das estimativas
k = [0;0];			% Ganhos de corre��o
K = k';				% Registro dos ganhos
sigma = diag(P)';		% Vari�ncias
lambda = .95;			% Fator de esquecimento


% Simula N passos de um sistema din�mico e da predi��o recursiva
for t=2:N

    % Simula��o de din�mica	
    %e(t,1) = 0.2*e(t-1)+v(t);				% Simula��o do ru�do					
    %y(t,1) = 0.7*y(t-1)+0.5*x(t-1)+1*e(t);		% Simula��o da sa�da
    
    % Estimador de MQ recursivo
    phi = [y(t-1) x(t-1)]';				% Regressores
    k= P*phi/(phi'*P*phi+lambda);		  	% Atualiza��o dos ganhos
    theta = theta + k*(y(t)-phi'*theta);		% Atualiza��o dos par�metros
     
    P = (P - P*phi*phi'*P/(phi'*P*phi+lambda))/lambda;	% Atualiza��o da matriz de covari�ncia

    sigma(t,:) = diag(P)';				% Armazena as vari�ncias
    THETA(t,:) = theta';				% Armazena os par�metros
    K(t,:) = k';					% Armazena os ganhos
end

clf			% Limpa a figura
plot(THETA)		% Plota a evolu��o dos par�metros
