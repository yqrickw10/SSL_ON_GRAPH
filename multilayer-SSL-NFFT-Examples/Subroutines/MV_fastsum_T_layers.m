function y=MV_fastsum_T_layers(x,S)
%MV_FASTSUM_T_LAYERS Calculates the arithmetic mean of a cell of l NFFT-fastsum objects applied to a vector x
% 
% Input:  x: vector
%         S: cell of l NFFT-fastsum objects containing, i.a., the function handle applyNormalizedAdjacency applying the matrix D^(i)^(-1/2) W^(i) D^(i)^(-1/2) to x
% 
% Output: y: arithmetic mean of D^(i)^(-1/2) W^(i) D^(i)^(-1/2) applied to x
% 
% This software is distributed under the GNU General Public License v2. See 
% COPYING for the full license text. If that file is not available, see 
% <http://www.gnu.org/licenses/>.
%
% Copyright (c) 2019-2020 Kai Bergermann, Toni Volkmer

l=size(S,2);
sum=0;
for i=1:l
    sum=sum+S{i}.applyNormalizedAdjacency(x);
end
y=(1/l)*sum;
end
