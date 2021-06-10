function y=fh_pml(w,Ls_p_fct)
%FH_PML Calculates the arithmetic mean of a cell of size [1,T] applied to a vector w
% 
% Input:  w: vector
%         Ls_p_fct: function handle
% 
% Output: y: arithmetic mean of Ls_p_fct applied to w
% 
% This software is distributed under the GNU General Public License v2. See 
% COPYING for the full license text. If that file is not available, see 
% <http://www.gnu.org/licenses/>.
%
% Copyright (c) 2019-2020 Kai Bergermann, Toni Volkmer

T=size(Ls_p_fct,2);
y=0;
for i=1:T
    y=y+(1/T)*Ls_p_fct{i}(w);
end
end
