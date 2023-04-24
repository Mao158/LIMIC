function C=SODW(X,W)
% function C=SODWkqw(X,W)
%
% 
% Returns the sum of all weighted outer products
%
% C=\sum_{i,j} (X(:,i)-X(:,j))(X(:,i)-X(:,j))'*W(i,j)


Q=W+W';
ii=find(sum(Q~=0));
Q=Q(ii,ii);
X=X(:,ii);
C=-(X*(Q-spdiags(sum(Q)',0,length(Q),length(Q))))*X';





