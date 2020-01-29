function Ypred = test_V(V, Sim, X, Y, hit)

labelSet = unique(Y);
W = construct_W(V, Sim);
XW = X * W';
sz=size(XW);
Ypred=[];
for i=1:sz(1) 
    t=sort(XW(i,:),'descend'); 
    ok=t(1:hit); 
    [m,n]=find(XW(i,:)==ok(1));
    Ypred=[Ypred labelSet(n(1))];
    for j=1:hit
        [m,n]=find(XW(i,:)==ok(j));
        if labelSet(n(1))==Y(i)
          Ypred(end)=Y(i);
          break
        end
    end
end
Ypred=Ypred';    
%[~, Ypred] = max(XW, [], 2); %hit 1
%Ypred = labelSet(Ypred);
end
