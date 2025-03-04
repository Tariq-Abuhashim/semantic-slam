function loadObject(s)

t2mOpts = {'NumHeaderLines', 1         , ...
            'ReadMode'      , 'char'       };
        
[A,ffn,nh,SR,hl] = txt2mat(s, t2mOpts{:});

C = textscan(hl,'%s %f');
fprintf('Object is %s with score %f\n', C{1}{1}, C{2});

C = textscan(A,'%f %f %f');

% off = (C{3}>-78);
% C{3}=C{3}(off==0);
% C{1}=C{1}(off==0);
% C{2}=C{2}(off==0);

% off = (C{1}>-50|C{1}<-55);
% C{3}=C{3}(off==0);
% C{1}=C{1}(off==0);
% C{2}=C{2}(off==0);

plot3(C{3}, -C{1}, -C{2}, '.', 'markersize', 4);
axis tight
axis equal
axis off