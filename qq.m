intcon=size(myexcelfile);
intcon=intcon(1,2);
f=myexcelfile{1,:};
f=-1*f';
A=myexcelfile{2,:};
b=[10];
lb = zeros(intcon,1);
ub = myexcelfile{3,:};
Aeq=zeros(1,intcon);
beq=[0];
x = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub)

