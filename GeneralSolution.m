(*===================*)
(*  GeneralSolution  *)
(*===================*)

$ThisDirectory=If[NotebookDirectory[]==$Failed,Directory[],NotebookDirectory[],NotebookDirectory[]];

<<xAct`xPlain`;
Block[{Print=None},

DefConstantSymbol[Lambda0,PrintAs->"\!\(\*SubscriptBox[\[Lambda],\(0\)]\)"];
DefConstantSymbol[Lambda1,PrintAs->"\!\(\*SubscriptBox[\[Lambda],\(1\)]\)"];
DefConstantSymbol[Lambda2,PrintAs->"\!\(\*SubscriptBox[\[Lambda],\(2\)]\)"];
DefConstantSymbol[Lambda3,PrintAs->"\!\(\*SubscriptBox[\[Lambda],\(3\)]\)"];
DefConstantSymbol[Lambda4,PrintAs->"\!\(\*SubscriptBox[\[Lambda],\(4\)]\)"];
DefConstantSymbol[Lambda5,PrintAs->"\!\(\*SubscriptBox[\[Lambda],\(5\)]\)"];

DefConstantSymbol[c1,PrintAs->"\!\(\*SubscriptBox[\[ScriptC],\(1\)]\)"];
DefConstantSymbol[c2,PrintAs->"\!\(\*SubscriptBox[\[ScriptC],\(2\)]\)"];
DefConstantSymbol[c4,PrintAs->"\!\(\*SubscriptBox[\[ScriptC],\(4\)]\)"];
DefConstantSymbol[c5,PrintAs->"\!\(\*SubscriptBox[\[ScriptC],\(5\)]\)"];
DefConstantSymbol[cF,PrintAs->"\!\(\*SubscriptBox[\[ScriptCapitalF],\(1\)]\)"];
DefConstantSymbol[cFp,PrintAs->"\!\(\*SubscriptBox[\[ScriptCapitalF],\(2\)]\)"];

DefConstantSymbol[Operator0,PrintAs->"\!\(\*SubscriptBox[\[ScriptCapitalO],\(0\)]\)"];
DefConstantSymbol[Operator1,PrintAs->"\!\(\*SubscriptBox[\[ScriptCapitalO],\(1\)]\)"];
DefConstantSymbol[Operator2,PrintAs->"\!\(\*SubscriptBox[\[ScriptCapitalO],\(2\)]\)"];
DefConstantSymbol[Operator3,PrintAs->"\!\(\*SubscriptBox[\[ScriptCapitalO],\(3\)]\)"];
DefConstantSymbol[Operator4,PrintAs->"\!\(\*SubscriptBox[\[ScriptCapitalO],\(4\)]\)"];
DefConstantSymbol[Operator5,PrintAs->"\!\(\*SubscriptBox[\[ScriptCapitalO],\(5\)]\)"];

DefConstantSymbol[Dim,PrintAs->"\[ScriptCapitalD]"];
DefConstantSymbol[Spin,PrintAs->"\[ScriptS]"];
DefConstantSymbol[NInd,PrintAs->"\[ScriptN]"];
DefConstantSymbol[KInd,PrintAs->"\[ScriptK]"];

];

Title@"General solution";

Subsection@"Original basis used for our work";

Comment@"We first consider the basis.";
LongExpr=Lambda0*Operator0+Lambda1*Operator1+Lambda2*Operator2+Lambda3*Operator3+Lambda4*Operator4+Lambda5*Operator5;
LongExpr//DisplayExpression;

Alpha[KInd_,Spin_,Dim_]:=(-1)^KInd/Product[(Dim+2*(Spin-l-2)),{l,1,KInd}];

VanishingExpr[Spin_,NInd_,Dim_]:=(
	Alpha[NInd+1,Spin,Dim]*((Spin-2*NInd)*(Spin-2*NInd-1)*(Spin-2*NInd-2)/2)*(
		2*(Lambda1+NInd*Lambda5)
		+(Lambda2+2*NInd*Lambda4)*(Dim+2*(Spin-NInd-2))
	)
	+Alpha[NInd,Spin,Dim]*(
		Lambda0*(Spin-2*NInd)
		+Lambda1*(Spin-2*NInd)^2
		+Lambda2*(Spin-2*NInd)*Binomial[Spin-2*NInd,2]
		+(Lambda3+(Spin-2*NInd)*Lambda4)*NInd*(Spin-2*NInd)*(Dim+2*(Spin-NInd-1))
		+(
			Lambda4*(Spin-2*NInd)*(Spin-2*NInd-1)
			+Lambda5*(Spin-2*NInd+1+Spin-2*NInd)
		)*NInd*(Spin-2*NInd)
	)
	+Alpha[NInd-1,Spin,Dim]*(
		(Lambda3
		+(Spin-2*NInd)*Lambda4
		+Lambda5)*NInd*(Spin-2*NInd+2)
	)
);

ReduceSys[NList_,Spin_,Dim_]:=Module[{Solution,Coefficients},
	Coefficients={Lambda0,Lambda1,Lambda2,Lambda3,Lambda4,Lambda5};
	Solution=Table[VanishingExpr[Spin,NInd,Dim]==0,{NInd,NList}];
	Solution//=Quiet@Solve[#,Coefficients]&;
	Solution//=First;
	Solution//=FullSimplify;
Solution];

Comment@"Here is the general solution.";
Expr=ReduceSys[{0,1,2,3,4,5,6,7,8,9,10},Spin,Dim];
DisplayExpression/@Expr;
Export["GeneralSolutionEquations.csv",Expr];

Comment@"Here is the case of \"s=1\". Note that there are fewer operators than six in the case of \"s=1\" and \"s=2\".";
Expr=ReduceSys[{0},1,Dim];
DisplayExpression/@Expr;

Comment@"The specific case of \"s=3\".";
Expr=ReduceSys[{0,1},3,Dim];
DisplayExpression/@Expr;

Subsection@"Alternative Fronsdal-type basis";

Comment@"Here we define a basis, where the Fronsdal combination and the trace of Fronsdal have their own coefficients. This makes it manifest that these coefficients completely disappear from the system which you provided.";
ToFronsdalRules={Lambda0->cF,Lambda1->c1-cF,Lambda2->c2+cF,Lambda3->cFp,Lambda4->1/2(2c4+cFp),Lambda5->c5-cFp};
DisplayExpression/@ToFronsdalRules;

Comment@"Here is the transformation of the operator sum.";
LongExpr//=(#/.ToFronsdalRules)&;
LongExpr//=(#~Collect~{c1,c2,c4,c5,cF,cFp})&;
LongExpr//DisplayExpression;

NewReduceSys[NList_,Spin_,Dim_]:=Module[{Solution,Coefficients},
	Coefficients={c1,c2,c4,c5,cF,cFp};
	Solution=Table[(VanishingExpr[Spin,NInd,Dim]/.ToFronsdalRules)==0,{NInd,NList}];
	Solution//=Quiet@Solve[#,Coefficients]&;
	Solution//=First;
	Solution//=FullSimplify;
Solution];

Comment@"Here is the general solution. Notice how the Fronsdal-type coefficients completely disappear.";
Expr=NewReduceSys[{0,1,2,3,4,5,6,7,8,9,10},Spin,Dim];
DisplayExpression/@Expr;

Comment@"Here is the case of \"s=1\". Note that there are fewer operators than six in the case of \"s=1\" and \"s=2\".";
Expr=NewReduceSys[{0},1,Dim];
DisplayExpression/@Expr;

Comment@"The specific case of \"s=3\".";
Expr=NewReduceSys[{0,1},3,Dim];
DisplayExpression/@Expr;

Subsection@"System coefficients";

Comment@"Here are the coefficients of the system of equations for general spin and dimension.";
Expr=Module[{Expr},
	Expr=D[VanishingExpr[Spin,NInd,Dim],#];
	Expr//=FullSimplify;
Expr]&/@{Lambda0,Lambda1,Lambda2,Lambda3,Lambda4,Lambda5};
Expr//MatrixForm//DisplayExpression;
Export["GeneralSolutionCoefficients.csv",Expr];

(*Expr=Module[{Expr},
	Expr=D[VanishingExpr[Spin,NInd,Dim]/.ToFronsdalRules,#];
	Expr//=FullSimplify;
Expr]&/@{c1,c2,c4,c5,cF,cFp};
Expr//MatrixForm//DisplayExpression;
Export["GeneralSolutionCoefficients.csv",Expr];*)

Comment@"Here is a grid of discrete ranks.";
NList[Spin_]:=Table[NInd,{NInd,0,Floor[Spin/2]}];
GetLength[System_,Coefficients_]:=Module[{Solution},
	Solution=System;
	Solution//=Quiet@Solve[#,Coefficients]&;
	Solution//=First;
	Solution//=FullSimplify;
	Solution//=Length;
Solution];
GetLengthLimit[Spin_,Dim_,Var_]:=Module[{System,Coefficients},
	Coefficients={Lambda0,Lambda1,Lambda2,Lambda3,Lambda4,Lambda5};
	Check[
		Switch[Var,
			"Dim",
				System=Table[Limit[VanishingExpr[Spin,NInd,DimVar],
					DimVar->Dim]==0,
					{NInd,NList[Spin]}];
			,
			"Spin",
				System=Table[Limit[VanishingExpr[SpinVar,NInd,Dim],
					SpinVar->Spin]==0,
					{NInd,NList[Spin]}];
		];
		System//=(#~GetLength~Coefficients)&;
	,
		System=X;
	];
System];
RankOfSystem[Spin_,Dim_]:=Module[{OutExpr,Coefficients,System,SpinLim,NIndLim,DimLim},
	Coefficients={Lambda0,Lambda1,Lambda2,Lambda3,Lambda4,Lambda5};
	Check[
		System=Table[Module[{MyExpr,MySpin,MyNInd,MyDim},
				MyExpr=VanishingExpr[MySpin,MyNInd,MyDim];
				MyExpr//=(#/.MySpin->Spin)&;
				MyExpr//=Simplify;
				MyExpr//=(#/.MyNInd->NInd)&;
				MyExpr//=(#/.MyDim->Dim)&;
				MyExpr//=Simplify;
			MyExpr==0]
			,
			{NInd,NList[Spin]}];
		System//=(#~GetLength~Coefficients)&;
		OutExpr=System;
	,
		OutExpr=X;
	];
OutExpr];
Expr=Table[RankOfSystem[Spin,Dim],{Dim,0,7},{Spin,{0,1,2,3,4,5,6,7}}];
Expr//MatrixForm//DisplayExpression;
Export["GeneralSolutionRanks.csv",Expr];

Supercomment@"This is the end of the script.";

Quit[];
