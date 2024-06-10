# return the printout and the number of ways to get the result 
# # from the number using the 4 basic arithmetic operations

num = 3
result = 36
MaxOp = 4

def __add__(a,b):
    return a+b
def __sub__(a,b):
    return a-b
def __mul__(a,b):
    return a*b
def __truediv__(a,b):
    return a/b
    
MathOps = ['+','-','*','/']
MathOpsFunc = {'+':__add__, '-':__sub__, '*':__mul__, '/':__truediv__}

def ArithmeticOps(num,result,currentResult, currentString, OpsCount, GoodResults):
    if OpsCount == MaxOp:
        return 
    for Op in MathOps:
        if MathOpsFunc[Op](currentResult,num) == result:
            GoodResults.append(currentString + (f"{Op}{num}={result}"))
        else:
            ArithmeticOps(num, result, MathOpsFunc[Op](currentResult,num),currentString + (f"{Op}{num}"), OpsCount+1, GoodResults)
    return 

GoodResults = []
ArithmeticOps(num,result, num, str(num), 0, GoodResults)

for i in range(len(GoodResults)):
    print(GoodResults[i])
print(len(GoodResults))