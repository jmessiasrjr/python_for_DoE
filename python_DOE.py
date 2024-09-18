########### ANOVA and Response Surface Method #############
########### Example from Montgomery problem 11-11 #########

import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools
from statsmodels.formula.api import ols
from pyDOE2 import *

############################################################

responseType = ['full'  , 'reduced'];   # full -> all variables; reduced -> variables with p-value > PR
modelType    = ['linear', 'quadratic'];

############# USER DEFINED #################################

rt = 1; #response model type 0-> full | 1-> reduced ANOVA for p < PR
mt = 1; #model type 0-> linear | 1-> quadratic
nr = 1; # number of replication, (1 for single computations), MUST insert all replicated values in y1 array
PR = 0.05; # max acceptable p-value
plan = bbdesign(3, center=3); # from pyDOE2 library, see pyDOE2 documentation

X  = np.tile(plan, (nr,1))

# Type the natural variable names as follow: x1, x2,... and so on.
# e.g.: x1 = [Central Point value, half range]
# if x1 -> [7.5, 10, 12.5], then x1 = [10, 2.5]
x1 = [175  , 25  ];  # Temperature -> A
x2 = [7.5  , 2.5 ];  # Rate        -> B
x3 = [20   , 5   ];  # Pressure    -> C

# if the design plan has more than 3 variable, add variables x4, x5, until fit it. 

measure = "y1"; 

y1  = np.array([535, 580, 596, 563, 645, 458, 350, 600, 595, 648, 532, 656, 653, 599, 620]); # this vector has to be the same length than the ((number of experiments on the DoE) * (replication)), and have to be aligned with plan's matrix order

############################################################

order=plan.shape[1]
alphabet = list(map(chr, range(65, 65+order)))
number   = list(str(s) for s in range(1,order+1))

powerSet=[]
for k in range(order):
    if (k == order-1 and mt == 0):
        break
    powerSet.extend(itertools.combinations(alphabet, k))
# for BB or CC designs (quadratic model):
    if (k == 1 and mt == 1):
        powerSet.extend(itertools.combinations(('I('+a+'**2)' for a in alphabet), k))

factors=[]
for j in range(len(powerSet)-1):
    factors.append("*".join(powerSet[j+1]))

df = pd.DataFrame( np.concatenate((X,y1.reshape(-1,1)),axis=1),columns=(alphabet+[measure]) )

equation = measure + ' ~ 1'
for f in factors:
    equation = " + ".join([equation, re.sub('\*(?![\*\d])', ':', f)])

results = ols(equation, data=df).fit()

ANOVA = sm.stats.anova_lm(results, typ=2)

print('ANOVA for all combinations: '+modelType[mt]+' model\n', ANOVA)
print('R2     = %8.4f ' % results.rsquared)
print('R2_adj = %8.4f ' % results.rsquared_adj)

F = ([s.strip('I()') for s in factors])
df.drop(columns=measure, inplace=True)

for i in range( len(F)-1, order-1, -1 ):
    df.insert(order, F[i], df.eval(F[i]), True)

p = np.array( ANOVA.iloc[:len(ANOVA)-1,3] )

factor = []
for i in range(0, p.size):
    if responseType[rt] == 'full':
        factor.append(i)
    else:
        if (p[i] < PR):
            factor.append(i)

Natural = F.copy()
for i in range(0, len(alphabet)):
    n = number[i]
    a = alphabet[i]
    Natural = list(map(lambda st: str.replace(st, a, '(' + str(eval(f'1./x{n}[1]')) + f'*{a}' + str(eval(f'-x{n}[0]/x{n}[1]')) + ')'), Natural))

# Calculate and print coded variables
Xmatrix = np.concatenate( (np.ones((X.shape[0],1)), df.to_numpy()[:,factor]), axis=1 )
b = np.linalg.lstsq(Xmatrix, y1, rcond=None)[0]

print('\nResponse surface (coded variables):   ', end='')

iterator = iter(b)
print('{:>10s}'.format(f'{next(iterator):+10.6f}'), end="")

for f in factor:
    print('{:>15s}'.format(f'{next(iterator):+10.6f}*'+F[f]), end="")

##### SymPy code to print response surface equation in natural (actual) variables ######

from sympy import Float, latex, postorder_traversal, preorder_traversal, simplify, symbols
from sympy.parsing.sympy_parser import parse_expr

symbols(','.join(alphabet))

iterator = iter(b)
expression = simplify(parse_expr( str(next(iterator)) + '+' + '+'.join([str(next(iterator)) + '*' + Natural[f] for f in factor])))

for f in preorder_traversal(expression):
    if isinstance(f, Float):
        expression = expression.subs(f, round(f, 6))

print("\nResponse surface (actual variables):  ",end='')

values=[str(s) for s in postorder_traversal(expression)][:-1]
nv = []
print('{:>10s}'.format(values[0]), end='')
for k in values:
    if any('*' + x in k for x in alphabet):
        if ('-' in k):
            nv.append(' '+k)
        else:
            nv.append('+'+k)

for f in F:
    for n in nv:
        if n.endswith(f):
            print('{:>15s}'.format(n), end='')
            break

###########  SymPy finish ########################

###########  recalculate reduced ANOVA, if selected --- rt = 1 #############

if responseType[rt] != 'full':
    print(f'\n\nANOVA for Reduced Model: {modelType[mt]} model with p < {PR:6.4f}')
    reduced = measure + ' ~ 1'
    for f in factor:
        reduced = " + ".join([reduced, re.sub('\*(?![\*\d])', ':', factors[f])])

    results = ols(reduced, data=df).fit()

    ANOVA=sm.stats.anova_lm(results, typ=2)
    print('\nANOVA:\n', ANOVA)
    print('R2     = %8.4f ' % results.rsquared)
    print('R2_adj = %8.4f ' % results.rsquared_adj)

print('\n###################################################\n')

########### Matplotlib plotting ####################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{amsmath}"
    ]),
})

def response(A, B, C, y):
    iterator = iter(y)
    value = next(iterator)
    for i in factor:
        value += next(iterator)*eval(Natural[i])
    return value

xline = np.linspace(x1[0] + np.min(plan)*x1[1], x1[0] + np.max(plan)*x1[1], 100); # first variable
yline = np.linspace(x2[0] + np.min(plan)*x2[1], x2[0] + np.max(plan)*x2[1], 100); # second variable

X, Y = np.meshgrid(xline, yline)

X2 = x3[0]*np.ones((X.shape[0], X.shape[1])); # third variable

zline = np.array(response(np.ravel(X), np.ravel(X2), np.ravel(Y), b))

Z = zline.reshape(X.shape)

f1=plt.figure(figsize=[6, 6])
ax = f1.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=8, cstride=8, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(zline), cmap=cm.coolwarm)
ax.set_xlabel('$X_{1}$')
ax.set_ylabel('$X_{2}$')
ax.set_zlabel('$X_{3}$')

f1.colorbar(surf, shrink=0.4, aspect=15)
ax.view_init(15, -135)

#### Save in files for LaTeX usage ####
plt.tight_layout()
plt.savefig("figure.pgf", bbox_inches='tight')
plt.savefig("figure.png", bbox_inches='tight')
##plt.show(); display on screen
