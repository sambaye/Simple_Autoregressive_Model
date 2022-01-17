import sys
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from numpy import *
import numpy.linalg as la


def Ar1Generate(dmu0,mu0,sigma,frames,atoms,alpha):
 Mu = zeros((frames,atoms,1))
 Dmu = zeros((frames,atoms,1))
 Mu[0] = mu0
 Dmu[0] = dmu0
 for i in range(frames-1):
   dmu =  alpha*dmu0 + sigma*random.standard_normal((atoms,1))
   Mu[i+1] = mu0 + dmu 
   Dmu[i] = dmu 
   mu0 = Mu[i+1]
   dmu0 = dmu
 return Mu,Dmu

def Ar2Generate(dmu0,dmu1,mu0,sigma,frames,atoms,alpha,beta):
 Mu = zeros((frames,atoms,1))
 Dmu = zeros((frames,atoms,1))
 Mu[0] = mu0
 Dmu[0] = dmu0
 Dmu[1] = dmu1
 for i in range(frames-1):
   dmu = alpha*dmu1 + beta*dmu0 + sigma*random.standard_normal((atoms,1))
   Mu[i+1] = mu0 + dmu 
   Dmu[i] = dmu
   mu0 = Mu[i+1]
   dmu0 = dmu1
   dmu1 = dmu
 return Mu,Dmu 

def AR1fit(Mu):
 Mudiff = diff(Mu,n=1,axis=0)
 D2 = 0.0
 DF = 0.0
 for j in range(len(Mudiff[0])):
  for i in range(len(Mudiff)):
   D2 += dot(Mudiff[i-1,j],Mudiff[i-1,j])
   DF += dot(Mudiff[i,j],Mudiff[i-1,j])
 afit = DF/D2
 return afit,Mudiff

def AR2fit(Mu):
 Mudiff = diff(Mu,n=1,axis=0)
 D2 = zeros((2,2))
 DF = zeros((2))
 for j in range(len(Mudiff[0])):
  for i in range(len(Mudiff)):
   D2[0,0] += dot(Mudiff[i-1,j],Mudiff[i-1,j])
   D2[0,1] += dot(Mudiff[i-2,j],Mudiff[i-1,j])
   D2[1,0] += dot(Mudiff[i-1,j],Mudiff[i-2,j])
   D2[1,1] += dot(Mudiff[i-2,j],Mudiff[i-2,j])
   DF[0] += dot(Mudiff[i,j],Mudiff[i-1,j])
   DF[1] += dot(Mudiff[i,j],Mudiff[i-2,j])
 afit = la.solve(D2,transpose(DF))
 return afit,Mudiff

def Ar1Residual(afit,Mudiff,Mu):
 var = 0.0
 DmuFit = zeros((len(Mudiff),len(Mudiff[0]),3))
 a = 0.0
 b = 0.0
 for i in range(len(Mudiff)):
  for j in range(len(Mudiff[0])):
   DmuFit[i,j] = (afit*Mudiff[i-1,j])
   a += sum(Mudiff[i,j]**2)
   b += afit*afit*sum(Mudiff[i-1,j]**2)
 var = (a - b)/(len(Mudiff)*len(Mudiff[0]))
 return DmuFit,var**0.5

def Ar2Residual(afit,bfit,Mudiff,Mu):
 var = 0.0
 a = 0.0
 b = 0.0
 c = 0.0
 d = 0.0
 DmuFit = zeros((len(Mudiff),len(Mudiff[0]),3))
 for i in range(len(Mudiff)):
  for j in range(len(Mudiff[0])):
   DmuFit[i,j] = (afit*Mudiff[i-1,j] + bfit*Mudiff[i-2,j])
   a += sum(Mudiff[i,j]**2)
   b += afit*afit*sum(Mudiff[i-1,j]**2)
   c += bfit*bfit*sum(Mudiff[i-2,j]**2)
   d += 2*afit*bfit*sum((Mudiff[i-1,j]*Mudiff[i-2,j]))
 var = (a - b - c -d)/(len(Mudiff)*len(Mudiff[0]))
 return DmuFit,var**0.5


def makeSigma(Mudiff,lag):
    Mudiff = diff(Mudiff,n=1,axis=0)
    #Mudiff = Mudiff[:,:,0]
    Sigma = zeros((lag+1,lag+1))
    for a in range(lag+1): ## loop over lags
     for b in range(lag+1):
      for i in range(len(Mudiff)): ## loop over frames
       for j in range(len(Mudiff[0])): ## loop over atoms
         Sigma[a,b] += dot(Mudiff[i-(a),j],Mudiff[i-(b),j])
         #/( (sum(Mudiff[i-(a),j,:]**2)**0.5)*(sum(Mudiff[i-(b),j,:]**2)**0.5)   )
    return Sigma

def makeVar(Sigma,lag):
    K = Sigma[:lag,:lag]
    k = Sigma[lag,:lag]
    k0 = Sigma[lag,lag]
    E = dot(la.inv(K),transpose(k))
    var = k0 - dot(k,dot(la.inv(K),transpose(k)))
    return E,var**0.5

def plotRes(binsPlot,histFit1,histFit2,tag):
    plt.plot(binsPlot[1:],log(histFit1),label=tag+"AR_1")
    plt.plot(binsPlot[1:],log(histFit2),label=tag+"AR_2")
    plt.xlabel("Residual")
    plt.ylabel("log(P(Residual))")
    plt.legend(loc="best")
    plt.savefig(tag+"auto_regress.png")
    plt.clf()
    

def main(argv):
 
 Kb = 0.0081345
 frames = 1000;
 atoms = 1000;   
 histSize = 30;

 atoms = 1
 mu0 = random.standard_normal((atoms,1))
 dmu0 = random.standard_normal((atoms,1))
 dmu1 = random.standard_normal((atoms,1))
 alpha = 0.25
 beta = 0.40
 sigma = 0.35
 T = 300
 beta = 1/(Kb*T)
 
## Generated Test Data For Autoregressive fit.
 MuA1,DmuA1 = Ar1Generate(dmu0,mu0,sigma,frames,atoms,alpha)
 MuA2,DmuA2 = Ar2Generate(dmu0,dmu1,mu0,sigma,frames,atoms,alpha,beta)
 SigmaA11 = makeSigma(MuA1,1)/((len(MuA1)-1)*(len(MuA1[0])))
 SigmaA12 = makeSigma(MuA1,2)/((len(MuA1)-1)*(len(MuA1[0])))
 SigmaA21 = makeSigma(MuA2,1)/((len(MuA2)-1)*(len(MuA2[0])))
 SigmaA22 = makeSigma(MuA2,2)/((len(MuA2)-1)*(len(MuA2[0])))

 # Generate 
 muA11,sigA11   = makeVar(SigmaA11,1)
 resA11 = random.standard_normal(frames)*sigA11
 muA12,sigA12 = makeVar(SigmaA12,2)
 resA12 = random.standard_normal(frames)*sigA12
 muA21,sigA21   = makeVar(SigmaA21,1)
 resA21 = random.standard_normal(frames)*sigA21
 muA22,sigA22 = makeVar(SigmaA22,2)
 resA22 = random.standard_normal(frames)*sigA22

 ## AR1 Compare Plots
 binsPlot = linspace(-1.0,1.0,30)
 histA11,xfitA11   = histogram(resA11,bins=binsPlot,density=True)  
 histA12,xfitA12   = histogram(resA12,bins=binsPlot,density=True)
 histA21,xfitA21   = histogram(resA21,bins=binsPlot,density=True)  
 histA22,xfitA22   = histogram(resA22,bins=binsPlot,density=True)


 plotRes(binsPlot,histA11,histA12,"AR_1_")
 plt.clf()
 plotRes(binsPlot,histA21,histA22,"AR_2_")
 
if __name__ == "__main__":
    main(sys.argv)
