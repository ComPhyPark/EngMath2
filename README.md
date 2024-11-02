# Lecture 1: Introduction
### Probability space
- sample space $\Omega$, 랜덤 프로세스의 모든 결과의 집합
- $F$: 사건(outcome을 모은것)의 집합, 사건은 $\Omega$의 부분집합
- $Pr: F\rightarrow \mathbb{R}$ 사건의 확률을 반환

### Pr이 만족해야 하는 것(Axiom)
- $0\leq Pr(E)\leq1$
- $Pr(\Omega)=1$
- $Pr\left(\bigcup E_i\right)=\sum Pr(E_i)$, mutually disjoint인 $E_i$들에 대해

### Lemma: Union Bound
$$Pr\left(\bigcup A_i\right)\leq\sum Pr(A_i)$$

### Lemma: Inclusion-Exclusion Principle
$$Pr\left(\bigcup_{i=1}^nE_i\right)=\sum_{k=1}^n\left((-1)^{k-1}\sum_{I\subseteq\{1,\dots,n\},|I|=k}Pr\left(\bigcap_{i\in I}E_i\right)\right)$$

### 이산 분포가 균일할 때
$$Pr(A)=\frac{|A|}{|\Omega|}$$

### 사건의 독립
$E$와 $F$가 독립임은
$$Pr(E\cap F)=Pr(E)Pr(F)$$

$E_1,\dots,E_k$가 mutually independent
$$Pr\left(\bigcap_{i\in I}E_i\right)=\prod_{i\in I}Pr(E_i)$$
for all $I$

### 다항식 검증 알고리즘 예시
두 다항식 같은지 판별, 교점은 많아야 $d$개  
크기가 $100d$인 set에서 하나를 뽑아 대입하면 다항식이 다른데 결과가 같을 확률은 커야 $1/100$

# Lecture 2: Events and Probability
### Conditional Probability
$Pr(F)>0$인 경우에 $F$에 대한 $E$의 조건부 확률은
$$Pr(E|F)=\frac{Pr(E\cap F)}{Pr(F)}$$

즉, $Pr(F)>0$이면 $E$와 $F$가 독립 iff $Pr(E|F)=Pr(E)$

### Verifying Matrix Multiplication Example
$AB=C$인지 확인, 벡터 $r$에 대해 $ABr=Cr$인지 확인($r$의 모든 원소는 1 or 0)  
$AB\neq C$이면 $Pr(ABr=Cr)\leq\frac12$  
pf: $AB-C$에서 0 아닌 원소 있는 열을 $i$행이라 하면, $r_i=0,1$ 중 많아야 하나만 $ABr=Cr$  
즉 $k$종류의 $r_i$에 대해 해보면 실패할 확률은 $2^{-k}$보다 작다

### Law of Total Probability
$E_1,\dots,E_n$이 mutually disjoint이고 $\bigcup_{i}E_i=\Omega$라면,
$$Pr(B)=\sum_i Pr(B\cap E_i)=\sum_i Pr(B|E_i)Pr(E_i)$$

### Bayes' Law
$E_1,\dots,E_n$이 mutually disjoint이고 $\bigcup_{i}E_i=\Omega$라면,
$$Pr(E_j|B)=\frac{Pr(E_j\cap B)}{Pr(B)}=\frac{Pr(B|E_j)Pr(E_j)}{\sum_{i=1}^n Pr(B|E_i)Pr(E_i)}$$

# Lecture 3: Events and Probability
### Conditional Independence
두 사건 $E,F$는 conditionally independent given $C$
$$Pr(E\cap F|C)=Pr(E|C)Pr(F|C)$$

### Naive Bayes' Classifier Example
어떤 $D$를 $c_1,\dots,c_n$ 중 하나로 분류하기  
$D$는 feature vector $x=(x_1,\dots,x_m)$으로 표현됨(ex. 특정 단어의 등장횟수)
$$
\begin{aligned}
Pr(c_j|x)&=\frac{Pr(x|c_j)Pr(c_j)}{Pr(x)}\\
&=\frac{\prod_{k=1}^mPr(x_k|c_j)Pr(c_j)}{Pr(x)}
\end{aligned}
$$
분모는 일정하므로, 분자를 최대화하는 $c_j$를 찾자  
Class Prior: $Pr(c_j)$는 학습 데이터 중 $c_j$의 비율로 추정  
Feature Likelihood: $Pr(x_k|c_j)$는 학습 데이터의 $c_j$중 $x_k$의 비율로 추정

### Simple Graph
$G=(V,E)$
- $V$: 공집합이 아닌 집합, 정점들
- $E$: 크기 2인 $V$의 부분집합으로 구성, 간선들

$deg(v)$: $v$에 인접한 간선의 개수, degree

Directed Graph: $E$의 원소가 집합이 아닌 순서쌍$(E\in V^2)$  
loop($u\rightarrow u$)가 없으면 simple directed Graph  
indegree: 들어오는 간선 개수, 0이면 source  
outdegree: 나가는 간선 개수, 0이면 sink  

### Common Graphs
완전그래프 $K_n$: 정점 $n$개, 모든 정점 쌍 사이에 간선/총 $\binom n2$개  
이분그래프 $G=(U,V,E)$: 정점들이 $U$와 $V$로 나누어지며, 모든 간선은 $U$의 정점과 $V$의 정점을 이음  
Independent set: 정점의 부분집합, 내부에 이웃한 정점 없어야함

### Handshaking Lemma
$$\sum_{v\in V}deg(v)=2|E|$$

### Cut
cut $S$: $V$를 $S$와 $V-S$로 분할하는것$(S\subset V, S\neq\emptyset)$  
$S$의 cut-set: $\{(u,v)\in E|u\in S, v\not\in S\}$  
cut의 size/weight: cut-set의 size/weight의 총합

### Randomized Min-Cut
Minimum Cut: cut 중 비용 최소인 것 찾기  
Edge Contraction: 간선을 고르고, 양 끝 정점을 합치기
1. 무작위로 아무 edge를 고르기
2. 고른 edge로 contract하기
3. 정점이 2개 남을 때까지 1~2 반복

어떤 min cut-set을 내놓을 확률이 최소 $\frac2{n(n-1)}$  
min cut-set의 size를 $k$라 두자.  
사건 $E_i$: $i$번째 iteration에서 $k$를 제외한 것을 뽑을확률  
사건 $F_i$: $\bigcap_{j=1}^iE_i$  

그런데, min cut-set 크기 $k$이므로 모든 정점의 degree는 $k$이상->$|E|\geq kn/2$  
$$Pr(E_1)=\frac{|E|-|C|}{|E|}\geq1-\frac k{nk/2}=1-\frac2n$$
비슷한 방식으로
$$Pr(E_i|F_{i-1})\geq1-\frac2{n-i+1}$$
모두 곱하면
$$Pr(F_{n-2})\geq\prod_{i=1}^{n-2}\left(\frac{n-i-1}{n-i+1}\right)=\frac2{n(n-1)}$$

$m$번 해서 최소를 리턴한다면 실패할 확률은 커야
$$\left(1-\frac2{n(n-1)}\right)^m\leq\exp\left(-\frac{2m}{n(n-1)}\right)$$
$m=n(n-1)\ln n$으로 잡으면 $1/n^2$보다 작아짐

# Lecture 4: Random Variable, Expectation
### Random Variable
real-valued function on $\Omega$  
$X: \Omega\rightarrow\mathbb{R}$

### Independence
$X, Y$는 독립
$$Pr(X=x\cap Y=y)=Pr(X=x)Pr(Y=y)$$
$X_1,\dots,X_k$는 mutually independent: $I\subseteq\{1,\dots,k\}$에 대해
$$Pr\left(\bigcap_{i\in I}(X_i=x_i)\right)=\prod_{i\in I}Pr(X_i=x_i)$$

### Probability Mass Function
$p: \mathbb{R}\rightarrow[0,1]$  
확률변수 $X$에 대한 probability mass function $P_X(x)=Pr(X=x)$

Joint PMF: $P_{X,Y}(x,y)=Pr(X=x,Y=y)$

$Y=g(X)$로 정의된 확률변수 $Y$: $P_Y(y)=\sum_{x\in g^{-1}(y)}Pr(X=x)$

### Expectation
$$E[X]=\sum_i i\cdot Pr(X=i)$$
$$E[g(X)]=\sum_{x}g(x)P_X(x)$$
두 번째 식의 증명은 $Y=g(X)$로 정의하여 할 수 있음

### Unbounded Expectation: St. Petersburg Paradox
tail 나올때까지 동전던지기, $i$번째에 나오면 $2^i$원 획득  
$i$번째에 처음으로 tail 나올확률: $(1/2)^i$  
얻는 돈 $X$의 기댓값: $\sum_{i=1}^\infty(1/2)^i2^i=\infty$

### Linearity of Expectation
$$E[X+Y]=E[X]+E[Y]$$
$$E[cX]=cE[X]$$
$X$와 $Y$가 독립일 필요는 없음

### Jensen's Inequality
$f$가 볼록함수이다
$$f(\lambda x_1+(1-\lambda)x_2)\leq \lambda f(x_1)+(1-\lambda)f(x_2)$$
$f$가 두 번 미분 가능하면 $f$가 볼록 iff $f''\geq 0$

$f$가 볼록함수이면
$$E[f(X)]\geq f(E[X])$$
증명: $f(x)\geq f(\mu)+f'(\mu)(x-\mu)$이므로 좌우변의 기댓값을 취한다

# Lecture 5: Random Variable, Expectation
### Bernoulli Distribution
$p$의 확률로 성공, $1-p$의 확률로 실패  
$Y = \begin{cases} 
	    1 & \text{if success} \\ 
        0 & \text{if failed} 
     \end{cases}$  
$E[Y]=Pr(Y=1)=p$

### Binomial Distribution
$n$개의 Bernoulli 확률변수의 합, 각각 확률 $p$  
$X\sim Bin(n,p)$로 표시  
$$Pr(X=j)=\binom njp^j(1-p)^{n-j}$$
기댓값의 선형성에 의해 $E[X]=np$

### Conditional Expectation
$$E[Y|Z=z]=\sum_yyPr(Y=y|Z=z)$$
Lemma
$$E[X]=\sum_yPr(Y=y)E[X|Y=y]$$
Linearyity
$$E\left[\sum_{i=1}^nX_i|Y=y\right]=\sum_{i=1}^nE[X_i|Y=y]$$
$E[Y|Z]$는 $Z$에 대한 함수로 이해할 수 있다  

Law of Total Expectation
$$E[Y]=E[E[Y|Z]]$$
증명: 위 Lemma에 대입

# Lecture 6: Random Variable, Expectation
### Geometric Distribution
$X\sim \text{Geom(p)}$
$$Pr(X=n)=(1-p)^{n-1}p$$
기하분포는 memoryless
$$Pr(X=n+k|X>k)=Pr(X=n)$$
지금까지의 실패를 잊어먹는 느낌

### Expectation of Geometric Distribution
Lemma: $X$가 음이 아닌 정수 값만을 가진다면
$$E[X]=\sum_{i=1}^\infty Pr(X\geq i)$$
이를 이용하면 기하분포 따르는 확률변수 $X$의 기댓값 $E[X]=1/p$

다른 증명: $E[X]=E[X|X=1]Pr(X=1)+E[X|X>1]Pr(X>1)=p+(1-p)(1+E[X])$

### Coupon Collector's Problem
Lemma: The harmonic number
$$H(n)=\sum_{i=1}^n1/i=\ln n+\Theta(1)$$

문제1: $n$종류의 쿠폰을 모두 모으려 할 때 필요한 쿠폰 수 기댓값  
$X_i$: $i$번째 새로운 종류를 얻는 데 필요한 쿠폰수  
$X_i\sim geom((n-i+1)/n)$, $E[X_i]=n/(n-i+1)$  
$E[X]=\sum_{i=1}^nE[X_i]=nH(n)=n\ln n+\Theta(n)$

문제2: $n$종류의 쿠폰을 $N$번 뽑았을 때 서로 다른 종류의 기댓값  
풀이1: $X_i$를 $i$번 쿠폰이 있으면 1, 없으면 0으로 정의  
$E[X_i]=1-(1-1/n)^N$, $E[X]=n(1-(1-1/n))^N$  
풀이2: $X_N$을 답이라 가정  
$E[X_N]=E[X_{N-1}]+(1-E[X_{N-1}]/n)$의 선형점화식 풀기

### Quicksort
pivot을 uniformly random하게 뽑는다면, 비교 횟수의 기댓값은 $2n\ln n+O(n)$  
$X_{ij}$를 $i$번 원소와 $j$번 원소가 비교되었는지로 정의$(i<j)$  
$i$와 $j$ 사이의 것을 pivot으로 고르게 되면 둘 사이 비교 X  
즉, $E[X_{ij}]=Pr(X_{ij}=1)=\frac2{j-i+1}$  
$E[X]=\sum_{k=2}^n(n+1-k)\frac2k=(n+1)\sum_{k=2}^n\frac2k-2(n-1)=2n\ln n+O(n)$

# Lecture 7: Moments and Deviations
### Markov's Inequality
$X$가 음이 아닌 값만을 갖는다면
$$Pr(X\geq a)\leq \frac{E[X]}a$$
증명: $E[X]$ 식을 $X\geq a$인 부분과 아닌 부분으로 나누어 증명

### Moment and Variance
$X$의 $k$-th moment를 $E[X^k]$로 정의

$X$의 variance(분산)
$$Var[X]=E[(X-E[X])^2]$$

$X, Y$의 Covariance(공분산)
$$Cov(X,Y)=E[(X-E[X])(Y-E[Y])]$$

기댓값의 선형성을 사용하면
$$\begin{aligned}
Var[X]&=E[X^2]-(E[X])^2\\
Var[X+Y]&=Var[X]+Var[Y]+2Cov[X,Y]\\
Cov(X,Y)&=E[XY]-E[X]E[Y]
\end{aligned}$$

$X, Y$가 독립이면
$$\begin{aligned}
E[XY]&=E[X]E[Y]\\
Cov(X,Y)&=0\\
Var[X+Y]&=Var[X]+Var[Y]
\end{aligned}$$

예를 들어, 베르누이 분포의 분산은 $p(1-p)$이므로, 이항분포의 분산은 $np(1-p)$

### Chebyshev's Inequality
$$Pr(|X-E[X]|\geq a)\leq\frac{Var[X]}{a^2}$$
증명: $Pr((X-E[X])^2\geq a^2)$와 같으므로 Markov's Inequality

### Variance of Geometric Distribution
$(1-p)/p^2$  
$$\begin{aligned}
E[X^2]&=Pr(X>1)E[X^2|X>1]+Pr(X=1)E[X^2|X=1]\\
&=(1-p)E[(1+X)^2]+p
\end{aligned}$$

### Coupon Collector's Problem Revisited
$$E[X]=nH(n)$$
$$Var[X]=\sum_{i=1}^nVar[X_i]\leq\sum_{i=1}^n\left(\frac n{n-i+1}\right)^2\leq\frac{\pi^2n^2}6$$
Chebyshev's inequality
$$pr(X\geq 2nH(n))\leq Pr(|X-nH(n)|\geq nH(n))\leq\frac{\pi^2n^2/6}{(nH(n))^2}=O(1/(\log n)^2)$$

### Median and Mean
Median의 정의
$$Pr(X\leq m)\geq1/2, Pr(X\geq m)\geq 1/2$$

Median의 성질  
$m$은 $E[|X-c|]$를 최소화하는 c이다  
cf. $E[X]$는 $E[(X-c)^2]$을 최소화  
증명: $c$가 median이 아니라 가정, $|x-c|-|x-m|$은 항상 $|m-c|$보다 큼을 잘 이용

정리: $|\mu-m|\leq \sigma$
$$\begin{aligned}
|\mu-m|&=|E[X]-m|=|E[X-m]|\\
&\leq E[|X-m|]\\
&\leq E[|X-\mu|]\\
&=E[\sqrt{(X-\mu)^2}]\\
&\leq\sqrt{E[(X-\mu)^2]}\\
&=\sigma
\end{aligned}$$
Jensen's inequality를 2번 씀

# Lectur 8: Chernoff Bound
### Moment Generating Function
$$M_X(t)=E[e^{tX}]$$
성질: $E[X^n]=M_X^{(n)}(0)$

$X$와 $Y$가 독립이면 $M_{X+Y}(t)=M_X(t)M_Y(t)$

Lemma: 여기에 Markov's Inequality를 적용하면
$$Pr(X\geq a)=Pr(e^tX\geq e^ta)\leq\frac{E[e^tX]}{e^{ta}}$$
이게 최소가 되는 $t$를 찾으면 좋을듯  
$t$가 음수인 범위에서는 $Pr(X\leq a)$에 관한 부등식을 얻음

### Poisson Trial
$X=\sum_{i=1}^nX_i$ where $X_i\sim\text{Bernoulli}(p_i)$  
$X_i$들을 Poisson Trials라 부름  
$\mu=E[X]=\sum_{i=1}^nE[X_i]=\sum_{i=1}^np_i$  
$M_{X_i}(t)=p_ie^t+(1-p_i)\leq\exp(p_i(e^t-1))$

### Chernoff Bound
$$Pr(X\geq(1+\delta)\mu)\leq\left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^\mu$$
증명
$$Pr(X\geq(1+\delta)\mu)\leq\frac{E[e^{tX}]}{\exp(t(1+\delta)\mu)}\leq\frac{\exp(\mu(e^t-1))}{\exp(t(1+\delta)\mu)}$$
Let $t=\ln(1+\delta)$

특수 케이스들  
$0<\delta\leq1$
$$Pr(X\geq(1+\delta)\mu)\leq\exp(-\mu\delta^2/3)$$
$R\geq 6\mu$
$$Pr(X\geq R)\leq2^{-R}$$

평균보다 작을 때
$$Pr(X\leq(1-\delta)\mu)\leq\left(\frac{e^{-\delta}}{(1-\delta)^{1-\delta}}\right)^\mu$$
더 간단하지만 약한 형태
$$Pr(X\leq(1-\delta)\mu)\leq\exp(-\mu\delta^2/2)$$

양 쪽 모두에 관한 bound
$$Pr(|X-\mu|\geq\delta\mu)\leq2\exp(-\mu\delta^2/3)$$

### Better Bounds for Special Cases
$Pr(X_i=1)=Pr(X_i=-1)=1/2$, $X=\sum_iX_i$
$$Pr(X\geq a)\leq\exp\left(-\frac{a^2}{2n}\right)$$
증명
$$E[e^{tX_i}]=\frac{e^t+e^{-t}}2\leq\exp\left(\frac{t^2}2\right)$$
$$E[e^tX]\leq\exp\left(\frac{t^2n}2\right)$$
$$Pr(X\geq a)\leq\exp\left(\frac{t^2n}2-ta\right)$$
Let $t=a/n$  
Corollary: $Pr(|X|\geq a)\leq2\exp(-a^2/2n)$

# Lecture 9: Balls and Bins, Poisson
### Set Balancing Example
$i$ property를 갖는 사람과 아닌 사람 수의 차의 최댓값을 최소화  
이것이 $\sqrt{4m\ln n}$보다 클 확률은 $2/n$보다 작다  
하나의 property에 대해 Better bound를 쓰면 $2/n^2$, 이후 union bound  
$i$ property를 갖는 수가 $\sqrt{4m\ln n}$보다 작으면 자명, 크면 bounding

### Hoeffding Bound
Special Case: $E[X_i]=\mu$ and $Pr(a\leq X_i\leq b)=1$
$$Pr\left(\left|\frac1n\sum_iX_i-\mu\right|\geq\epsilon\right)\leq2\exp\left(\frac{-2n\epsilon^2}{(b-a)^2}\right)$$
General Case: $E[X_i]=\mu_i$ and $Pr(a_i\leq X_i\leq b_i)=1$
$$Pr\left(\left|\sum_iX_i-\sum_i\mu_i\right|\geq\epsilon\right)\leq2\exp\left(\frac{-2\epsilon^2}{\sum_i(b_i-a_i)^2}\right)$$

Hoeffding's Lemma: $Pr(a\leq X\leq b)=1$ and $E[X]=0$ then
$$E[e^{\lambda X}]\leq e^{\lambda^2(b-a)^2/8}$$
증명
$$e^{\lambda x}\leq\frac{b-x}{b-a}e^{\lambda a}+\frac{x-a}{b-a}e^{\lambda b}$$
에서 양변의 기댓값 취한 뒤, 이것을 $e^{L(\lambda(b-a))}$로 두고 $L(h)$를 2차 테일러 전개  

Hoeffding Bound의 증명  
$Z_i=X_i-\mu_i$로 잡고 위 Lemma 적용한 뒤 합치고, 적당한 $\lambda$값을 대입

### Balls and Bins
$n$개의 bin에 $m$개의 ball을 무작위로 던지는 상황  
만약 $n\rightarrow\infty$이고, $m/n$을 유지한다면?  
주어진 bin에 몇 개의 공이 있는가? 공이 가장 많은 bin에는 몇 개가 있는가?

### Birthday Paradox Example
사람 $m$명의 생일이 겹치는 쌍이 있을 확률은? 한 해는 $n$일이다  
모두가 생일이 다를 확률은 $\prod_{i=1}^{m-1}(1-i/n)$  
$k$가 $n$에 비해 충분히 작으면 $1-i/n\approx e^{-i/n}$  
즉, 위 확률을 근사하면 $\exp(-\sum_{i=1}^{m-1}i/n)=\exp(-m(m-1)/2n)\approx\exp(-m^2/2n)$  
$m=\sqrt{2n\ln2}$일 때 이 확률이 약 1/2

더 정확하게: 확률 1/2지점은 $\lfloor\sqrt n\rfloor$보단 크고, $\lceil2\sqrt n\rceil$보단 작다  
lower bound: $\binom k2$개 쌍이 모두 달라야 함을 바탕으로 union bound 사용  
upper bound: 반으로 잘라서 나중 $\lceil\sqrt n\rceil$명이 맨처음 $\lceil\sqrt n\rceil$명과 달라야하므로

### Maximum Load
$n$이 충분히 크다면, $n$개의 ball을 $n$개의 bin에 던질 때  
가장 ball이 많은 bin의 ball 수가 $3\ln n/\ln\ln n$보다 클 확률은 $1/n$보다 작다

특정 bin이 $M$개 이상의 ball을 갖는 확률은 $\binom nM(1/n)^M$보다 작다  
이것은 다시 $1/M!$보다 작고, 다시 $(e/M)^M$보다 작다  
Union bound를 쓰면 어떤 bin이 $M$개 이상의 ball을 갖는 확률은 $n(e/M)^M$보다 작다  
$M=3\ln n/\ln\ln n$일 때 이 식을 어떻게 잘 하면 $1/n$ 이하가 된다

# Lecture 10: Poisson
### Poisson Distribution
$X\sim\text{Poisson}(\mu)$임은
$$Pr(X=j)=\frac{e^{-\mu}\mu^j}{j!}$$
$E[X]=\mu$가 된다

왜 이런 식이 나왔는가? Balls and Bins에서 어떤 bin이 $r$개의 ball 얻을 확률은
$$p_r=\binom mr\left(\frac1n\right)^r\left(1-\frac1n\right)^{m-r}=\frac1{r!}\frac{m\dots(m-r+1)}{n^r}\left(1-\frac1n\right)^{m-r}$$
$n$과 $m$을 충분히 키우면
$$p_r\approx\frac1{r!}\left(\frac mn\right)^r\left(1-\frac1n\right)^{m}\approx\frac1{r!}\left(\frac mn\right)^re^{-m/n}$$

### Moment Generating Function for Poisson
$X\sim\text{Poisson}(\mu)$ 이면
$$M_X(t)=\exp(\mu(e^t-1))$$
증명
$$E[e^tX]=\sum_{k=0}^\infty\frac{e^{-\mu}\mu^k}{k!}e^{tk}=e^{-\mu}\sum_{k=0}^\infty\frac{(\mu e^t)^k}{k!}=e^{-\mu}\exp(\mu e^t)$$
미분해보면 $E[X]=\mu$, $Var[X]=\mu$

$X\sim\text{Poisson}(\mu)$이고 $Y\sim\text{Poisson}(\lambda)$이면 $X+Y\sim\text{Poisson}(\mu+\lambda)$  
증명은 MGF를 곱하면됨

MGF가 Chernoff Bound에서 썼던 upper bound와 같으므로, chernoff bound와 동일한 식이 성립

### Limit of Binomial Distribution
$X_n\sim\text{Bin}(n,\lambda/n)$이면
$$\lim_{n\to\infty}Pr(X_n=k)=\frac{e^{-\lambda}\lambda^k}{k!}$$
증명: $e^{-p}(1-p^2)\leq1-p\leq e^{-p}$임을 사용, 샌드위치 정리

# Lecture 11: Poisson
### Notation
$m$개의 공이 $n$개의 바구니로 독립으로,무작위로 던져짐  
$X_i^{(m)}$을 $i$번째 바구니의 공 개수라 하고  
$Y_i^{(m)}$을 $\text{Poisson}(m/n)$ 따르는 확률변수라 한다  
$X_i^{(m)}$들은 독립이 아니고, $Y_i^{(m)}$들은 독립

### Theorem
$\sum_{i=1}^nY_i^{(m)}=k$ 라는 조건이 있는 조건부 분포는 $X_i^{(m)}$들의 분포와 같다  
증명: 그냥 계산하면됨/$Y_i^{(m)}$의 총합이 $\text{Poisson}(m)$ 따름 이용

### Theorem 2
$f:\mathbb{N}^n\to[0,\infty)$ 에 대해
$$E[f(X_1^{(m)},\dots,X_n^{(m)})]\leq e\sqrt mE[f(Y_1^{(m)},\dots,Y_n^{(m)})]$$
증명
$$E[f(Y_1^{(m)},\dots,Y_n^{(m)})]\geq E[f(X_1^{(m)},\dots,X_n^{(m)})]\frac{e^{-m}m^m}{m!}$$
에서 $m!\leq e\sqrt m(m/e)^m$을 이용(적분 사다리꼴 근사와 볼록성)

Corollary: $Y$들로 확률 $p$를 얻었다면 $X$들에 대한 확률은 커야 $e\sqrt mp$

특수 케이스: 좌변이 $m$에 대해 단조면 상수가 2로 줄어듦

### Lower Bound of Maximum Load
$n$개의 공이 $n$개의 바구니에 담길 때, 최댓값이 $\ln n/\ln\ln n$ 보다 클 확률은 적어도 $1-1/n$
$$Pr(Y_i^{(m)}<M)\leq1-Pr(Y_i^{(m)}=M)=1-\frac1{eM!}$$
$Y$들은 독립이므로
$$Pr(\text{maxload}<M)\leq\left(1-\frac1{eM!}\right)^n\leq\exp\left(-\frac n{eM!}\right)$$
이제 대입하고 열심히 뭔가 하면 됨

# Lecture 12: Poisson
### Set Membership
전체집합 $U$와 부분집합 $S=\{s_1,\dots,s_m\}$  
$U$의 원소 $x$가 $S$의 원소인지 결정하기
1. $S$의 원소이면 TRUE를 리턴
2. 아니면 FALSE를 리턴  

예컨대 사용 불가능한 비밀번호인지 확인하기 등

Approximate Set Membership Problem: 아닐 때는 TRUE를 리턴할 수도 있음  
간단한 방법: hashing을 해 겹치는게 있으면 TRUE, 아니면 FALSE  
결과가 $b$비트라고 하면, false positive의 확률은
$$1-\left(1-\frac1{2^b}\right)^m\approx1-\exp\left(-\frac m{2^b}\right)$$
즉 $b$를 $\Omega(\log m)$정도 스케일로 둬야 이 확률이 꽤 작아짐

### Bloom Filter
$k$개의 해시함수 $h_1,\dots,h_k$, $n$종류의 결과값을 내놓음  
길이 $n$의 0/1 배열의 $h_i(s_j)$번째 원소를 1로 만들기  
$x\in S$의 답을 $A[h_i(x)]=1$ for all $i$로 결정하기->false negative는 안일어남  
$k$를 증가시키면 아래의 두 가지 효과가 있다
- 체크를 여러 번 하므로 false positive 확률 감소
- 배열에서 1이 많아지므로 false positive 확률 증가

$m$개의 원소에 대해 배열에 값을 모두 적은 후, 어떤 특정 비트가 0일 확률은
$$\left(1-\frac1n\right)^{km}\approx\exp\left(-\frac{km}n\right)$$
이 확률을 $p$라 놓으면 false positive의 확률은
$$(1-p)^k=(1-\exp(-km/n))^k$$
이것을 $f(k)$라 놓고, $\ln f(k)=g(k)$로 정의하면
$$g(k)=k\ln(1-\exp(-km/n))$$
이것을 미분하여 최소인 점을 찾자
$$g'(k)=\ln(1-\exp(-km/n))+\frac{km}n\frac{\exp(-km/n)}{1-\exp(-km/n)}=\ln(1-p)-\ln p\frac p{1-p}$$
0이 되는 지점은 $p=1/2$, $k=(n/m)\ln2$, 이 점에서
$$f(k)=2^{-k}=\exp\left(-(\ln2)^2\frac nm\right)\approx0.6185^{n/m}$$
false positive 확률이 $c$가 되는 지점은
$$n/m=\frac{\ln(1/c)}{(\ln2)^2}\approx2.081\ln(1/c)$$
실제로 0인 비트의 비율이 $p$인가?(기댓값만 $p$인 것은 아닌가)  
푸아송 분포에 대한 Chernoff Bound와 Poisson approximation을 생각하면
$$Pr(|X-np|\geq\epsilon n)\leq2e\sqrt{mk}\exp(-n\epsilon^2/3p)$$

### Symmetry Breaking
$n$명의 고객의 해시값이 모두 다르게 하기 위해선 몇비트를 써야하는가?  
$\binom n2$개의 조합이 있으므로 간단한 union bound를 쓰면 같은 쌍이 있을 확률은 $n(n-1)/2^{b+1}$보다 작다  
즉, $b>3\log_2n$으로 두면 겹칠 확률이 $1/n$보다 작아짐

### Random Graph Model
$G=(V,E)$에서 $|V|=n$이라 하자/여기서 무작위의 그래프를 얻는법  
- $G_{n,p}$ model: 각 간선을 $p$의 확률로 선택  
- $G_{n,N}$ model: 간선이 $N$개인 subgraph 중 하나를 균일한 확률로 선택  
$G_{n,N}$의 분포는 $G_{n,p}$를 간선이 $N$개 될 때 까지 돌린 결과와 같다

# Lecture 13: Probabilistic Method
무언가의 존재를 그 사건의 확률이 양수임으로 보일 수 있다

### Clique
정점의 부분집합, 근데 모두가 연결된  
$K_k$ clique는 정점이 $k$개인 clique  
Maximum clique: 크기가 가장 큰 clique  
Maximal clique: 더 이상 커질 수 없는 clique

### The Counting Argument
$\binom nk2^{-\binom k2+1}<1$ 이면 $K_n$의 간선들을 단색 $K_k$ subgraph 없게 2가지 색으로 칠할 수 있다  
$K_n$의 간선들을 랜덤한 색으로 칠했다고 가정(1/2씩)  
$K_n$의 clique $\binom nk$개를 나열한 뒤, $A_i$를 $i$번째 clique가 단색인 사건으로 잡자  
$Pr(A_i)=2^{-\binom k2+1}$ 이 된다
$$Pr\left(\bigcup A_i\right)\leq\sum Pr(A_i)=\binom nk2^{-\binom k2+1}<1$$
즉, $Pr(\bigcap \bar{A_i})>0$이므로 단색 clique 없는 색칠이 존재함

### Expectation Argument
$Pr(X\geq\mu)>0$이고 $Pr(X\leq\mu)>0$이다  
이를 이용한 증명: 간선 $m$개인 그래프에서 size가 $m/2$ 이상인 cut이 존재한다  
$X_i$를 $e_i$가 cut-set에 포함되는지 여부로 설정->$E[X_i]=1/2$, $E[X]=m/2$  
이를 이용한 큰 cut 알고리즘 제시: A와 B를 랜덤하게 잡아서 size가 $m/2$보다 클때까지  
1번의 랜덤이 성공할 확률을 $p$라 두면  
$m/2\leq(m/2-1)(1-p)+mp$ 이므로 $p\geq1/(m/2+1)$

# Lecture 14: Probabilistic Method
### MAX-SAT
CNF(Conjunctive Normal Form): OR로 이루어진 clause들이 AND로 묶인 형태  
Literal: positive($x$)와 negative($\bar x$)  
SAT: 주어진 formula가 satisfiable인지 결정하기  
MAX-SAT: 가장 많은 clause를 만족시키기  
$i$번째 clause가 $k_i$개의 literal 가진다면 무작위로 배정했을 때 만족되는 clause의 기댓값은  
$\sum_{i=1}^m(1-2^{-k_i})$  
즉, 이보다 많거나 같은 clause를 만족시키는 배정이 있다

### Derandomization Using Conditional Expectation
확률론적 argument를 deterministic하게 바꾸기  
예시: big-cut algorithm에서 $v_1,\dots,v_n$ 순서로 들어갈 집합을 정한다 결정  
$v_i$가 속하는 집합 이름을 $x_i$로 두자  
$E[C(A,B)|x_1,\dots,x_k]$를 $x_k$까지 결정했을 때 size의 기댓값이라 하자  
우선 $E[C(A,B)|x_1]\geq m/2$임은 자명하다  
이제 $E[C(A,B)|x_1,\dots,x_{k+1}]\geq E[C(A,B)|x_1,\dots,x_k]$가 되도록 해 보자  
즉, 점점 결정할수록 기댓값이 증가하는  
$$E[C(A,B)|x_1,\dots,x_k]=E[C(A,B)|x_1,\dots,x_k,A]/2+E[C(A,B)|x_1,\dots,x_k,B]/2$$
이므로, 둘 중 하나는 좌변보다 크다->이것을 $x_{k+1}$로 정하자  
방법은? $v_1,\dots,v_k$중에 더 적게 속한 쪽으로: $v_{k+2}$ 이후 정점을 포함하는 간선은 아직 확률이 1/2이다  
이 greedy한 알고리즘은 적어도 $m/2$ 이상의 cut size를 보장한다  
