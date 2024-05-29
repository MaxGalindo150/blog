---
title: "NSGA-II"
excerpt: "This project is to implement the NSGA-II algorithm"
collection: portfolio
---

 NSGA-II implementation results
---------------------------------

To test my implementation of the NSGA-II algorithm, I took some of the problems solved in the original paper [1] and used the [pymoo](https://pymoo.org/problems/test_problems.html) library to obtain those problems. The source code of this implementation can be found in my [GitHub repository](https://github.com/MaxGalindo150/Reinfircement_Learning_Porjects/tree/master/Multi-Objective).

ZDT
-----
The ZDT problem suite is based on the construction process:

$$min f_{1}(x)$$

$$min f_{2}(x) = g(x)h(f_{1}(x),g(x))$$

where two objective have to be minimized. The function $$g(x)$$ can be considered as the function for convergenece and usually $$g(x)=1$$ holds for pareto-optimal solutions (except for ZDT5).

ZTD1
------

This is a $$30$$-variable problem ($$n=30$$) with a convex Pareto-optimal set:

**Definition**: 

$$f_{1} = x_{1}$$

$$g(x) = 1 + \frac{9}{n-1} \sum_{i=2}^{n}x_{i}$$

$$h(f_{1},g)=1-\sqrt{f_{1}/g}$$

Where $$0\leq x_{i} \leq 1$$ $$i = 1,...,n$$

**Optimum**

$$0\leq x_{i}^{*} \leq 1$$ and $$x_{i}^{*}=0$$ for $$i = 2,...,n$$


| ![nondom](../../images/ZDT1.png) | 
|:--:| 
| Figure 1: *Pareto front found by NSGA-II for ZDT1* |

ZTD2
------

This is a $$30$$-variable problem ($$n=30$$) with a convex Pareto-optimal set:

**Definition**: 

$$f_{1} = x_{1}$$

$$g(x) = 1 + \frac{9}{n-1} \sum_{i=2}^{n}x_{i}$$

$$h(f_{1},g)=1-(f_{1}/g)^{2}$$

Where $$0\leq x_{i} \leq 1$$ $$i = 1...,n$$

**Optimum**

$$0\leq x_{i}^{*} \leq 1$$ and $$x_{i}^{*}=0$$ for $$i = 2,...,n$$


| ![nondom](../../images/zdt2.png) | 
|:--:| 
| Figure 2: *Pareto front found by NSGA-II for ZDT2* |

ZTD3
------

This is also a $$30$$-variable problem ($$n=30$$) with a number of disconnected Pareto-optimal fronts:

**Definition**: 

$$f_{1} = x_{1}$$

$$g(x) = 1 + \frac{9}{n-1} \sum_{i=2}^{n}x_{i}$$

$$h(f_{1},g)=1-\sqrt{f_{1}/g} - (f_{1}/g)sin(10\pi f_{1})$$

Where $$0\leq x_{i} \leq 1$$ $$i = 1,...,n$$

**Optimum**

$$0\leq x_{1}^{*} \leq 0.0830$$

$$0.1822\leq x_{1}^{*} \leq 0.2577$$

$$0.4093\leq x_{1}^{*} \leq 0.4538$$

$$0.6183\leq x_{1}^{*} \leq 0.6525$$

$$0.8233\leq x_{1}^{*} \leq 0.8518$$

$$x_{i}=0$$ for $$i=2,...,n$$ 


| ![nondom](../../images/zdt3.png) | 
|:--:| 
| Figure 3: *Pareto front found by NSGA-II for ZDT3* |

ZTD5
------

In ZDT5 in variables are decodec by bitsrings. At all $$11$$ discrete variables are used, where $$x_{1}$$ is represented by $$30$$ bits and the rest $$x_{2}$$ to $x_{11}$ by $$5$$ bits each. The function $$u(x)$$ does nothing else than count the number of $$1$$ of the corresponding variable. Also, note that the objective function is deceptive, because the values of $$v(u(xi))$$ are decreasing with the number of $$1’$$s, but has its minimum when all variables are indeed $$1$$.

**Definition**: 

$$f_{1} = x_{1} + u(x_{1})$$

$$g(x) =\sum_{i=2}^{n}v(u(x_{i}))$$

$$v(u(x_{i})) = 
\begin{cases} 
   2 + u(x_{i}) & \text{if } u(x_{i}) < 5 \\
   1 & \text{if } u(x_{i}) = 5 
\end{cases}
$$

$$h(f_{1},g)=1/f_{1}(x_{i})$$

**Optimum**

$$0\leq u(x_{i}^{*}) \leq 30$$ and $$u(x_{i}^{*})=5$$ for $$i = 2,...,n$$


| ![nondom](../../images/zdt5.png) | 
|:--:| 
| Figure 4: *Pareto front found by NSGA-II for ZDT5* |

DTLZ2
------

This function can also be used to investigate an MOEA’s ability to scale up its performance in large number of objectives. Like in DTLZ1, for $$M>3$$, the Pareto-optimal solutions must lie inside the first octant of the unit sphere in a three-objective plot with $$f_{M}$$ as one of the axes. Since all Pareto-optimal solutions require to satisfy $$\sum_{m=1}^{M}f_{m}^{2}=1$$ , the difference between the left term with the obtained solutions and one can be used as a metric for convergence as well. Besides the suggestions given in DTLZ1, the problem can be made more difficult by replacing each variable $$x_{i}$$ (for $$i=1$$ to $$(M−1)$$) with the mean value of $$p$$ variables: $$x_{i}=\frac{1}{p}\sum_{k=(i−1)p + 1}^{ip}x_{k}$$.

**Definition**: 

$$\begin{array}{ll}\text { Min. } & f_1(\mathbf{x})=\left(1+g\left(\mathbf{x}_M\right)\right) \cos \left(x_1 \pi / 2\right) \cdots \cos \left(x_{M-2} \pi / 2\right) \cos \left(x_{M-1} \pi / 2\right), \\ \text { Min. } & f_2(\mathbf{x})=\left(1+g\left(\mathbf{x}_M\right)\right) \cos \left(x_1 \pi / 2\right) \cdots \cos \left(x_{M-2} \pi / 2\right) \sin \left(x_{M-1} \pi / 2\right), \\ \text { Min. } & f_3(\mathbf{x})=\left(1+g\left(\mathbf{x}_M\right)\right) \cos \left(x_1 \pi / 2\right) \cdots \sin \left(x_{M-2} \pi / 2\right), \\ \vdots & \vdots \\ \text { Min. } & f_M(\mathbf{x})=\left(1+g\left(\mathbf{x}_M\right)\right) \sin \left(x_1 \pi / 2\right), \\ \text { with } & g\left(\mathbf{x}_M\right)=\sum_{x_i \in \mathbf{x}_M}\left(x_i-0.5\right)^2, \\ & 0 \leq x_i \leq 1, \quad \text { for } i=1,2, \ldots, n .\end{array}$$

The $$x_{M}$$ vector is constructed with $$k=n−M+1$$ variables.

**Optimum**

The Pareto-optimal solutions corresponds to $$x_{i}=0.5$$ for all $$x_{i}\in x_{M}$$.


| ![nondom](../../images/dtlz2.png) | 
|:--:| 
| Figure 5: *Pareto front found by NSGA-II for DTLZ2* |


Discussion
-------

The results presented above can be compared with the results obtained in the original article [1] or with the results presented by the [pymoo](https://pymoo.org/problems/test_problems.html) library. The algorithm produces good results, but there is room for improvement. This could be achieved by enhancing the binary tournament, the crossover and mutation operators, as well as refining the initialization of the first elements of our population.

I have implemented this algorithm to solve some reinforcement learning problems using the [MO-Gymnasium API](https://mo-gymnasium.farama.org/index.html). We will discuss these results in another section."

References
======
[1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002, April). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2).



