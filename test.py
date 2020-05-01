from scipy import stats
import numpy as np
import matplotlib.pylab as plt

# create some normal random noisy data
ser = 50*np.random.rand() * np.random.normal(10, 10, 100) + 20

# plot normed histogram
plt.hist(ser, normed=True)

# find minimum and maximum of xticks, so we know
# where we should compute theoretical distribution
xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(ser))

# lets try the normal distribution first
m, s = stats.norm.fit(ser) # get mean and standard deviation
pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval
plt.plot(lnspc, pdf_g, label="Norm") # plot it

# exactly same as above
ag,bg,cg = stats.gamma.fit(ser)
pdf_gamma = stats.gamma.pdf(lnspc, ag, bg,cg)
plt.plot(lnspc, pdf_gamma, label="Gamma")

# guess what :)
ab,bb,cb,db = stats.beta.fit(ser)
pdf_beta = stats.beta.pdf(lnspc, ab, bb,cb, db)
plt.plot(lnspc, pdf_beta, label="Beta")

plt.show()