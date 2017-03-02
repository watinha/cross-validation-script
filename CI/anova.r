fscore <- read.table('fscore.csv', header=TRUE, sep=';')
anova <- aov(F.Score ~ Algorithm, fscore)
summary(anova)
tuk <- TukeyHSD(anova, ordered=T)
tuk

png(file="confidence-intervals.png", height=900, width=800)
par(mar=c(5.1, 15, 5.1, 2))
plot(tuk, las=1)
title(xlab="Difference in F-Score means according to the machine learning approach used", line=2)

print(tuk, digits=20)
