library(UsingR)
library(ggplot2)
library(gridExtra)
library(grid)

getAvgRewards <- function(name){
  name <- paste(name,"averageReward.txt", sep="/")
  strings <- names(read.csv(name, check.names=FALSE))
  
  return(as.numeric(strings[1:200]))
}

getMatrixData <- function(ph.name, size=12){
  values = as.character(0:(size-1))
  
  n = 1
  mat = list()
  
  for(i in values){
    file.name <- gsub("X",i,ph.name)
    file.data <- getAvgRewards(file.name)
    
    mat[[n]] <- file.data
    
    n <- n+1
  }
  return(mat)
}

averageSet <- function(data){
  
  mu = c(rep.int(0,200))
  for(i in 1:length(data)){
    mu <- mu + data[[i]]
  }
  
  return(mu/length(data))
  
}

makeTable <- function(data){
  table <- as.data.frame(table(t=1:200))
  table$t <- as.numeric(table$t)
  
  table$r <- data
  
  return(table)
}

getPlot <- function(data, max=5, title=""){
  
  table <- makeTable(data)
  
  p <- ggplot(data=table, aes(x=table$t, y=table$r))
  p <- p + geom_line() + labs(x= "time-step", y="average reward") + ylim(-0.5, max) + ggtitle(title) + theme(
    plot.title = element_text(color="black", size=18, face="bold.italic"))
  
  return(p)
  
}

#Generating plots (move in the folder containing all tests results)
test.mat <- getMatrixData("med_exp_X_1")

p.1 = getPlot(test.mat[[1]])
p.2 = getPlot(test.mat[[2]])
p.3 = getPlot(test.mat[[3]])
p.4 = getPlot(test.mat[[4]])
p.5 = getPlot(test.mat[[5]])
p.6 = getPlot(test.mat[[6]])
p.7 = getPlot(test.mat[[7]])
p.8 = getPlot(test.mat[[8]])
p.9 = getPlot(test.mat[[9]])
p.10 = getPlot(test.mat[[10]])
p.11 = getPlot(test.mat[[11]])
p.12 = getPlot(test.mat[[12]])

title <- textGrob("Baseline Performance in 7x7 maze",gp=gpar(fontsize=20))
grid.arrange(p.1,p.2,p.3,p.4,p.5,p.6,p.7,p.8,p.9,p.10,p.11,p.12, nrow=2, top=title)

title <- "Average Baseline Performance in 5x5 maze"
getPlot(averageSet(test.mat), title=title)
getPlot(averageSet(test.mat), max=1, title=title)

#Performing paired t-tests
#small maze
data.base <- getMatrixData("run_exp_X_1")
data.hier <- getMatrixData("run_h_exp_X_1")

avg.base <- averageSet(data.base)
avg.hier <- averageSet(data.hier)

t.test(avg.base, avg.hier, paired=TRUE, conf.level = 0.95)

#medium maze
data.base.med <- getMatrixData("med_exp_X_1")
data.hier.med <- getMatrixData("med_h_exp_X_1")

avg.base.med <- averageSet(data.base.med)
avg.hier.med <- averageSet(data.hier.med)

t.test(avg.base.med, avg.hier.med, paired=TRUE, conf.level = 0.95)
 
