# install.packages('EasyABC')
library("rstudioapi")
setwd(dirname(getActiveDocumentContext()$path))
library(EasyABC)
set.seed(123)
Mixture<-function(theta){ c(abs(theta[1]) + rnorm(1,0,sd = sqrt(0.05)),abs(theta[2]) + rnorm(1,0,sd = sqrt(0.05))) }
prior=list(c("normal",0,1),c("normal",0,1))
y_obs=c(1.5,1.5)
ABC_Marjoram<-ABC_mcmc(method="Marjoram", model=Mixture, prior=prior,n_rec=100,
                       n_between_sampling=1,
                       summary_stat_target=y_obs)
Data <- as.data.frame(ABC_Marjoram$param)
# 假设 Data 是您的数据框
# 转换为矩阵以避免列名
names(Data) <- rep("", ncol(Data))
# 写入 CSV 文件，不包含列名
write.csv(Data, file = 'easyabc_Marjoram.csv', row.names = FALSE)


# library(ggplot2)
# Data <- as.data.frame(ABC_Marjoram$param)
# colnames(Data) <- c('theta1','theta2')
# density_plot <- ggplot(Data,aes(y=theta2,x=theta1))+theme_classic() +
#   stat_density2d(aes(color = ..level..), size = 0.5) +
#   scale_y_continuous(name = expression(theta[2]))+
#   scale_x_continuous(name = expression(theta[1]))+
#   theme(legend.position ='none') +
#   theme(axis.text = element_text(size=10),axis.title.x =element_text(size=12),axis.title.y =element_text(size=12))
# trace_plot <- ggplot(data=Data[30001:40000,],aes(x=theta1,y=theta2,color='red'))+
#   geom_count( show.legend = FALSE) +  # 画点，大小与点出现的次数成正比
#   geom_path(size = 0.5,color='lightgrey') +
#   theme_classic() +scale_x_continuous(name = expression(theta[1]))+
#   scale_y_continuous(name = expression(theta[2]))+
#   theme(axis.text = element_text(size=10),axis.title.x =element_text(size=12),axis.title.y =element_text(size=12))
