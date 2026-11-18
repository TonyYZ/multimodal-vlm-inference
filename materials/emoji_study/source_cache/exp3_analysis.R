rm(list = ls(all = TRUE))

library(plyr)
library(ggplot2)
library(sciplot)
library(lme4)

####################################
# Demographic data
####################################

header <- names(read.csv("demographic.csv", header = TRUE, nrows = 1, skip = 0))
demographic <- read.csv("demographic.csv", header = FALSE, col.names = header, skip = 1)
demographic <- demographic[2:nrow(demographic),]

length.unique <- function(x){length(unique(x))}

# Confirm number of English speaker participants :

length.unique(demographic$Participant.Private.ID[demographic$Object.ID=="object-5"&demographic$Response=="Yes"])

# 60 native speaker participants

####################################
# Test data
####################################

header <- names(read.csv("task.csv", header = TRUE, nrows = 1, skip = 0))
data <- read.csv("task.csv", header = FALSE, col.names = header, skip = 1)
data <- data[2:nrow(data),]

data <- subset(data, Response.Type=="response")
data <- subset(data, Response != "")

is.numeric(data$Response)
data$Response <- as.numeric(data$Response)
is.numeric(data$Response)

# Check data :

length.unique(data$Spreadsheet..Emoji) # 10 emojis tested
length.unique(data$Spreadsheet..Emoji.Type) # 2 emoji types tested
length.unique(data$Spreadsheet..Condition) # 2 conditions tested

# Rename factors and levels :

data$Condition <- ordered(data$Spreadsheet..Condition, levels=c("Target","Control"), labels=c("Target","Control"))
data$Emoji.Type <- ordered(data$Spreadsheet..Emoji.Type, levels=c("Post", "Pro"), labels=c("Post","Pro"))
data$Emoji <- ordered(data$Spreadsheet..Emoji, levels=c("Erupt","Cigarette","Umbrella","Sprout","Toilet","Graduate","Sleep","Phone","Hatch","Plane"), labels=c("Erupt","Cigarette","Umbrella","Sprout","Toilet","Graduate","Sleep","Phone","Hatch","Plane"))

####################################
# Plot data
####################################

# Plot by environment :

dfc <- ddply(data, c("Participant.Private.ID","Condition","Emoji.Type","Emoji"),
             function(data)c(Response=mean(data$Response)))
dfc1 <- ddply(dfc, c("Condition","Emoji.Type","Emoji"),
              function(data)c(MEAN=mean(data$Response), SE=se(data$Response)))
dfc1 <- ddply(dfc, c("Condition","Emoji.Type"),
              function(data)c(MEAN=mean(data$Response), SE=se(data$Response)))

boxplot.data <- aggregate(Response~Participant.Private.ID+Condition+Emoji.Type, FUN=mean, data=data)

plot <- ggplot(dfc1, aes(x=Emoji.Type, y=MEAN, fill=factor(Condition))) + 
  ggtitle("Endorsement of target and control inferences") + 
  xlab("Emoji type") + ylab("% Endorsement") + 
  geom_bar(width=.9, position="dodge", stat="identity", alpha=.75) +
  scale_y_continuous(limits=c(NA,NA), breaks=c(0,25,50,75,100)) +
  theme_bw() + scale_fill_manual(name="Inference type", 
                                 breaks=c("Target","Control"), labels=c("Target","Control"),
                                 values=c("forestgreen","orangered")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_errorbar(aes(ymin=MEAN-SE, ymax=MEAN+SE), width=.5, position=position_dodge(.9), col="black")
p <- 
  plot +
  geom_dotplot(
    data = boxplot.data, 
    aes(y=Response),
    col="black", alpha=.5, 
    binaxis='y', stackdir='center', binwidth = 1,
    position = "dodge"
  ) 
p
quartz(width=6,height=3.5)
print(p)

####################################
# Linear regression models
####################################

# Test for effect of condition (target vs. control) :

post$ConditionC <- ifelse(post$Condition=="Target", -.5, .5)

post.lm = lmer(Response~ConditionC+(1+ConditionC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=post)
post.lm_ConditionC = update(post.lm, .~.-ConditionC)
anova(post.lm, post.lm_ConditionC) # X^2(1)=212, p<.001 ***